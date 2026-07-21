# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Normalization for the external vLLM call and KV-cache contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig
from models.common.llm_runtime.paged_kv_cache import torch_dtype_for_ttnn

_IGNORED_VLLM_KWARGS = frozenset(
    {
        "page_tables_per_layer",
        "prompt_tokens",
        "output_tokens",
        "slot_remap",
        "rope_deltas_all_users",
    }
)


class VLLMAdapter:
    """Convert vLLM-facing calls into the configured runtime call surface.

    The adapter owns no TT tensors or model/runtime resources. Model-specific
    construction supplies the already-derived KV shape and dtype expectations.
    """

    def __init__(
        self,
        *,
        trace_config: TraceConfig,
        paged_kv_cache_config: PagedKVCacheConfig,
        expected_num_layers: int,
        expected_kv_heads_per_device: int | None = None,
        expected_head_dim: int | None = None,
        model_kv_cache_dtype: Any | Sequence[Any],
    ) -> None:
        self.trace_config = trace_config
        self.paged_kv_cache_config = paged_kv_cache_config
        self.expected_num_layers = int(expected_num_layers)
        self.expected_kv_heads_per_device = (
            None if expected_kv_heads_per_device is None else int(expected_kv_heads_per_device)
        )
        self.expected_head_dim = None if expected_head_dim is None else int(expected_head_dim)
        if model_kv_cache_dtype is None:
            raise TypeError("model_kv_cache_dtype must be supplied from model metadata")
        self._model_kv_cache_dtypes = _as_dtype_tuple(model_kv_cache_dtype)

        if self.expected_num_layers <= 0:
            raise ValueError("expected_num_layers must be positive")
        if int(paged_kv_cache_config.block_size) <= 0:
            raise ValueError("PagedKVCacheConfig.block_size must be positive")
        if int(paged_kv_cache_config.max_num_blocks) <= 0:
            raise ValueError("PagedKVCacheConfig.max_num_blocks must be positive")
        if not isinstance(trace_config.prefill_enabled, bool) or not isinstance(trace_config.decode_enabled, bool):
            raise TypeError("TraceConfig prefill_enabled/decode_enabled must be bool")
        if len(self._model_kv_cache_dtypes) not in (1, self.expected_num_layers):
            raise ValueError("model_kv_cache_dtype must be uniform or contain one dtype per model layer")

        uniform_model_dtype = _uniform_dtype(self._model_kv_cache_dtypes)
        if uniform_model_dtype is not None and paged_kv_cache_config.dtype != uniform_model_dtype:
            raise ValueError(
                "PagedKVCacheConfig.dtype does not match the model-owned KV cache dtype: "
                f"{paged_kv_cache_config.dtype!r} != {uniform_model_dtype!r}"
            )

    def normalize_prefill(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize legacy prefill (tokens, page_table) calls."""

        normalized = _bind_positional(args, kwargs, ("tokens", "page_table"), "prefill")
        self._drop_ignored_kwargs(normalized)
        self._validate_trace_hint(normalized, operation="prefill")
        _require_arguments(normalized, ("tokens", "page_table"), "prefill")
        _normalize_tensor(normalized, "tokens", torch.long)
        _normalize_tensor(normalized, "page_table", torch.int32)
        _normalize_tensor(normalized, "prompt_lens", torch.long)
        _normalize_tensor(normalized, "start_pos", torch.long)
        return normalized

    def normalize_decode(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize legacy decode (tokens, start_pos, page_table) calls."""

        normalized = _bind_positional(args, kwargs, ("tokens", "start_pos", "page_table"), "decode")
        self._drop_ignored_kwargs(normalized)
        self._validate_trace_hint(normalized, operation="decode")
        _require_arguments(normalized, ("tokens", "start_pos", "page_table"), "decode")
        _normalize_tensor(normalized, "tokens", torch.long)
        _normalize_tensor(normalized, "start_pos", torch.long)
        _normalize_tensor(normalized, "page_table", torch.int32)

        tokens = normalized["tokens"]
        if tokens.ndim == 2 and tokens.shape[-1] == 1:
            normalized["tokens"] = tokens.reshape(-1)
        return normalized

    def validate_trace_hint(self, operation: str, enable_trace: bool) -> None:
        """Validate a dynamic trace choice against the static capability ceiling."""

        if not isinstance(enable_trace, bool):
            raise TypeError("enable_trace must be bool")
        if operation == "prefill":
            configured = self.trace_config.prefill_enabled
        elif operation == "decode":
            configured = self.trace_config.decode_enabled
        else:
            raise ValueError(f"Unknown trace operation {operation!r}")
        if enable_trace and not configured:
            raise ValueError(
                f"enable_trace={enable_trace} for {operation} disagrees with static "
                f"TraceConfig policy ({configured})"
            )

    def resolve_legacy_kv_cache_config(
        self,
        kv_cache_shape: Sequence[int],
        dtype: torch.dtype,
        num_layers: int,
    ) -> PagedKVCacheConfig:
        """Validate vLLM's legacy KV spec and return a resolved frozen config."""

        shape = tuple(int(dim) for dim in kv_cache_shape)
        if len(shape) != 4:
            raise ValueError(f"KV cache shape must have rank 4, got {shape}")

        num_blocks, kv_heads, block_size, head_dim = shape
        config = self.paged_kv_cache_config
        if num_blocks <= 0:
            raise ValueError("KV cache num_blocks must be positive")
        if num_blocks > int(config.max_num_blocks):
            raise ValueError(f"KV cache num_blocks={num_blocks} exceeds max_num_blocks={config.max_num_blocks}")
        if block_size != int(config.block_size):
            raise ValueError(
                f"vLLM KV block size {block_size} does not match configured block size {config.block_size}"
            )
        if self.expected_kv_heads_per_device is not None and kv_heads != self.expected_kv_heads_per_device:
            raise ValueError(
                f"vLLM KV heads {kv_heads} do not match model-derived KV heads " f"{self.expected_kv_heads_per_device}"
            )
        if self.expected_head_dim is not None and head_dim != self.expected_head_dim:
            raise ValueError(
                f"vLLM KV head dimension {head_dim} does not match model-derived head dimension {self.expected_head_dim}"
            )
        if int(num_layers) != self.expected_num_layers:
            raise ValueError(
                f"vLLM KV layer count {num_layers} does not match model-derived layer count {self.expected_num_layers}"
            )

        _validate_vllm_torch_dtype(dtype, self._model_kv_cache_dtypes)

        configured_num_blocks = getattr(config, "num_blocks", None)
        if configured_num_blocks is not None and int(configured_num_blocks) != num_blocks:
            raise ValueError(
                f"PagedKVCacheConfig is already resolved to {configured_num_blocks} blocks; "
                f"vLLM requested {num_blocks}"
            )
        if not dataclasses.is_dataclass(config):
            raise TypeError("PagedKVCacheConfig must be a dataclass for immutable capacity resolution")
        return dataclasses.replace(config, num_blocks=num_blocks)

    @staticmethod
    def _drop_ignored_kwargs(kwargs: dict[str, Any]) -> None:
        for key in _IGNORED_VLLM_KWARGS:
            kwargs.pop(key, None)

    def _validate_trace_hint(self, kwargs: dict[str, Any], *, operation: str) -> None:
        if "enable_trace" not in kwargs:
            return
        self.validate_trace_hint(operation, kwargs["enable_trace"])


def _bind_positional(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    names: tuple[str, ...],
    operation: str,
) -> dict[str, Any]:
    if len(args) > len(names):
        raise TypeError(f"{operation} accepts at most {len(names)} positional arguments, got {len(args)}")
    normalized = dict(kwargs)
    for name, value in zip(names, args):
        if name in normalized:
            raise TypeError(f"{operation} got multiple values for argument {name!r}")
        normalized[name] = value
    return normalized


def _require_arguments(kwargs: Mapping[str, Any], names: tuple[str, ...], operation: str) -> None:
    missing = [name for name in names if name not in kwargs]
    if missing:
        raise TypeError(f"{operation} missing required arguments: {', '.join(missing)}")


def _normalize_tensor(kwargs: dict[str, Any], name: str, dtype: torch.dtype) -> None:
    value = kwargs.get(name)
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        kwargs[name] = value.to(dtype=dtype)
    else:
        kwargs[name] = torch.as_tensor(value, dtype=dtype)


def _as_dtype_tuple(dtype: Any | Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(dtype, (list, tuple)):
        if not dtype:
            raise ValueError("model_kv_cache_dtype cannot be empty")
        return tuple(dtype)
    return (dtype,)


def _uniform_dtype(dtypes: tuple[Any, ...]) -> Any | None:
    first = dtypes[0]
    return first if all(dtype == first for dtype in dtypes[1:]) else None


def _torch_surrogate(device_dtype: Any) -> torch.dtype:
    if isinstance(device_dtype, torch.dtype):
        return device_dtype
    return torch_dtype_for_ttnn(device_dtype)


def _validate_vllm_torch_dtype(dtype: torch.dtype, model_dtypes: tuple[Any, ...]) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"vLLM KV dtype must be a torch.dtype, got {type(dtype).__name__}")
    expected = {_torch_surrogate(model_dtype) for model_dtype in model_dtypes}
    if dtype not in expected or len(expected) != 1:
        expected_names = ", ".join(sorted(str(item) for item in expected))
        raise ValueError(f"vLLM KV dtype {dtype} does not match model-owned dtype surrogate {expected_names}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Normalization for the external vLLM call and KV-cache contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VLLMAdapterConfig:
    """Fully resolved static policy for the vLLM compatibility boundary."""

    trace: TraceConfig
    paged_kv_cache: PagedKVCacheConfig
    expected_num_layers: int
    expected_kv_heads_per_device: int | None
    expected_head_dim: int | None
    model_kv_cache_dtypes: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_resolved_adapter_config(self)

    @classmethod
    def resolve(
        cls,
        *,
        trace: TraceConfig,
        paged_kv_cache: PagedKVCacheConfig,
        expected_num_layers: int,
        expected_kv_heads_per_device: int | None = None,
        expected_head_dim: int | None = None,
        model_kv_cache_dtype: Any | Sequence[Any],
    ) -> "VLLMAdapterConfig":
        if type(trace) is not TraceConfig:
            raise TypeError("trace must be a TraceConfig")
        if type(paged_kv_cache) is not PagedKVCacheConfig:
            raise TypeError("paged_kv_cache must be a PagedKVCacheConfig")

        resolved_num_layers = _resolve_positive_int("expected_num_layers", expected_num_layers)
        resolved_kv_heads = _resolve_optional_positive_int("expected_kv_heads_per_device", expected_kv_heads_per_device)
        resolved_head_dim = _resolve_optional_positive_int("expected_head_dim", expected_head_dim)

        if model_kv_cache_dtype is None:
            raise TypeError("model_kv_cache_dtype must be supplied from model metadata")
        if isinstance(model_kv_cache_dtype, Sequence) and not isinstance(model_kv_cache_dtype, (str, bytes)):
            model_kv_cache_dtypes = tuple(model_kv_cache_dtype)
        else:
            model_kv_cache_dtypes = (model_kv_cache_dtype,)

        return cls(
            trace=trace,
            paged_kv_cache=paged_kv_cache,
            expected_num_layers=resolved_num_layers,
            expected_kv_heads_per_device=resolved_kv_heads,
            expected_head_dim=resolved_head_dim,
            model_kv_cache_dtypes=model_kv_cache_dtypes,
        )


class VLLMAdapter:
    """Convert vLLM-facing calls into the configured runtime call surface.

    `Llama3Generator` calls `normalize_prefill` or
    `normalize_decode` before choosing an eager or traced execution
    target. The normalized dictionary contains typed host tensors and an
    explicit Boolean trace choice. During vLLM initialization,
    `resolve_legacy_kv_cache_config` validates the external cache shape
    and returns the one allowed capacity-resolved runtime configuration.

    The adapter owns no TT tensors or model/runtime resources. Model-specific
    construction supplies the already-derived KV shape and dtype expectations.
    """

    def __init__(self, config: VLLMAdapterConfig) -> None:
        if not isinstance(config, VLLMAdapterConfig):
            raise TypeError("config must be a VLLMAdapterConfig")
        self.config = config

    # Public API

    def normalize_prefill(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize legacy prefill (tokens, page_table) calls."""

        normalized = _bind_positional(args, kwargs, ("tokens", "page_table"), "prefill")
        self._drop_ignored_kwargs(normalized)
        self._validate_trace_selection(normalized, operation="prefill")
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
        self._validate_trace_selection(normalized, operation="decode")
        _require_arguments(normalized, ("tokens", "start_pos", "page_table"), "decode")
        _normalize_tensor(normalized, "tokens", torch.long)
        _normalize_tensor(normalized, "start_pos", torch.long)
        _normalize_tensor(normalized, "page_table", torch.int32)

        tokens = normalized["tokens"]
        if tokens.ndim == 2 and tokens.shape[-1] == 1:
            normalized["tokens"] = tokens.reshape(-1)
        return normalized

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
        config = self.config.paged_kv_cache
        if num_blocks <= 0:
            raise ValueError("KV cache num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("KV cache block_size must be positive")
        if (
            self.config.expected_kv_heads_per_device is not None
            and kv_heads != self.config.expected_kv_heads_per_device
        ):
            raise ValueError(
                f"vLLM KV heads {kv_heads} do not match model-derived KV heads "
                f"{self.config.expected_kv_heads_per_device}"
            )
        if self.config.expected_head_dim is not None and head_dim != self.config.expected_head_dim:
            raise ValueError(
                f"vLLM KV head dimension {head_dim} does not match model-derived head dimension "
                f"{self.config.expected_head_dim}"
            )
        if int(num_layers) != self.config.expected_num_layers:
            raise ValueError(
                f"vLLM KV layer count {num_layers} does not match model-derived layer count "
                f"{self.config.expected_num_layers}"
            )

        _validate_vllm_torch_dtype(dtype, self.config.model_kv_cache_dtypes)

        configured_num_blocks = config.num_blocks
        if configured_num_blocks is not None and int(configured_num_blocks) != num_blocks:
            raise ValueError(
                f"PagedKVCacheConfig is already resolved to {configured_num_blocks} blocks; "
                f"vLLM requested {num_blocks}"
            )
        return dataclasses.replace(
            config,
            block_size=block_size,
            max_num_blocks=num_blocks,
            num_blocks=num_blocks,
        )

    # Private implementation

    @staticmethod
    def _drop_ignored_kwargs(kwargs: dict[str, Any]) -> None:
        for key in _IGNORED_VLLM_KWARGS:
            kwargs.pop(key, None)

    def _validate_trace_selection(self, kwargs: dict[str, Any], *, operation: str) -> None:
        if "enable_trace" not in kwargs:
            raise TypeError(f"{operation} requires an explicit enable_trace boolean")
        enable_trace = kwargs["enable_trace"]
        if not isinstance(enable_trace, bool):
            raise TypeError("enable_trace must be bool")
        if operation == "prefill":
            configured = self.config.trace.prefill_enabled
        elif operation == "decode":
            configured = self.config.trace.decode_enabled
        else:
            raise ValueError(f"Unknown trace operation {operation!r}")
        if enable_trace and not configured:
            raise ValueError(
                f"enable_trace={enable_trace} for {operation} disagrees with static "
                f"TraceConfig policy ({configured})"
            )


def _resolve_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if resolved <= 0 or resolved != value:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _resolve_optional_positive_int(name: str, value: Any) -> int | None:
    return None if value is None else _resolve_positive_int(name, value)


def _validate_resolved_adapter_config(config: VLLMAdapterConfig) -> None:
    if type(config.trace) is not TraceConfig:
        raise TypeError("trace must be a TraceConfig")
    if type(config.paged_kv_cache) is not PagedKVCacheConfig:
        raise TypeError("paged_kv_cache must be a PagedKVCacheConfig")
    _require_resolved_positive_int("expected_num_layers", config.expected_num_layers)
    if config.expected_kv_heads_per_device is not None:
        _require_resolved_positive_int("expected_kv_heads_per_device", config.expected_kv_heads_per_device)
    if config.expected_head_dim is not None:
        _require_resolved_positive_int("expected_head_dim", config.expected_head_dim)
    if not isinstance(config.model_kv_cache_dtypes, tuple):
        raise TypeError("model_kv_cache_dtypes must be a tuple")
    if not config.model_kv_cache_dtypes:
        raise ValueError("model_kv_cache_dtypes cannot be empty")
    if len(config.model_kv_cache_dtypes) not in (1, config.expected_num_layers):
        raise ValueError("model_kv_cache_dtypes must be uniform or contain one dtype per model layer")

    first_dtype = config.model_kv_cache_dtypes[0]
    uniform_model_dtype = (
        first_dtype if all(dtype == first_dtype for dtype in config.model_kv_cache_dtypes[1:]) else None
    )
    if uniform_model_dtype is not None and config.paged_kv_cache.dtype != uniform_model_dtype:
        raise ValueError(
            "PagedKVCacheConfig.dtype does not match the model-owned KV cache dtype: "
            f"{config.paged_kv_cache.dtype!r} != {uniform_model_dtype!r}"
        )


def _require_resolved_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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


def _validate_vllm_torch_dtype(dtype: torch.dtype, model_dtypes: tuple[Any, ...]) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"vLLM KV dtype must be a torch.dtype, got {type(dtype).__name__}")
    expected = {
        model_dtype if isinstance(model_dtype, torch.dtype) else torch_dtype_for_ttnn(model_dtype)
        for model_dtype in model_dtypes
    }
    if dtype not in expected or len(expected) != 1:
        expected_names = ", ".join(sorted(str(item) for item in expected))
        raise ValueError(f"vLLM KV dtype {dtype} does not match model-owned dtype surrogate {expected_names}")

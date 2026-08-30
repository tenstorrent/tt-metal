# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Normalization for the external vLLM call and KV-cache contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import torch

from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig
from models.common.llm_runtime.paged_kv_cache import torch_dtype_for_ttnn

# These plugin fields are meaningful to other model families, but the registered
# text-only TTTv2 paths do not implement their hybrid-cache or mRoPE semantics.
# Sampling history and slot-remap fields are intentionally *not* ignored: they
# carry request-owned penalty and RNG lifecycle state through the common runtime.
_IGNORED_VLLM_KWARGS = frozenset(
    {
        "page_tables_per_layer",
        "rope_deltas_all_users",
    }
)
_SAMPLING_STATE_VLLM_KWARGS = frozenset({"prompt_tokens", "output_tokens", "slot_remap"})


class _NormalizedPrefillRequiredKwargs(TypedDict):
    tokens: torch.Tensor  # ↓ Core request
    page_table: torch.Tensor
    prompt_lens: torch.Tensor | None  # ↓ Sequence metadata
    start_pos: torch.Tensor | None
    empty_slots: Sequence[int] | None  # ↓ Lane routing
    kv_cache: Any  # ↓ Borrowed resources
    sampling_params: Any  # ↓ Sampling


class NormalizedPrefillKwargs(_NormalizedPrefillRequiredKwargs, total=False):
    prompt_tokens: Any  # ↓ Request-owned sampling state
    output_tokens: Any
    slot_remap: Any


class _NormalizedDecodeRequiredKwargs(TypedDict):
    tokens: torch.Tensor  # ↓ Core request
    start_pos: torch.Tensor
    page_table: torch.Tensor
    kv_cache: Any  # ↓ Borrowed resources
    sampling_params: Any  # ↓ Sampling
    reset_batch: bool  # ↓ State transition


class NormalizedDecodeKwargs(_NormalizedDecodeRequiredKwargs, total=False):
    prompt_tokens: Any  # ↓ Request-owned sampling state
    output_tokens: Any
    slot_remap: Any


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
    target. Each call returns typed host tensors and its validated Boolean
    trace choice separately. During vLLM initialization,
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

    def normalize_prefill(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        prompt_lens: Sequence[int] | torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        compatibility_kwargs: Mapping[str, Any] | None = None,  # ↓ Compatibility
    ) -> tuple[NormalizedPrefillKwargs, bool]:
        """Normalize one explicit prefill request and return trace intent separately."""

        self._validate_compatibility_kwargs(compatibility_kwargs, operation="prefill")
        self._validate_trace_selection(enable_trace, operation="prefill")
        normalized: NormalizedPrefillKwargs = {
            "tokens": tokens,
            "page_table": page_table,
            "prompt_lens": prompt_lens,
            "start_pos": start_pos,
            "empty_slots": empty_slots,
            "kv_cache": kv_cache,
            "sampling_params": sampling_params,
        }
        _copy_supplied_sampling_state(normalized, compatibility_kwargs)
        _normalize_tensor(normalized, "tokens", torch.long)
        _normalize_tensor(normalized, "page_table", torch.int32)
        _normalize_tensor(normalized, "prompt_lens", torch.long)
        _normalize_tensor(normalized, "start_pos", torch.long)
        return normalized, enable_trace

    def normalize_decode(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        compatibility_kwargs: Mapping[str, Any] | None = None,  # ↓ Compatibility
    ) -> tuple[NormalizedDecodeKwargs, bool]:
        """Normalize one explicit decode request and return trace intent separately."""

        self._validate_compatibility_kwargs(compatibility_kwargs, operation="decode")
        self._validate_trace_selection(enable_trace, operation="decode")
        normalized: NormalizedDecodeKwargs = {
            "tokens": tokens,
            "start_pos": start_pos,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "sampling_params": sampling_params,
            "reset_batch": reset_batch,
        }
        _copy_supplied_sampling_state(normalized, compatibility_kwargs)
        _normalize_tensor(normalized, "tokens", torch.long)
        _normalize_tensor(normalized, "start_pos", torch.long)
        _normalize_tensor(normalized, "page_table", torch.int32)

        tokens = normalized["tokens"]
        if tokens.ndim == 2 and tokens.shape[-1] == 1:
            normalized["tokens"] = tokens.reshape(-1)
        return normalized, enable_trace

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
    def _validate_compatibility_kwargs(
        compatibility_kwargs: Mapping[str, Any] | None,
        *,
        operation: str,
    ) -> None:
        if compatibility_kwargs is None:
            return
        if not isinstance(compatibility_kwargs, Mapping):
            raise TypeError("compatibility_kwargs must be a mapping")
        supported_keys = _IGNORED_VLLM_KWARGS | _SAMPLING_STATE_VLLM_KWARGS
        unknown_keys = sorted(key for key in compatibility_kwargs if key not in supported_keys)
        if unknown_keys:
            raise TypeError(f"{operation} got an unexpected keyword argument {unknown_keys[0]!r}")

    def _validate_trace_selection(self, enable_trace: bool, *, operation: str) -> None:
        if not isinstance(enable_trace, bool):
            raise TypeError("enable_trace must be bool")
        if operation == "prefill":
            configured = self.config.trace.prefill_enabled
        elif operation == "decode":
            configured = self.config.trace.decode_enabled
        else:
            raise ValueError(f"Unknown trace operation {operation!r}")
        # TraceConfig describes the trace targets made available at construction.
        # The unchanged vLLM loader constructs the default ``all`` superset and
        # presents its effective static mode through these required per-operation
        # booleans. Reject only requests for a trace target that was not built;
        # an unselected operation intentionally uses eager execution.
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


def _copy_supplied_sampling_state(
    normalized: NormalizedPrefillKwargs | NormalizedDecodeKwargs,
    compatibility_kwargs: Mapping[str, Any] | None,
) -> None:
    """Forward only request state that the external caller actually supplied."""

    if compatibility_kwargs is None:
        return
    for name in ("prompt_tokens", "output_tokens", "slot_remap"):
        value = compatibility_kwargs.get(name)
        if value is not None:
            normalized[name] = value


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

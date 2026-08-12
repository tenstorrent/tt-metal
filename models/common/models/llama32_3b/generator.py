# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM construction and compatibility delegation for Llama 3.2-3B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, TraceMode, WarmupConfig
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.llm_runtime.vllm_adapter import NormalizedPrefillKwargs, VLLMAdapter, VLLMAdapterConfig
from models.common.models.llama32_3b.executor import Llama32_3BExecutorConfig, build_llama32_3b_executor
from models.common.models.llama32_3b.hf_adaptor import from_pretrained
from models.common.models.llama32_3b.model import Llama32_3BPagedAttentionConfig

_PROVISIONAL_BLOCK_SIZE = 32


@dataclass(frozen=True)
class Llama32_3BGeneratorConfig:
    """Validated construction inputs for one vLLM-facing Llama generator."""

    hf_model: str
    mesh_device: Any
    max_batch_size: int
    max_seq_len: int
    n_layers: int | None = None
    tt_data_parallel: int = 1
    optimizations: Any = "performance"
    trace_mode: TraceMode = "all"
    device_sampling_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hf_model, str) or not self.hf_model:
            raise ValueError("hf_model must be a non-empty string")
        if self.mesh_device is None:
            raise ValueError("mesh_device is required")
        _validate_positive_int("max_batch_size", self.max_batch_size)
        _validate_positive_int("max_seq_len", self.max_seq_len)
        _validate_positive_int("tt_data_parallel", self.tt_data_parallel)
        if self.n_layers is not None:
            _validate_positive_int("n_layers", self.n_layers)
        if self.max_batch_size % self.tt_data_parallel != 0:
            raise ValueError(
                f"max_batch_size={self.max_batch_size} must be divisible by "
                f"tt_data_parallel={self.tt_data_parallel}"
            )
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        TraceConfig(mode=self.trace_mode)


class Llama32_3BGenerator:
    """Adapt vLLM's model interface to the model-owned execution target.

    vLLM constructs this facade with `initialize_vllm_model`, resolves
    KV capacity through `allocate_kv_cache`, warms the configured
    programs, and then calls `prefill_forward` and
    `decode_forward`. Each forward call is normalized by
    `VLLMAdapter`, dispatched to the eager or traced executor, and
    delegated to ``Llama32_3BExecutor`` or ``LaneGroupExecutor``.

    This class owns dispatch policy but no TT resources. `cleanup`
    delegates to the target that owns those resources.
    """

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "accepts_trace_mode": True,
    }
    requires_prefill_trace_warmup = True

    def __init__(self, target: Any, adapter: VLLMAdapter):
        self.target = target
        self._adapter = adapter

    # Public vLLM API

    @property
    def model(self):
        return self.target.model

    @property
    def model_args(self):
        return self.target.model_args

    @property
    def mesh_device(self):
        return self.target.mesh_device

    @property
    def cache_path(self):
        return self.target.cache_path

    @property
    def already_warmed_up_prefill(self):
        return self.target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value):
        self.target.already_warmed_up_prefill = value

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
    ) -> int:
        """Return the unpadded per-submesh KV token budget for vLLM sizing."""

        return int(max_model_len)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
        trace_mode: TraceMode = "all",
        device_sampling_enabled: bool = True,
    ):
        """Build the configured single-lane or data-parallel Llama target."""

        hf_model = getattr(hf_config, "_name_or_path", None)
        if not hf_model:
            raise ValueError("hf_config must provide a non-empty _name_or_path")
        return build_llama32_3b_generator(
            Llama32_3BGeneratorConfig(
                hf_model=str(hf_model),
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                tt_data_parallel=tt_data_parallel,
                optimizations=optimizations,
                trace_mode=trace_mode,
                device_sampling_enabled=device_sampling_enabled,
            )
        )

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        """Resolve the late vLLM capacity, then allocate a borrowed cache handle."""

        supplied = (kv_cache_shape is not None, dtype is not None, num_layers is not None)
        if not any(supplied):
            return self.target.allocate_kv_cache()
        if not all(supplied):
            raise TypeError("kv_cache_shape, dtype, and num_layers must be supplied together")

        resolved = self._adapter.resolve_legacy_kv_cache_config(kv_cache_shape, dtype, num_layers)
        self.target.configure_paged_kv_cache(resolved)
        return self.target.allocate_kv_cache()

    def compile_prefill(
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
    ) -> None:
        """Normalize a vLLM prefill call and compile its selected target."""

        normalized, trace_requested = self._adapter.normalize_prefill(
            tokens,
            page_table,
            enable_trace=enable_trace,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        execution = self._select_prefill_execution(normalized, trace_requested)
        return self.target.compile_prefill(execution=execution, **normalized)

    def compile_decode(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
    ) -> None:
        """Normalize a vLLM decode call and compile its selected target."""

        normalized, trace_requested = self._adapter.normalize_decode(
            tokens,
            start_pos,
            page_table,
            enable_trace=enable_trace,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )
        execution = self._select_execution("decode", trace_requested)
        return self.target.compile_decode(execution=execution, **normalized)

    def prefill_forward(
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
        **compatibility_kwargs: Any,  # ↓ Compatibility
    ) -> Any:
        """Normalize and dispatch one vLLM prefill call."""

        normalized, trace_requested = self._adapter.normalize_prefill(
            tokens,
            page_table,
            enable_trace=enable_trace,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            compatibility_kwargs=compatibility_kwargs,
        )
        execution = self._select_prefill_execution(normalized, trace_requested)
        return self.target.prefill_forward(execution=execution, **normalized)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
        **compatibility_kwargs: Any,  # ↓ Compatibility
    ) -> Any:
        """Normalize and dispatch one vLLM decode call."""

        normalized, trace_requested = self._adapter.normalize_decode(
            tokens,
            start_pos,
            page_table,
            enable_trace=enable_trace,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            compatibility_kwargs=compatibility_kwargs,
        )
        execution = self._select_execution("decode", trace_requested)
        return self.target.decode_forward(
            execution=execution,
            read_from_device=read_from_device,
            **normalized,
        )

    def read_decode_output(
        self,
        tt_out: Any,
        *,
        async_read: bool = False,
    ) -> Any:
        """Delegate vLLM's raw decode-output read."""

        return self.target.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(
        self,
        tt_out: Any,
        *,
        is_tokens: bool = False,
    ) -> tuple[Any, Any]:
        """Delegate vLLM's asynchronous host-output completion."""

        return self.target.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        return self.target.warmup_model_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        return self.target.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=max_batch_size,
            num_blocks=num_blocks,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    def cleanup(self):
        """Release every resource owned by the concrete target."""

        return self.target.cleanup()

    # Private implementation

    def _select_prefill_execution(
        self,
        normalized: NormalizedPrefillKwargs,
        trace_requested: bool,
    ):
        # Static trace intent is authoritative. Eligibility and configured
        # coverage are preflighted by the selected execution target; this
        # facade must never turn a required trace miss into eager KV writes.
        return self._select_execution("prefill", trace_requested)

    def _select_execution(self, operation: str, enable_trace: bool):
        if not enable_trace:
            return self.target.eager_execution
        execution = getattr(self.target, f"traced_{operation}_execution")
        if execution is None:
            raise RuntimeError(f"vLLM requested unavailable traced {operation} execution")
        return execution


def build_llama32_3b_generator(config: Llama32_3BGeneratorConfig) -> Llama32_3BGenerator:
    """Construct lane-local models/executors and compose their shared target surface."""

    per_lane_max_batch_size = config.max_batch_size // config.tt_data_parallel
    submeshes = (
        [config.mesh_device]
        if config.tt_data_parallel == 1
        else list(_create_submeshes(config.mesh_device, config.tt_data_parallel))
    )
    if len(submeshes) != config.tt_data_parallel:
        raise ValueError(f"Expected {config.tt_data_parallel} submeshes, got {len(submeshes)}")

    max_num_blocks = (
        config.max_seq_len + _PROVISIONAL_BLOCK_SIZE - 1
    ) // _PROVISIONAL_BLOCK_SIZE + per_lane_max_batch_size
    lanes = []
    try:
        for submesh in submeshes:
            paged_attention_config = Llama32_3BPagedAttentionConfig(
                block_size=_PROVISIONAL_BLOCK_SIZE,
                max_num_blocks=max_num_blocks,
            )
            llm = from_pretrained(
                mesh_device=submesh,
                hf_model=config.hf_model,
                instruct="Instruct" in config.hf_model,
                max_batch_size=per_lane_max_batch_size,
                max_seq_len=config.max_seq_len,
                optimizations=config.optimizations,
                n_layers=config.n_layers,
                dtype=ttnn.bfloat8_b,
                paged_attention_config=paged_attention_config,
            )
            model_kv_cache_dtypes, _, _, _ = _model_kv_metadata(llm.model)
            executor_config = Llama32_3BExecutorConfig(
                trace=TraceConfig(mode=config.trace_mode),
                warmup=WarmupConfig(),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=_PROVISIONAL_BLOCK_SIZE,
                    max_num_blocks=max_num_blocks,
                    dtype=model_kv_cache_dtypes[0],
                ),
                device_sampling_enabled=config.device_sampling_enabled,
            )
            lanes.append(build_llama32_3b_executor(llm, executor_config))

        adapter = _build_vllm_adapter(lanes[0])
    except BaseException as primary:
        _cleanup_after_construction_failure(lanes, primary)
        raise

    target = lanes[0] if config.tt_data_parallel == 1 else LaneGroupExecutor(lanes, mesh_device=config.mesh_device)
    return Llama32_3BGenerator(target, adapter)


def _build_vllm_adapter(lane) -> VLLMAdapter:
    model_kv_cache_dtypes, num_layers, kv_heads_per_device, head_dim = _model_kv_metadata(lane.model)
    return VLLMAdapter(
        VLLMAdapterConfig.resolve(
            trace=lane.config.trace,
            paged_kv_cache=lane.config.paged_kv_cache,
            expected_num_layers=num_layers,
            expected_kv_heads_per_device=kv_heads_per_device,
            expected_head_dim=head_dim,
            model_kv_cache_dtype=model_kv_cache_dtypes,
        )
    )


def _model_kv_metadata(model) -> tuple[tuple[Any, ...], int, int, int]:
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("Llama-3.2-3B model must contain at least one attention layer")

    attention_configs = tuple(layer.attention.config for layer in layers)
    model_config = model.config
    num_layers = int(model_config.n_layers)
    if len(attention_configs) != num_layers:
        raise ValueError(f"Model config declares {num_layers} layers but exposes {len(attention_configs)}")

    num_devices = int(model_config.num_devices)
    n_kv_heads = int(attention_configs[0].n_kv_heads)
    if n_kv_heads % num_devices != 0:
        raise ValueError(f"n_kv_heads={n_kv_heads} must be divisible by num_devices={num_devices}")

    head_dim = int(attention_configs[0].head_dim)
    if any(
        int(attention_config.n_kv_heads) != n_kv_heads or int(attention_config.head_dim) != head_dim
        for attention_config in attention_configs
    ):
        raise ValueError("Every Llama layer must expose the same KV head shape")

    return (
        tuple(attention_config.kv_cache_dtype for attention_config in attention_configs),
        num_layers,
        n_kv_heads // num_devices,
        head_dim,
    )


def _create_submeshes(mesh_device, tt_data_parallel):
    from models.tt_transformers.tt.generator import create_submeshes

    return create_submeshes(mesh_device, tt_data_parallel)


def _cleanup_after_construction_failure(lanes, primary):
    failures = []
    for lane in lanes:
        try:
            lane.cleanup()
        except BaseException as error:
            failures.append(error)
    if failures:
        setattr(primary, "cleanup_failures", failures)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")

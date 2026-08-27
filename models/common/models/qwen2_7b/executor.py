# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Qwen2-7B execution composition and cleanup root."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch

from models.common.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeRuntime, DecodeRuntimeConfig
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.paged_kv_cache import PagedKVCacheManager
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.runtime import PrefillRuntime
from models.common.llm_runtime.program_compiler import ProgramCompiler
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures
from models.common.llm_runtime.trace_compiler import TraceCompiler
from models.common.llm_runtime.warmup import WarmupCoordinator, WarmupCoordinatorConfig
from models.common.models.qwen2_7b.hf_adaptor import Qwen2ForCausalLM
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class Qwen2ExecutorConfig:
    """Immutable aggregate policy paired with one model-owned executor."""

    trace: TraceConfig
    warmup: WarmupConfig
    paged_kv_cache: PagedKVCacheConfig
    device_sampling_enabled: bool

    def __post_init__(self) -> None:
        nested_configs = (
            ("trace", self.trace, TraceConfig),
            ("warmup", self.warmup, WarmupConfig),
            ("paged_kv_cache", self.paged_kv_cache, PagedKVCacheConfig),
        )
        for name, value, expected_type in nested_configs:
            if type(value) is not expected_type:
                raise TypeError(f"{name} must be exactly {expected_type.__name__}")
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")


class Qwen2Executor:
    """Compose every runtime owner for one Qwen2-7B execution lane.

    Construction wires one `PrefillRuntime`, `DecodeRuntime`,
    `ProgramCompiler`, and `EagerExecutor`. Trace-enabled
    configurations add one `TraceCompiler` and one
    `TracedExecutor` over that exact eager instance.

    ``Qwen2Generator`` or ``LaneGroupExecutor`` first configures and allocates
    KV cache, then warms or compiles programs, and finally calls
    `prefill_forward` and `decode_forward` with an explicit
    execution target. `cleanup` is the deterministic resource-release
    root and makes the executor terminal.
    """

    requires_prefill_trace_warmup = True

    def __init__(self, model: Any, runtime_config: Any, config: Qwen2ExecutorConfig) -> None:
        if not isinstance(config, Qwen2ExecutorConfig):
            raise TypeError("config must be a Qwen2ExecutorConfig")
        iter_modules = getattr(model, "iter_executor_named_modules", None)
        if not callable(iter_modules):
            raise TypeError("model must provide iter_executor_named_modules()")
        can_enable_trace = getattr(runtime_config, "can_enable_trace", None)
        if not callable(can_enable_trace):
            raise TypeError("runtime_config must provide can_enable_trace()")
        model_config = getattr(model, "config", None)
        mesh_device = getattr(model_config, "mesh_device", None)
        if mesh_device is None:
            raise ValueError("model.config.mesh_device is required")

        self.model = model
        self.runtime_config = runtime_config
        self.model_args = runtime_config
        self.config = config
        self.mesh_device = mesh_device
        self.cache_path = getattr(runtime_config, "model_cache_path", None)
        self._terminal = False
        self._cleaned_up = False
        self._sampling_buffers_loaded = False
        self._runtime_configuration_sealed = False
        self._q128_topk_tile_ends_warmed: set[bool] = set()

        sampling = getattr(model, "sampling", None)
        if config.device_sampling_enabled:
            if not isinstance(sampling, Sampling1D):
                raise TypeError("device sampling requires model.sampling to be a Sampling1D")
            is_resolved = getattr(getattr(sampling, "config", None), "is_resolved", None)
            if not callable(is_resolved) or not is_resolved():
                raise ValueError("model.sampling must have a resolved Sampling1DConfig")

        self.kv_cache_manager = PagedKVCacheManager(model, config.paged_kv_cache)
        self.page_table_layout = self._resolve_page_table_layout()
        self.output_reader = OutputReader(mesh_device)
        self.prefill_runtime = PrefillRuntime(
            PrefillRuntimeConfig.resolve(
                model=model,
                output_reader=self.output_reader,
                page_table_layout=self.page_table_layout,
                max_batch_size=int(model.config.max_batch_size),
                max_prefill_chunk_size=int(runtime_config.max_prefill_chunk_size),
                device_sampling_enabled=config.device_sampling_enabled,
                can_enable_trace=runtime_config.can_enable_trace,
                supports_batched_prefill=bool(runtime_config.supports_batched_prefill),
                disable_batched_prefill=bool(runtime_config.disable_batched_prefill),
                max_prefill_batch_size=int(runtime_config.max_prefill_batch_size),
                batched_prefill_batched_extract=bool(runtime_config.batched_prefill_batched_extract),
            )
        )
        self.decode_runtime = DecodeRuntime(
            DecodeRuntimeConfig.resolve(
                model=model,
                output_reader=self.output_reader,
                lane_capacity=int(model.config.max_batch_size),
                page_table_layout=self.page_table_layout,
                device_sampling_enabled=config.device_sampling_enabled,
                force_greedy_top_k=config.warmup.include_decode_top_k,
            )
        )
        self.program_compiler = ProgramCompiler(mesh_device, lambda: self.kv_cache_manager.bound_context)
        self.eager_executor = EagerExecutor(
            prefill=self.prefill_runtime,
            decode=self.decode_runtime,
            program_compiler=self.program_compiler,
        )
        self.trace_compiler: TraceCompiler | None = None
        self.traced_executor: TracedExecutor | None = None
        if config.trace.mode != "none":
            self.trace_compiler = TraceCompiler(self.program_compiler)
            self.traced_executor = TracedExecutor(
                eager=self.eager_executor,
                trace_compiler=self.trace_compiler,
                trace_mode=config.trace.mode,
            )
        self.eager_execution = self.eager_executor
        self.traced_prefill_execution = (
            self.traced_executor if config.trace.prefill_enabled and self.traced_executor is not None else None
        )
        self.traced_decode_execution = (
            self.traced_executor if config.trace.decode_enabled and self.traced_executor is not None else None
        )
        self._prefill_execution = self.traced_prefill_execution or self.eager_executor
        self._decode_execution = self.traced_decode_execution or self.eager_executor

        prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
        self.warmup = WarmupCoordinator(
            config=WarmupCoordinatorConfig.resolve(
                warmup=config.warmup,
                trace=config.trace,
                prefill=self.prefill_runtime.config,
                decode=self.decode_runtime.config,
                prefill_sequence_lengths=prefill_sequence_lengths,
            ),
            execution=self.traced_executor or self.eager_executor,
            ensure_sampling_buffers=self._ensure_sampling_buffers,
            validate_bound_cache=self._validate_bound_cache,
        )

    # Public model execution API

    @property
    def model_config(self):
        return self.model.config

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        return self.kv_cache_manager.config

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def already_warmed_up_prefill(self) -> bool:
        return self.warmup.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("already_warmed_up_prefill compatibility value must be bool")
        # Compatibility writes are intentionally non-authoritative. Warmup
        # coverage is derived from the coordinator's completed case ledger.
        return None

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        """Resolve vLLM-owned KV geometry before the first allocation."""

        self._ensure_active()
        if self._runtime_configuration_sealed:
            raise RuntimeError("runtime configuration is sealed")
        if not isinstance(config, PagedKVCacheConfig):
            raise TypeError("config must be a PagedKVCacheConfig")
        if self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("paged KV cache configuration is already resolved")
        if config.dtype != self.kv_cache_manager.config.dtype:
            raise ValueError("resolved paged KV cache cannot change dtype")
        if config.memory_config != self.kv_cache_manager.config.memory_config:
            raise ValueError("resolved paged KV cache cannot change memory_config")
        self.model.configure_paged_attention(
            block_size=config.block_size,
            max_num_blocks=config.max_num_blocks,
        )
        self.kv_cache_manager.configure(config)
        self.config = replace(self.config, paged_kv_cache=config)
        self._refresh_page_table_layout()

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        num_layers: int | None = None,
    ) -> list[list[Any]]:
        """Allocate and bind the model-owned paged KV cache."""

        self._ensure_active()
        supplied = (kv_cache_shape is not None, dtype is not None, num_layers is not None)
        if any(supplied):
            if not all(supplied):
                raise TypeError("kv_cache_shape, dtype, and num_layers must be supplied together")
            shape = tuple(int(dimension) for dimension in kv_cache_shape)
            if len(shape) != 4:
                raise ValueError(f"KV cache shape must have rank 4, got {shape}")
            expected_layers = len(self.kv_cache_manager.per_layer_dtypes)
            if int(num_layers) != expected_layers:
                raise ValueError(f"vLLM KV layer count {num_layers} does not match model layer count {expected_layers}")
            self.kv_cache_manager.validate_vllm_cache_spec(
                block_size=shape[2],
                dtype=dtype,
                num_blocks=shape[0],
            )
            if self.kv_cache_manager.config.num_blocks is None:
                self.kv_cache_manager.configure(replace(self.kv_cache_manager.config, num_blocks=shape[0]))
                self._refresh_page_table_layout()
            elif self.kv_cache_manager.config.num_blocks != shape[0]:
                raise ValueError(
                    f"Paged KV cache is resolved to {self.kv_cache_manager.config.num_blocks} blocks, not {shape[0]}"
                )
            if any(tuple(expected) != shape for expected in self.kv_cache_manager.cache_shapes):
                raise ValueError(
                    f"vLLM KV shape {shape} does not match model-derived shapes {self.kv_cache_manager.cache_shapes}"
                )
        if not self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("Paged KV cache capacity must be resolved before allocation")
        self._seal_runtime_configuration()
        return self.kv_cache_manager.allocate()

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> None:
        """Compile prefill on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._prefill_execution).compile_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> None:
        """Compile decode on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._decode_execution).compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Validate ownership and execute one prefill call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._prefill_execution).prefill_forward(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Validate ownership and execute one decode call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._decode_execution).decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
        )

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
    ) -> bool:
        """Classify whether the prefill request can use this lane's trace."""

        if self.traced_executor is None or not self.config.trace.prefill_enabled:
            return False
        return self.prefill_runtime.can_trace(
            tokens=tokens,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
        )

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        self._ensure_active()
        return self.decode_runtime.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
        self._ensure_active()
        return self.decode_runtime.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        self._ensure_active()
        if enable_trace:
            self._warmup_q128_topk_tile_ends(
                kv_cache=kv_cache,
                can_sample_on_device=can_sample_on_device,
                enable_trace=True,
            )
        self.warmup.warmup_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )
        if not enable_trace:
            self._warmup_q128_topk_tile_ends(
                kv_cache=kv_cache,
                can_sample_on_device=can_sample_on_device,
                enable_trace=False,
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
        self._ensure_active()
        return self.warmup.warmup_decode(
            kv_cache=kv_cache,
            max_batch_size=max_batch_size,
            num_blocks=num_blocks,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    def cleanup(self) -> None:
        """Release runtime, trace, program, sampling, and KV resources in order."""

        self._terminal = True
        if self._cleaned_up:
            return

        failures = []
        actions = [
            self.decode_runtime.drain_external_outputs,
            self.output_reader.drain,
            self.prefill_runtime.cleanup,
            self.decode_runtime.cleanup_transients,
        ]
        if self.trace_compiler is not None:
            actions.append(self.trace_compiler.cleanup)
        actions.append(self.program_compiler.cleanup)
        if self.config.device_sampling_enabled:
            actions.append(self.model.sampling.release)
        actions.append(self.kv_cache_manager.release)

        for action in actions:
            try:
                action()
            except BaseException as error:
                failures.append(error)
        if failures:
            _raise_cleanup_failures(failures, "Qwen2Executor")
        self._cleaned_up = True

    # Private implementation

    def _warmup_q128_topk_tile_ends(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool,
        enable_trace: bool,
    ) -> None:
        """Prime Qwen2 Q128 tile ends in eager and traced execution."""

        if (
            enable_trace in self._q128_topk_tile_ends_warmed
            or not can_sample_on_device
            or (enable_trace and self.traced_executor is None)
            or 128 not in self.warmup.config.prefill_sequence_lengths
            or not self.prefill_runtime.config.static_q128_topk_supported
            or self.warmup.config.prime_q128_tile_ends
        ):
            return
        sampling = SamplingParams(
            temperature=torch.ones(1),
            top_k=torch.full((1,), 32, dtype=torch.int32),
            top_p=torch.full((1,), 0.08),
        )
        execution = self.traced_executor if enable_trace else self.eager_executor
        for sequence_length in (32, 64, 96):
            page_table_width = (
                sequence_length + self.page_table_layout.block_size - 1
            ) // self.page_table_layout.block_size
            self.compile_prefill(
                tokens=torch.zeros((1, sequence_length), dtype=torch.long),
                page_table=torch.zeros((1, page_table_width), dtype=torch.int32),
                prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
                empty_slots=[0],
                kv_cache=kv_cache,
                sampling_params=sampling,
                execution=execution,
            )
        self._q128_topk_tile_ends_warmed.add(enable_trace)

    def _resolve_page_table_layout(self) -> PageTableLayout:
        kv_config = self.kv_cache_manager.config
        # The direct demo resolves num_blocks=max_num_blocks before constructing
        # this executor, so its geometry is final immediately and it physically
        # allocates that maximum. vLLM constructs against max_num_blocks only as
        # a cheap capacity ceiling: no KV tensor is allocated until vLLM supplies
        # num_blocks, _refresh_page_table_layout installs the final geometry, and
        # allocate_kv_cache seals all runtime configs.
        physical_num_blocks = kv_config.num_blocks or kv_config.max_num_blocks
        return PageTableLayout.resolve(
            block_size=int(kv_config.block_size),
            model_max_sequence_length=int(self.model.config.max_seq_len),
            physical_num_blocks=int(physical_num_blocks),
            max_prefill_chunk_size=min(
                int(self.runtime_config.max_prefill_chunk_size),
                int(self.model.config.max_seq_len),
            ),
        )

    def _refresh_page_table_layout(self) -> None:
        # vLLM reaches this boundary after choosing the physical block count and
        # before PagedKVCacheManager.allocate(), compilation, warmup, or tracing.
        # Re-resolve complete immutable configs so their normal construction
        # checks and capacity ceilings describe vLLM's authoritative geometry.
        # Runtime and resource-owner identities remain unchanged.
        layout = self._resolve_page_table_layout()
        current_prefill = self.prefill_runtime.config
        prefill_config = PrefillRuntimeConfig.resolve(
            model=self.model,
            output_reader=self.output_reader,
            page_table_layout=layout,
            max_batch_size=current_prefill.max_batch_size,
            max_prefill_chunk_size=current_prefill.max_prefill_chunk_size,
            device_sampling_enabled=current_prefill.device_sampling_enabled,
            can_enable_trace=current_prefill.can_enable_trace,
            supports_batched_prefill=current_prefill.supports_batched_prefill,
            disable_batched_prefill=current_prefill.disable_batched_prefill,
            max_prefill_batch_size=current_prefill.max_prefill_batch_size,
            batched_prefill_batched_extract=current_prefill.batched_prefill_batched_extract,
        )
        current_decode = self.decode_runtime.config
        decode_config = DecodeRuntimeConfig.resolve(
            model=self.model,
            output_reader=self.output_reader,
            lane_capacity=current_decode.lane_capacity,
            page_table_layout=layout,
            device_sampling_enabled=current_decode.device_sampling_enabled,
            force_greedy_top_k=current_decode.force_greedy_top_k,
        )
        warmup_config = WarmupCoordinatorConfig.resolve(
            warmup=self.warmup.config.warmup,
            trace=self.config.trace,
            prefill=prefill_config,
            decode=decode_config,
            prefill_sequence_lengths=self.warmup.config.prefill_sequence_lengths,
        )

        self.prefill_runtime.config = prefill_config
        self.decode_runtime.config = decode_config
        self.warmup.config = warmup_config
        self.page_table_layout = layout

    def _seal_runtime_configuration(self) -> None:
        """Seal final geometry immediately before physical KV allocation."""

        self.warmup.seal_configuration()
        self._runtime_configuration_sealed = True

    def _ensure_sampling_for(self, sampling_params: Any) -> None:
        if sampling_params is None:
            return
        if not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")
        self._ensure_sampling_buffers()

    def _ensure_sampling_buffers(self) -> None:
        if self._sampling_buffers_loaded:
            return
        if self.trace_compiler is not None and self.trace_compiler.trace_active:
            raise RuntimeError("cannot materialize sampling buffers after trace activation")
        self.model.sampling.load_device_buffers()
        self._sampling_buffers_loaded = True

    def _validate_bound_cache(self, kv_cache: Any) -> None:
        if self.kv_cache_manager.bound_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before execution")
        if kv_cache is not None:
            self.kv_cache_manager.validate_borrowed_handle(kv_cache)

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("Qwen2Executor is terminal; construct a new executor")
        if self.prefill_runtime.transient_orphan_count or self.decode_runtime.transient_orphan_count:
            raise RuntimeError("Qwen2Executor has unreleased transient resources; clean up this executor")


def build_qwen2_7b_executor(llm: Qwen2ForCausalLM, config: Qwen2ExecutorConfig) -> Qwen2Executor:
    """Build one executor around an already-loaded Qwen model adapter."""

    return Qwen2Executor(llm.model, llm.runtime_config, config)


def _raise_cleanup_failures(failures: list[BaseException], owner: str) -> None:
    primary, *additional = failures
    attach_cleanup_failures(
        primary,
        additional,
        note=f"{owner} cleanup also encountered {{count}} failure(s)",
    )
    raise primary

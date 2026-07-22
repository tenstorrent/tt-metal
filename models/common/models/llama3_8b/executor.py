# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Llama 3.1-8B execution composition and cleanup root."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from models.common.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.paged_kv_cache import PagedKVCacheManager
from models.common.llm_runtime.prefill import PrefillRuntime
from models.common.llm_runtime.program_compiler import ProgramCompiler
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures
from models.common.llm_runtime.trace_compiler import TraceCompiler
from models.common.llm_runtime.warmup import WarmupCoordinator
from models.common.modules.sampling.sampling_1d import Sampling1D

if TYPE_CHECKING:
    from models.common.models.llama3_8b.hf_adaptor import Llama3ForCausalLM


@dataclass(frozen=True)
class Llama3ExecutorConfig:
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


class Llama3Executor:
    """Thin configured single-lane facade and deterministic cleanup root."""

    requires_prefill_trace_warmup = True

    def __init__(self, model: Any, runtime_config: Any, config: Llama3ExecutorConfig) -> None:
        if not isinstance(config, Llama3ExecutorConfig):
            raise TypeError("config must be a Llama3ExecutorConfig")
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
            model=model,
            mesh_device=mesh_device,
            output_reader=self.output_reader,
            page_table_layout=self.page_table_layout,
            max_batch_size=int(model.config.max_batch_size),
            max_prefill_chunk_size=int(runtime_config.max_prefill_chunk_size),
            cluster_shape=list(mesh_device.shape),
            device_sampling_enabled=config.device_sampling_enabled,
            can_enable_trace=runtime_config.can_enable_trace,
        )
        self.decode_runtime = DecodeRuntime(
            model,
            mesh_device,
            self.output_reader,
            lane_capacity=int(model.config.max_batch_size),
            page_table_layout=self.page_table_layout,
            device_sampling_enabled=config.device_sampling_enabled,
            force_greedy_top_k=config.warmup.include_decode_top_k,
        )
        self.program_compiler = ProgramCompiler(mesh_device, lambda: self.kv_cache_manager.bound_context)
        self.eager_executor = EagerExecutor(
            prefill=self.prefill_runtime,
            decode=self.decode_runtime,
            program_compiler=self.program_compiler,
        )
        self.trace_compiler: TraceCompiler | None = None
        self.traced_executor: TracedExecutor | None = None
        if config.trace.mode == "none":
            self.execution: EagerExecutor | TracedExecutor = self.eager_executor
        else:
            self.trace_compiler = TraceCompiler(self.program_compiler, mesh_device, config.trace)
            self.traced_executor = TracedExecutor(eager=self.eager_executor, trace_compiler=self.trace_compiler)
            self.execution = self.traced_executor

        prefill_sequence_lengths = tuple(
            int(value) for value in (getattr(runtime_config, "trace_prefill_supported_seq_lens", ()) or (128,))
        )
        self.warmup = WarmupCoordinator(
            config=config.warmup,
            trace_config=config.trace,
            execution=self.execution,
            eager=self.eager_executor,
            trace_compiler=self.trace_compiler,
            model=model,
            page_table_layout=self.page_table_layout,
            prefill_sequence_lengths=prefill_sequence_lengths,
            device_sampling_enabled=config.device_sampling_enabled,
            ensure_sampling_buffers=self._ensure_sampling_buffers,
            validate_bound_cache=self._validate_bound_cache,
        )

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

    @property
    def device_decode_feedback_enabled(self) -> bool:
        return self.decode_runtime.device_feedback_enabled

    @device_decode_feedback_enabled.setter
    def device_decode_feedback_enabled(self, value: bool) -> None:
        self.decode_runtime.device_feedback_enabled = bool(value)

    @property
    def trace_id_prefill(self):
        if self.trace_compiler is None:
            return None
        ids = [
            record.artifact.trace_id
            for record in self.trace_compiler.traces.values()
            if record.operation == "prefill" and record.artifact is not None
        ]
        return ids[0] if ids else None

    @property
    def trace_ids_decode(self) -> list[int]:
        if self.trace_compiler is None:
            return []
        return [
            record.artifact.trace_id
            for record in self.trace_compiler.traces.values()
            if record.operation == "decode" and record.artifact is not None
        ]

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        self._ensure_active()
        self.kv_cache_manager.configure(config)
        self._refresh_page_table_layout()

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        num_layers: int | None = None,
    ) -> list[list[Any]]:
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
        return self.kv_cache_manager.allocate()

    def compile_prefill(self, **kwargs: Any) -> None:
        self._ensure_active()
        self._validate_bound_cache(kwargs.get("kv_cache"))
        self._ensure_sampling_for(kwargs.get("sampling_params"))
        return self.execution.compile_prefill(**kwargs)

    def compile_decode(self, **kwargs: Any) -> None:
        self._ensure_active()
        self._validate_bound_cache(kwargs.get("kv_cache"))
        self._ensure_sampling_for(kwargs.get("sampling_params"))
        return self.execution.compile_decode(**kwargs)

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
        enable_trace=None,
    ):
        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return self.execution.prefill_forward(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
            enable_trace=enable_trace,
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        enable_trace=None,
    ):
        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return self.execution.decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
            enable_trace=enable_trace,
        )

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        self._ensure_active()
        return self.decode_runtime.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False):
        self._ensure_active()
        return self.decode_runtime.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_active()
        return self.warmup.warmup_prefill(*args, **kwargs)

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_active()
        return self.warmup.warmup_decode(*args, **kwargs)

    def cleanup(self) -> None:
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
            _raise_cleanup_failures(failures, "Llama3Executor")
        self._cleaned_up = True

    def _resolve_page_table_layout(self) -> PageTableLayout:
        kv_config = self.kv_cache_manager.config
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
        layout = self._resolve_page_table_layout()
        self.page_table_layout = layout
        self.prefill_runtime.page_table_layout = layout
        self.prefill_runtime.block_size = layout.block_size
        self.prefill_runtime.max_actual_page_table_width = layout.raw_capacity_width
        self.prefill_runtime.canonical_page_table_width = layout.prefill_width
        self.decode_runtime.page_table_layout = layout
        self.warmup.page_table_layout = layout

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
            raise RuntimeError("Llama3Executor is terminal; construct a new executor")
        if self.prefill_runtime.transient_orphan_count or self.decode_runtime.transient_orphan_count:
            raise RuntimeError("Llama3Executor has unreleased transient resources; clean up this executor")


def build_llama3_executor(llm: Llama3ForCausalLM, config: Llama3ExecutorConfig) -> Llama3Executor:
    return Llama3Executor(llm.model, llm.runtime_config, config)


def _raise_cleanup_failures(failures: list[BaseException], owner: str) -> None:
    primary, *additional = failures
    attach_cleanup_failures(
        primary,
        additional,
        note=f"{owner} cleanup also encountered {{count}} failure(s)",
    )
    raise primary

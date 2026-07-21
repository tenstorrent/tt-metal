# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Configured single-lane LLM execution runtime."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch

import ttnn
from models.common.llm_runtime.config import LLMExecutorConfig, PagedKVCacheConfig
from models.common.llm_runtime.graph_compiler import (
    GraphKey,
    InputRefreshPolicy,
    LLMGraphCompiler,
    OutputSpec,
    PersistentInputs,
    TraceCapturePlan,
)
from models.common.llm_runtime.module_input_validation import (
    suspend_module_input_validation,
    validate_module_input_configs,
)
from models.common.llm_runtime.output_reader import OutputReader, PendingRead
from models.common.llm_runtime.paged_kv_cache import PagedKVCacheManager
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.sampling import SamplingParams, format_sampling_params

_SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
_MAX_BATCHED_PREFILL_TOKENS = 128 * 1024
_PAGE_TABLE_WIDTH_ALIGNMENT = 8


class Mode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class _PrefillRequest:
    kind: str
    source_rows: tuple[int, ...]
    slots: tuple[int, ...]
    tokens: torch.Tensor
    page_table: torch.Tensor
    prompt_lengths: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    last_token_indices: tuple[int, ...]
    padded_sequence_length: int
    padded_batch_size: int
    chunk_page_table_width: int | None
    trace_eligible: bool

    def graph_key(self, sampling_path: str) -> GraphKey:
        return GraphKey(
            mode="prefill",
            batch_size=self.padded_batch_size,
            page_table_width=int(self.page_table.shape[-1]),
            sampling_path=sampling_path,
            sequence_length=self.padded_sequence_length,
            chunk_page_table_width=self.chunk_page_table_width,
        )


@dataclass
class _ExternalDecodeOutput:
    raw_value: Any
    owned_values: Any
    host_value: Any = None
    pending: PendingRead | None = None
    released: bool = False
    deallocated_tensor_ids: set[int] = field(default_factory=set, repr=False)


@dataclass
class _TransientOrphan:
    values: Any
    deallocated_tensor_ids: set[int] = field(default_factory=set, repr=False)


class LLMExecutor:
    """Concrete configured runtime for one model and one mesh lane."""

    requires_prefill_trace_warmup = True

    def __init__(self, model: Any, runtime_config: Any, config: LLMExecutorConfig):
        if not isinstance(config, LLMExecutorConfig):
            raise TypeError("config must be an LLMExecutorConfig")
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
        self.mode: Mode | None = None
        self.already_warmed_up_prefill = False
        self.device_decode_feedback_enabled = True

        sampling = getattr(model, "sampling", None)
        if config.device_sampling_enabled:
            if not isinstance(sampling, Sampling1D):
                raise TypeError("device sampling requires model.sampling to be a Sampling1D")
            is_resolved = getattr(getattr(sampling, "config", None), "is_resolved", None)
            if not callable(is_resolved) or not is_resolved():
                raise ValueError("model.sampling must have a resolved Sampling1DConfig")

        self._kv_manager = PagedKVCacheManager(model, config.paged_kv_cache)
        self._graph_compiler = LLMGraphCompiler(
            mesh_device,
            config.trace,
            lambda: self._kv_manager.bound_context,
        )
        self._output_reader = OutputReader(mesh_device)
        self._trace_plans: dict[GraphKey, TraceCapturePlan] = {}
        self._previous_decode_page_table: torch.Tensor | None = None
        self._external_by_raw_id: dict[int, _ExternalDecodeOutput] = {}
        self._external_by_host_id: dict[int, _ExternalDecodeOutput] = {}
        self._transient_orphans: list[_TransientOrphan] = []
        self._terminal = False
        self._cleaned_up = False
        self._sampling_buffers_loaded = False
        self._traces_captured = False
        self._prefill_trace_sampling_path: str | None = None

    @property
    def model_config(self):
        return self.model.config

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        return self._kv_manager.config

    @property
    def kv_cache_manager(self) -> PagedKVCacheManager:
        return self._kv_manager

    @property
    def graph_compiler(self) -> LLMGraphCompiler:
        return self._graph_compiler

    @property
    def output_reader(self) -> OutputReader:
        return self._output_reader

    @property
    def _kv_cache(self):
        return self._kv_manager.bound_cache

    @property
    def terminal(self) -> bool:
        return self._terminal

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        self._ensure_active()
        self._kv_manager.configure(config)

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
            expected_layers = len(self._kv_manager.per_layer_dtypes)
            if int(num_layers) != expected_layers:
                raise ValueError(f"vLLM KV layer count {num_layers} does not match model layer count {expected_layers}")
            self._kv_manager.validate_vllm_cache_spec(
                block_size=shape[2],
                dtype=dtype,
                num_blocks=shape[0],
            )
            if self._kv_manager.config.num_blocks is None:
                self._kv_manager.configure(replace(self._kv_manager.config, num_blocks=shape[0]))
            elif self._kv_manager.config.num_blocks != shape[0]:
                raise ValueError(
                    f"Paged KV cache is resolved to {self._kv_manager.config.num_blocks} blocks, " f"not {shape[0]}"
                )
            expected_shapes = self._kv_manager.cache_shapes
            if any(tuple(expected) != shape for expected in expected_shapes):
                raise ValueError(f"vLLM KV shape {shape} does not match model-derived shapes {expected_shapes}")

        return self._kv_manager.allocate()

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[list[Any]] | None = None,
        prompt_lens: torch.Tensor | None = None,
        empty_slots: list[int] | None = None,
        start_pos: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
        enable_trace: bool | None = None,
    ) -> None:
        self._ensure_active()
        self._validate_sampling_request(sampling_params)
        self._validate_bound_cache(kv_cache)
        requests = self._plan_prefill(tokens, page_table, prompt_lens, empty_slots, start_pos)
        sampling_path = "topk" if sampling_params is not None else "logits"
        trace_requested = self._resolve_trace_hint("prefill", enable_trace)
        if sampling_params is not None:
            self._ensure_sampling_buffers()

        for request in requests:
            request_sampling = _slice_sampling_params(sampling_params, request.source_rows)
            if not trace_requested or not request.trace_eligible:
                with self._validation_context("prefill"):
                    self._warm_eager_invocation(
                        lambda request=request, request_sampling=request_sampling: (
                            self._compile_prefill_invocation(request, request_sampling)
                        )
                    )
                continue
            key = request.graph_key(sampling_path)
            trace_eligible = self._prefill_trace_sampling_path in (None, sampling_path)
            with self._validation_context("prefill"):
                self._graph_compiler.compile(
                    key,
                    lambda context, request=request, request_sampling=request_sampling: (
                        self._compile_prefill_invocation(request, request_sampling)
                    ),
                    output_spec=_owned_invocation_output_spec,
                    trace_eligible=trace_eligible,
                )
            if trace_eligible:
                self._trace_plans.setdefault(
                    key,
                    self._make_prefill_trace_plan(request, request_sampling),
                )
        return None

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[list[Any]] | None = None,
        sampling_params: SamplingParams | None = None,
        enable_trace: bool | None = None,
    ) -> None:
        self._ensure_active()
        self._validate_decode_inputs(tokens, start_pos, page_table)
        self._validate_sampling_request(sampling_params)
        trace_requested = self._resolve_trace_hint("decode", enable_trace)
        page_table = self._normalize_decode_page_table(
            page_table,
            start_pos,
            allow_one_step_feedback_lag=self._use_device_feedback(sampling_params),
        )
        self._validate_bound_cache(kv_cache)
        sampling_path = self._decode_sampling_path(sampling_params, int(tokens.shape[0]))
        if sampling_params is not None:
            self._ensure_sampling_buffers()

        if not trace_requested:
            with self._validation_context("decode"):
                self._warm_eager_invocation(
                    lambda: self._compile_decode_invocation(
                        tokens,
                        start_pos,
                        page_table,
                        sampling_params,
                        sampling_path,
                    )
                )
            return None

        key = self._decode_graph_key(page_table, sampling_path)
        with self._validation_context("decode"):
            self._graph_compiler.compile(
                key,
                lambda context: self._compile_decode_invocation(
                    tokens,
                    start_pos,
                    page_table,
                    sampling_params,
                    sampling_path,
                ),
                output_spec=_owned_invocation_output_spec,
            )
        self._trace_plans.setdefault(
            key,
            self._make_decode_trace_plan(
                tokens,
                start_pos,
                page_table,
                sampling_params,
                sampling_path,
            ),
        )
        return None

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[list[Any]] | None = None,
        prompt_lens: torch.Tensor | None = None,
        empty_slots: list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        start_pos: torch.Tensor | None = None,
        enable_trace: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        self._ensure_active()
        self._validate_sampling_request(sampling_params)
        self._validate_bound_cache(kv_cache)
        self.mode = Mode.PREFILL
        trace_requested = self._resolve_trace_hint("prefill", enable_trace)

        requests = self._plan_prefill(tokens, page_table, prompt_lens, empty_slots, start_pos)
        batch_size = int(tokens.shape[0])
        sampling_path = "topk" if sampling_params is not None else "logits"
        output_logits = torch.zeros(batch_size, 1, int(self.model.vocab_size))
        output_tokens = torch.zeros(batch_size, dtype=torch.int32)
        row_log_probs: list[tuple[tuple[int, ...], Any]] = []

        for request in requests:
            request_sampling = _slice_sampling_params(sampling_params, request.source_rows)
            raw_output, owned = self._execute_prefill_request(
                request,
                request_sampling,
                sampling_path,
                enable_trace=trace_requested,
            )
            try:
                host_output = self._output_reader.read(raw_output, blocking=True)
                host_primary, host_log_probs = _split_output(host_output)
                if sampling_params is not None:
                    sampled = _process_output_tokens(
                        host_primary,
                        max(request.padded_batch_size, len(request.source_rows)),
                        self.cluster_shape,
                    )
                    for source_row, slot in zip(request.source_rows, request.slots):
                        token_index = slot if request.kind == "batched" else 0
                        output_tokens[source_row] = sampled.reshape(-1)[token_index].to(torch.int32)
                    if host_log_probs is not None:
                        row_log_probs.append((request.source_rows, host_log_probs))
                else:
                    if request.kind == "batched":
                        for source_row, slot in zip(request.source_rows, request.slots):
                            output_logits[source_row] = _process_output_prefill(
                                host_primary,
                                slot,
                                int(self.model.vocab_size),
                                self.cluster_shape,
                            )
                    else:
                        relative_last = (request.last_token_indices[0] - request.cached_tokens[0]) % 32
                        output_logits[request.source_rows[0]] = _process_output_prefill(
                            host_primary,
                            relative_last,
                            int(self.model.vocab_size),
                            self.cluster_shape,
                        )
            except BaseException as primary:
                failures = self._release_or_retain_transient(owned)
                _attach_cleanup_failures(primary, failures)
                raise
            failures = self._release_or_retain_transient(owned)
            if failures:
                _raise_cleanup_failures(failures)

        if sampling_params is not None:
            return output_tokens, _merge_log_probs(row_log_probs, batch_size)
        return output_logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[list[Any]] | None = None,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
        reset_batch: bool = False,
        enable_trace: bool | None = None,
    ) -> tuple[Any, Any]:
        self._ensure_active()
        self._validate_decode_inputs(tokens, start_pos, page_table)
        self._validate_sampling_request(sampling_params)
        trace_requested = self._resolve_trace_hint("decode", enable_trace)
        page_table = self._normalize_decode_page_table(
            page_table,
            start_pos,
            allow_one_step_feedback_lag=self._use_device_feedback(sampling_params),
        )
        self._validate_bound_cache(kv_cache)
        self.mode = Mode.DECODE

        batch_size = int(tokens.shape[0])
        sampling_path = self._decode_sampling_path(sampling_params, batch_size)
        raw_output, owned = self._execute_decode_request(
            tokens,
            start_pos,
            page_table,
            sampling_params,
            sampling_path,
            enable_trace=trace_requested,
            reset_batch=reset_batch,
        )
        self._previous_decode_page_table = page_table.clone()

        if not read_from_device:
            if owned is not None:
                record = _ExternalDecodeOutput(raw_value=raw_output, owned_values=owned)
                self._external_by_raw_id[id(raw_output)] = record
            return raw_output

        try:
            host_output = self._output_reader.read(raw_output, blocking=True)
            normalized = self._normalize_decode_host_output(
                host_output,
                is_tokens=sampling_params is not None,
            )
        except BaseException as primary:
            failures = self._release_or_retain_transient(owned)
            _attach_cleanup_failures(primary, failures)
            raise
        failures = self._release_or_retain_transient(owned)
        if failures:
            _raise_cleanup_failures(failures)
        return normalized

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        self._ensure_active()
        if not async_read:
            host = self._output_reader.read(tt_out, blocking=True)
            self._release_external_record(self._external_by_raw_id.get(id(tt_out)))
            return host

        pending = self._output_reader.submit(tt_out)
        record = self._external_by_raw_id.get(id(tt_out))
        if record is not None:
            record.host_value = pending.value
            record.pending = pending
            self._external_by_host_id[id(pending.value)] = record
        return pending.value, list(pending.events)

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> tuple[Any, Any]:
        completed = self._output_reader.complete(tt_out)
        record = self._external_by_host_id.get(id(tt_out))
        self._release_external_record(record)
        return self._normalize_decode_host_output(completed, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace: bool,
        can_sample_on_device: bool,
        **kwargs,
    ) -> None:
        self._ensure_active()
        self._validate_warmup_hints("prefill", enable_trace, can_sample_on_device)
        self._validate_bound_cache(kv_cache)
        if self.already_warmed_up_prefill:
            return
        if enable_trace:
            self._prefill_trace_sampling_path = "topk" if can_sample_on_device else "logits"

        if can_sample_on_device:
            self._ensure_sampling_buffers()

        sequence_lengths = self._configured_prefill_sequence_lengths()
        max_batch_size = int(self.model.config.max_batch_size)
        block_size = int(self.paged_kv_cache_config.block_size)
        raw_page_table_width = self._page_table_widths()[0]

        try:
            for sequence_length in sequence_lengths:
                configured_batches = self.config.warmup.prefill_batch_sizes if sequence_length == 128 else (1,)
                for batch_size in configured_batches:
                    if batch_size > max_batch_size:
                        continue
                    tokens = torch.zeros((batch_size, sequence_length), dtype=torch.long)
                    prompt_lens = torch.full((batch_size,), sequence_length, dtype=torch.long)
                    width = _num_blocks(sequence_length, block_size)
                    page_table = torch.zeros((batch_size, width), dtype=torch.int32)
                    self.compile_prefill(
                        tokens=tokens,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        prompt_lens=prompt_lens,
                        empty_slots=list(range(batch_size)),
                        sampling_params=None,
                        enable_trace=enable_trace,
                    )
                    if can_sample_on_device:
                        self.compile_prefill(
                            tokens=tokens,
                            page_table=page_table,
                            kv_cache=kv_cache,
                            prompt_lens=prompt_lens,
                            empty_slots=list(range(batch_size)),
                            sampling_params=_greedy_sampling_params(batch_size),
                            enable_trace=enable_trace,
                        )

                cached_tokens = block_size
                prompt_length = cached_tokens + sequence_length
                if prompt_length <= raw_page_table_width * block_size:
                    tokens = torch.zeros((1, prompt_length), dtype=torch.long)
                    prompt_lens = torch.tensor([prompt_length], dtype=torch.long)
                    page_table = torch.zeros((1, _num_blocks(prompt_length, block_size)), dtype=torch.int32)
                    start_pos = torch.tensor([cached_tokens], dtype=torch.long)
                    self.compile_prefill(
                        tokens=tokens,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        prompt_lens=prompt_lens,
                        empty_slots=[0],
                        start_pos=start_pos,
                        sampling_params=None,
                        enable_trace=enable_trace,
                    )
                    if can_sample_on_device:
                        self.compile_prefill(
                            tokens=tokens,
                            page_table=page_table,
                            kv_cache=kv_cache,
                            prompt_lens=prompt_lens,
                            empty_slots=[0],
                            start_pos=start_pos,
                            sampling_params=_greedy_sampling_params(1),
                            enable_trace=enable_trace,
                        )
            if enable_trace:
                self._maybe_capture_traces("prefill")
        except BaseException:
            self.already_warmed_up_prefill = False
            raise
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        kv_cache,
        enable_trace: bool,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        **kwargs,
    ) -> None:
        self._ensure_active()
        self._validate_warmup_hints("decode", enable_trace, can_sample_on_device)
        self._validate_bound_cache(kv_cache)
        lane_batch = int(self.model.config.max_batch_size)
        if int(max_batch_size) != lane_batch:
            raise ValueError(f"decode warmup batch {max_batch_size} does not match lane capacity {lane_batch}")
        if int(num_blocks) <= 0:
            raise ValueError("decode warmup num_blocks must be positive")

        if can_sample_on_device:
            self._ensure_sampling_buffers()

        tokens = torch.zeros(lane_batch, dtype=torch.long)
        start_pos = torch.zeros(lane_batch, dtype=torch.long)
        page_table = torch.zeros((lane_batch, int(num_blocks)), dtype=torch.int32)
        self.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=None,
            enable_trace=enable_trace,
        )
        if can_sample_on_device:
            self.compile_decode(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=_greedy_sampling_params(lane_batch),
                enable_trace=enable_trace,
            )
            if self.config.warmup.include_decode_top_k:
                self.compile_decode(
                    tokens=tokens,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    sampling_params=_topk_sampling_params(lane_batch),
                    enable_trace=enable_trace,
                )
        if enable_trace:
            self._maybe_capture_traces("decode")

    def cleanup(self) -> None:
        self._terminal = True
        if self._cleaned_up:
            return

        failures = self._drain_external_decode_outputs()
        try:
            self._output_reader.drain()
        except BaseException as error:
            failures.append(error)
        failures.extend(self._release_transient_orphans())
        if failures:
            _raise_cleanup_failures(failures)

        try:
            self._graph_compiler.cleanup()
        except BaseException:
            raise

        if self.config.device_sampling_enabled:
            self.model.sampling.release()

        self._kv_manager.release()
        self._trace_plans.clear()
        self._cleaned_up = True

    def _plan_prefill(self, tokens, page_table, prompt_lens, empty_slots, start_pos):
        raw_page_table_width, prefill_page_table_width, _ = self._page_table_widths()
        return _plan_prefill_requests(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            block_size=int(self.paged_kv_cache_config.block_size),
            max_batch_size=int(self.model.config.max_batch_size),
            max_prefill_chunk_size=int(self.runtime_config.max_prefill_chunk_size),
            can_enable_trace=self.runtime_config.can_enable_trace,
            max_actual_page_table_width=raw_page_table_width,
            canonical_page_table_width=prefill_page_table_width,
        )

    def _configured_prefill_sequence_lengths(self) -> tuple[int, ...]:
        sequence_lengths = self.config.warmup.prefill_seq_lens
        if sequence_lengths is None:
            sequence_lengths = tuple(getattr(self.runtime_config, "trace_prefill_supported_seq_lens", ()) or (128,))
        return tuple(int(value) for value in sequence_lengths)

    def _page_table_widths(self) -> tuple[int, int, int]:
        block_size = int(self.paged_kv_cache_config.block_size)
        model_width = _num_blocks(int(self.model.config.max_seq_len), block_size)
        physical_width = self.paged_kv_cache_config.num_blocks
        if physical_width is None:
            physical_width = self.paged_kv_cache_config.max_num_blocks
        raw_width = min(model_width, int(physical_width))
        decode_width = _round_up(raw_width, _PAGE_TABLE_WIDTH_ALIGNMENT)

        max_prefill_length = max(self._configured_prefill_sequence_lengths())
        padding_blocks = _num_blocks(max_prefill_length - 1, block_size)
        prefill_width = _round_up(raw_width + padding_blocks, _PAGE_TABLE_WIDTH_ALIGNMENT)
        return raw_width, prefill_width, decode_width

    def _normalize_decode_page_table(
        self,
        page_table: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        allow_one_step_feedback_lag: bool = False,
    ) -> torch.Tensor:
        raw_width, _, decode_width = self._page_table_widths()
        normalized = torch.zeros(
            (int(page_table.shape[0]), decode_width),
            dtype=torch.int32,
            device=page_table.device,
        )
        block_size = int(self.paged_kv_cache_config.block_size)
        for row, position in enumerate(start_pos):
            position = int(position)
            used_blocks = _num_blocks(max(0, position + 1), block_size)
            if used_blocks > raw_width:
                raise ValueError("decode position exceeds the configured paged-KV capacity")
            if int(page_table.shape[1]) < used_blocks:
                raise ValueError(f"page table is too narrow for decode row {row}")
            copy_blocks = used_blocks
            # Async scheduling can submit one decode before prior device feedback reaches the host.
            # At a block end, preserve the next mapping that the device-side position may already need.
            if allow_one_step_feedback_lag and position >= 0 and (position + 1) % block_size == 0:
                copy_blocks = min(used_blocks + 1, raw_width, int(page_table.shape[1]))
            normalized[row, :copy_blocks] = page_table[row, :copy_blocks].to(torch.int32)
        return normalized

    def _compile_prefill_invocation(self, request, sampling_params):
        try:
            output, owned = self._run_prefill_eager(request, sampling_params)
        except BaseException:
            raise
        return {"output": output, "owned": owned}

    def _compile_decode_invocation(self, tokens, start_pos, page_table, sampling_params, sampling_path):
        output, owned = self._run_decode_eager(
            tokens,
            start_pos,
            page_table,
            sampling_params,
            sampling_path,
            device_feedback=self._use_device_feedback(sampling_params),
        )
        return {"output": output, "owned": owned}

    def _execute_prefill_request(self, request, sampling_params, sampling_path, *, enable_trace):
        if not enable_trace or not request.trace_eligible:
            return self._run_prefill_eager(request, sampling_params)

        requested_key = request.graph_key(sampling_path)
        requested_graph = self._graph_compiler.assert_executable(requested_key)
        canonical_key = request.graph_key(self._prefill_trace_sampling_path or sampling_path)
        graph = (
            requested_graph if canonical_key == requested_key else self._graph_compiler.assert_executable(canonical_key)
        )
        hidden = self._graph_compiler.replay(
            canonical_key,
            lambda artifact, decision: self._refresh_prefill_trace(
                artifact,
                request,
                sampling_params,
                decision,
            ),
            reset_batch=True,
        )
        if graph.trace is None:
            raise RuntimeError(f"Required prefill trace {canonical_key!r} was not captured")
        values = graph.trace.persistent_inputs.values
        output = self._finish_traceable_prefill(
            request,
            hidden,
            sampling_params,
            values["kpt"],
            values["position_inputs"],
        )
        return output, (output,)

    def _execute_decode_request(
        self,
        tokens,
        start_pos,
        page_table,
        sampling_params,
        sampling_path,
        *,
        enable_trace,
        reset_batch,
    ):
        if not enable_trace:
            return self._run_decode_eager(
                tokens,
                start_pos,
                page_table,
                sampling_params,
                sampling_path,
                device_feedback=False,
            )

        key = self._decode_graph_key(page_table, sampling_path)
        page_table_changed = self._previous_decode_page_table is None or not torch.equal(
            self._previous_decode_page_table, page_table
        )
        feedback = self._use_device_feedback(sampling_params)
        output = self._graph_compiler.replay(
            key,
            lambda artifact, decision: self._refresh_decode_trace(
                artifact,
                tokens,
                start_pos,
                page_table,
                sampling_params,
                sampling_path,
                decision,
            ),
            reset_batch=reset_batch,
            device_feedback_enabled=self.device_decode_feedback_enabled,
            feedback_compatible=feedback,
            page_table_changed=page_table_changed,
        )
        return output, None

    def _run_prefill_eager(self, request, sampling_params):
        if request.trace_eligible:
            relative_last = max(
                last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)
            )
            host_inputs = self._prepare_prefill_inputs_host(
                request.tokens,
                request.page_table,
                last_token_idx=max(request.last_token_indices),
            )
            device_inputs, position_inputs, kpt = self._stage_prefill_inputs_and_kpt(
                host_inputs,
                sampling_params,
                self._prefill_sampling_batch_size(request),
                relative_last=relative_last,
                sequence_length=request.padded_sequence_length,
                force_topk=True,
            )
            hidden = None
            owned = (device_inputs, position_inputs, kpt)
            try:
                hidden = self._run_prefill_hidden_body(request, device_inputs)
                output = self._finish_traceable_prefill(
                    request,
                    hidden,
                    sampling_params,
                    kpt,
                    position_inputs,
                )
            except BaseException as primary:
                failures = self._release_or_retain_transient((hidden, owned))
                _attach_cleanup_failures(primary, failures)
                raise
            return output, (output, hidden, owned)
        output, owned = self._run_untraceable_prefill(request, sampling_params)
        return output, (output, owned)

    def _run_untraceable_prefill(self, request, sampling_params):
        owned_inputs = []
        output = None
        position_inputs = None
        kpt = self._make_device_kpt(
            sampling_params,
            self._prefill_sampling_batch_size(request),
            force_topk=True,
        )
        try:
            tokens = request.tokens
            cached = request.cached_tokens[0]
            relative_last = request.last_token_indices[0] - cached
            use_chunking = request.padded_sequence_length > int(self.runtime_config.max_prefill_chunk_size)
            use_prefix = cached > 0

            if use_chunking or use_prefix:
                chunk_size = (
                    _max_prefill_chunk_size(
                        request.padded_sequence_length,
                        int(self.runtime_config.max_prefill_chunk_size),
                    )
                    if use_chunking
                    else request.padded_sequence_length
                )
                last_chunk_start = (relative_last // chunk_size) * chunk_size
                chunk_relative_last = relative_last % chunk_size
                for chunk_start in range(cached, cached + request.padded_sequence_length, chunk_size):
                    relative_start = chunk_start - cached
                    relative_end = relative_start + chunk_size
                    chunk_tokens = tokens[:, relative_start:relative_end]
                    block_size = int(self.paged_kv_cache_config.block_size)
                    chunk_start_block = chunk_start // block_size
                    chunk_width = _num_blocks(chunk_size, block_size)
                    mapped_blocks = min(
                        chunk_width,
                        max(0, _num_blocks(request.prompt_lengths[0], block_size) - chunk_start_block),
                    )
                    chunk_page_table = torch.full(
                        (int(request.page_table.shape[0]), chunk_width),
                        -1,
                        dtype=torch.int32,
                        device=request.page_table.device,
                    )
                    if mapped_blocks:
                        chunk_page_table[:, :mapped_blocks] = request.page_table[
                            :, chunk_start_block : chunk_start_block + mapped_blocks
                        ]
                    host_inputs = self._prepare_prefill_inputs_host(
                        chunk_tokens,
                        request.page_table,
                        start_pos=chunk_start,
                        chunk_page_table=chunk_page_table,
                        chunk_start_idx=chunk_start,
                        last_token_idx=request.last_token_indices[0],
                    )
                    device_inputs = self._stage_prefill_device_inputs(host_inputs)
                    owned_inputs.append(device_inputs)
                    position_inputs = _copy_host_to_device(
                        self._prepare_prefill_position_inputs_host(chunk_relative_last, chunk_size),
                        mesh_device=self.mesh_device,
                    )
                    owned_inputs.append(position_inputs)
                    output = self.model.prefill_forward(
                        self.model.embed_prefill(device_inputs[0]),
                        [device_inputs[1], device_inputs[2]],
                        user_id=0,
                        page_table=device_inputs[3],
                        chunk_page_table=device_inputs[4],
                        chunk_start_idx=chunk_start,
                        get_last_token=-1,
                        chunk_start_idx_tensor=device_inputs[6],
                        last_token_slice=(position_inputs[0], position_inputs[1]),
                        last_token_index=position_inputs[2] if sampling_params is not None else None,
                    )
                    if relative_start == last_chunk_start:
                        break
                    _deallocate_owned_ttnn(output)
                    output = None
            else:
                host_inputs = self._prepare_prefill_inputs_host(
                    tokens,
                    request.page_table,
                    last_token_idx=request.last_token_indices[0],
                )
                device_inputs = self._stage_prefill_device_inputs(host_inputs)
                owned_inputs.append(device_inputs)
                position_inputs = _copy_host_to_device(
                    self._prepare_prefill_position_inputs_host(relative_last, request.padded_sequence_length),
                    mesh_device=self.mesh_device,
                )
                owned_inputs.append(position_inputs)
                output = self.model.prefill_forward(
                    self.model.embed_prefill(device_inputs[0]),
                    [device_inputs[1], device_inputs[2]],
                    user_id=0,
                    page_table=device_inputs[3],
                    get_last_token=-1,
                    last_token_slice=(position_inputs[0], position_inputs[1]),
                    last_token_index=position_inputs[2] if sampling_params is not None else None,
                )

            if sampling_params is not None:
                selected = _pad_prefill_logits(output, self.model.sampling)
                output = self._sample_device(selected, kpt)
            else:
                output = ttnn.untilize(output, use_multicore=True)
            return output, (owned_inputs, kpt)
        except BaseException as primary:
            failures = self._release_or_retain_transient((output, owned_inputs, kpt))
            _attach_cleanup_failures(primary, failures)
            raise

    def _run_prefill_hidden_body(self, request, device_inputs):
        return self.model.prefill_forward(
            self.model.embed_prefill(device_inputs[0]),
            [device_inputs[1], device_inputs[2]],
            user_id=list(range(request.padded_batch_size)) if request.kind == "batched" else 0,
            page_table=device_inputs[3],
            chunk_page_table=device_inputs[4],
            get_last_token=-1,
            batch_size=request.padded_batch_size,
            chunk_start_idx_tensor=device_inputs[6],
        )

    def _finish_traceable_prefill(self, request, hidden, sampling_params, kpt, position_inputs):
        relative_last = [last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)]
        if request.kind == "batched" or sampling_params is not None:
            padded_last = list(relative_last) + [0] * (request.padded_batch_size - len(relative_last))
            logits = self.model.post_process_batched_prefill_output(
                hidden,
                padded_last,
                request.padded_batch_size,
                request.padded_sequence_length,
                last_token_slice=(position_inputs[0], position_inputs[1]),
                last_token_index=position_inputs[2],
            )
        else:
            logits = self.model.post_process_prefill_output(
                hidden,
                relative_last[0],
                last_token_slice=(position_inputs[0], position_inputs[1]),
            )

        if sampling_params is not None:
            logits = _pad_prefill_logits(logits, self.model.sampling)
            return self._sample_device(logits, kpt)
        return ttnn.untilize(logits, use_multicore=True)

    def _run_decode_eager(
        self,
        tokens,
        start_pos,
        page_table,
        sampling_params,
        sampling_path,
        *,
        device_feedback,
    ):
        host_inputs = self._prepare_decode_inputs_host(tokens, start_pos, page_table)
        device_inputs, kpt = self._stage_device_inputs_and_kpt(
            host_inputs,
            sampling_params,
            int(tokens.shape[0]),
            force_topk=False,
        )
        owned = (device_inputs, kpt)
        try:
            output = self._run_decode_body(
                device_inputs,
                sampling_params,
                kpt,
                device_feedback=device_feedback,
            )
        except BaseException as primary:
            failures = self._release_or_retain_transient(owned)
            _attach_cleanup_failures(primary, failures)
            raise
        return output, (output, owned)

    def _run_decode_body(self, device_inputs, sampling_params, kpt, *, device_feedback):
        tt_tokens, tt_current_pos, tt_rot_idxs, tt_page_table = device_inputs
        rot_mats = self.model.rope_setup.get_rot_mats(tt_rot_idxs)
        logits = self.model.decode_forward(
            self.model.embed_decode(tt_tokens),
            tt_current_pos,
            rot_mats,
            page_table=tt_page_table,
        )
        if sampling_params is not None:
            output = self._sample_device(logits, kpt)
            if device_feedback:
                sampled_tokens = ttnn.reshape(output[0], tt_tokens.shape)
                ttnn.copy(input_a=sampled_tokens, input_b=tt_tokens)
                self.model.increment_positions(tt_current_pos, tt_rot_idxs)
            return output
        return self.model.gather_and_untilize_logits(logits), None

    def _make_prefill_trace_plan(self, request, sampling_params):
        def prepare():
            relative_last = max(
                last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)
            )
            host_inputs = self._prepare_prefill_inputs_host(
                request.tokens,
                request.page_table,
                last_token_idx=max(request.last_token_indices),
            )
            device_inputs, position_inputs, kpt = self._stage_prefill_inputs_and_kpt(
                host_inputs,
                sampling_params,
                self._prefill_sampling_batch_size(request),
                relative_last=relative_last,
                sequence_length=request.padded_sequence_length,
                force_topk=True,
            )
            return {"device_inputs": device_inputs, "position_inputs": position_inputs, "kpt": kpt}

        def capture(persistent: PersistentInputs):
            values = persistent.values
            with suspend_module_input_validation():
                return self._run_prefill_hidden_body(request, values["device_inputs"])

        return TraceCapturePlan(
            prepare_inputs=prepare,
            capture=capture,
            refresh_policy=InputRefreshPolicy(
                every_replay=("tokens", "page_table", "last_token", "sampling"),
            ),
        )

    def _make_decode_trace_plan(
        self,
        tokens,
        start_pos,
        page_table,
        sampling_params,
        sampling_path,
    ):
        def prepare():
            host_inputs = self._prepare_decode_inputs_host(tokens, start_pos, page_table)
            device_inputs, kpt = self._stage_device_inputs_and_kpt(
                host_inputs,
                sampling_params,
                int(tokens.shape[0]),
                force_topk=False,
            )
            return {"device_inputs": device_inputs, "kpt": kpt}

        def capture(persistent: PersistentInputs):
            values = persistent.values
            with suspend_module_input_validation():
                return self._run_decode_body(
                    values["device_inputs"],
                    sampling_params,
                    values["kpt"],
                    device_feedback=self._use_device_feedback(sampling_params),
                )

        return TraceCapturePlan(
            prepare_inputs=prepare,
            capture=capture,
            refresh_policy=InputRefreshPolicy(
                every_replay=("sampling",),
            ),
        )

    def _refresh_prefill_trace(self, artifact, request, sampling_params, decision):
        values = artifact.persistent_inputs.values
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        host_inputs = self._prepare_prefill_inputs_host(
            request.tokens,
            request.page_table,
            last_token_idx=max(request.last_token_indices),
        )
        for host_index, device_index in ((0, 0), (2, 3), (3, 4), (4, 6)):
            host_tensor = host_inputs[host_index]
            device_tensor = values["device_inputs"][device_index]
            if host_tensor is not None:
                ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        position_inputs = self._prepare_prefill_position_inputs_host(
            relative_last,
            request.padded_sequence_length,
        )
        _copy_host_to_device(position_inputs, values["position_inputs"])
        if sampling_params is not None:
            self._refresh_kpt(values["kpt"], sampling_params, self._prefill_sampling_batch_size(request), True)

    def _refresh_decode_trace(
        self,
        artifact,
        tokens,
        start_pos,
        page_table,
        sampling_params,
        sampling_path,
        decision,
    ):
        values = artifact.persistent_inputs.values
        if decision.full:
            host_inputs = self._prepare_decode_inputs_host(tokens, start_pos, page_table)
            _copy_host_to_device(host_inputs, values["device_inputs"])
        elif decision.page_table:
            host_inputs = self._prepare_decode_inputs_host(tokens, start_pos, page_table)
            ttnn.copy_host_to_device_tensor(host_inputs[3], values["device_inputs"][3])
        self._refresh_kpt(values["kpt"], sampling_params, int(tokens.shape[0]), False)

    def _prepare_prefill_inputs_host(
        self,
        tokens,
        page_table,
        *,
        start_pos=0,
        chunk_page_table=None,
        chunk_start_idx=None,
        last_token_idx=None,
    ):
        if tokens.ndim != 2:
            raise ValueError("prefill tokens must be rank 2")
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tokens_tt = ttnn.from_torch(
            tokens.reshape(1, 1, 1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )

        rope = self.model.rope_setup
        rope.load_device_weights()
        matrix_length = int(rope.cos_matrix.shape[2])
        if matrix_length <= 0:
            raise ValueError("rotary position table must not be empty")
        start_pos = int(start_pos)
        sequence_length = int(tokens.shape[-1])
        real_end = start_pos + sequence_length
        if start_pos < 0:
            raise ValueError("prefill start position must be nonnegative")
        if last_token_idx is not None and int(last_token_idx) + 1 > matrix_length:
            raise ValueError(f"Sequence length {int(last_token_idx) + 1} exceeds rotary capacity {matrix_length}")
        position_indices = torch.arange(start_pos, real_end, dtype=torch.long).clamp(max=matrix_length - 1)
        position_indices_tt = ttnn.from_torch(
            position_indices.reshape(1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )

        page_table_tt = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        chunk_tt = None
        if chunk_page_table is not None:
            chunk_tt = ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
        chunk_start_tt = None
        if chunk_start_idx is not None:
            chunk_start_tt = ttnn.from_torch(
                torch.tensor([int(chunk_start_idx)], dtype=torch.int32),
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
        return tokens_tt, position_indices_tt, page_table_tt, chunk_tt, chunk_start_tt

    def _prepare_prefill_position_inputs_host(self, relative_last, sequence_length):
        relative_last = int(relative_last)
        sequence_length = int(sequence_length)
        if relative_last < 0 or relative_last >= sequence_length:
            raise ValueError("prefill last-token position must fall within the padded sequence")
        block_start = (relative_last // 32) * 32
        hidden_width = int(self.model.config.dim)
        bounds = (
            (0, 0, block_start, 0),
            (1, 1, block_start + 32, hidden_width),
        )
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        slice_bounds = tuple(
            ttnn.from_torch(
                torch.tensor(bound, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            for bound in bounds
        )
        row_index = ttnn.from_torch(
            torch.tensor([[relative_last % 32]], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        return (*slice_bounds, row_index)

    def _prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        batch_size = int(tokens.shape[0])
        if batch_size > 32:
            raise ValueError("decode token input padding supports at most 32 lane slots")
        padded_tokens = torch.nn.functional.pad(tokens.reshape(-1), (0, 32 - batch_size))
        tokens_tt = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(
                padded_tokens,
                device=None,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )
        nonnegative_pos = torch.maximum(current_pos, torch.zeros_like(current_pos))
        rope_indices = self.model.rope_setup.get_rot_idxs(nonnegative_pos, on_host=True)
        mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            dims=(None, None),
            mesh_shape=self.cluster_shape,
        )
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=mapper,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=mapper,
        )
        return tokens_tt, current_pos_tt, rope_indices, page_table_tt

    def _sample_device(self, logits, kpt):
        if kpt is None:
            return self.model.sampling.decode_forward(logits, tt_out_tok=None)
        return self.model.sampling.decode_forward(
            logits,
            k=kpt[0],
            p=kpt[1],
            temp=kpt[2],
            tt_out_tok=None,
        )

    def _stage_device_inputs_and_kpt(
        self,
        host_inputs,
        sampling_params,
        batch_size,
        *,
        force_topk,
    ):
        device_inputs = None
        try:
            device_inputs = _copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
            kpt = self._make_device_kpt(sampling_params, batch_size, force_topk)
        except BaseException as primary:
            failures = self._release_or_retain_transient(device_inputs)
            _attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, kpt

    def _stage_prefill_device_inputs(self, host_inputs):
        raw_inputs = None
        rot_mats = None
        try:
            raw_inputs = _copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
            prepare_rot_mats = getattr(self.model, "prepare_prefill_rot_mats", None)
            if not callable(prepare_rot_mats):
                raise TypeError("model must provide prepare_prefill_rot_mats()")
            rot_mats = tuple(prepare_rot_mats(raw_inputs[1]))
            if len(rot_mats) != 2:
                raise ValueError("prepare_prefill_rot_mats() must return cosine and sine tensors")
        except BaseException as primary:
            failures = self._release_or_retain_transient((rot_mats, raw_inputs))
            _attach_cleanup_failures(primary, failures)
            raise
        return (
            raw_inputs[0],
            rot_mats[0],
            rot_mats[1],
            raw_inputs[2],
            raw_inputs[3],
            raw_inputs[1],
            raw_inputs[4],
        )

    def _stage_prefill_inputs_and_kpt(
        self,
        host_inputs,
        sampling_params,
        batch_size,
        *,
        relative_last,
        sequence_length,
        force_topk,
    ):
        device_inputs = None
        position_inputs = None
        kpt = None
        try:
            device_inputs = self._stage_prefill_device_inputs(host_inputs)
            position_inputs = _copy_host_to_device(
                self._prepare_prefill_position_inputs_host(relative_last, sequence_length),
                mesh_device=self.mesh_device,
            )
            kpt = self._make_device_kpt(sampling_params, batch_size, force_topk)
        except BaseException as primary:
            failures = self._release_or_retain_transient((device_inputs, position_inputs, kpt))
            _attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, position_inputs, kpt

    def _make_device_kpt(self, sampling_params, batch_size, force_topk):
        host = self._make_host_kpt(sampling_params, batch_size, force_topk)
        if host is None:
            return None
        return tuple(_copy_host_to_device(host, mesh_device=self.mesh_device))

    def _make_host_kpt(self, sampling_params, batch_size, force_topk):
        if sampling_params is None:
            return None
        allow_argmax = bool(self.model.sampling.config.allow_force_argmax) and not force_topk
        values = _formatted_sampling_values(sampling_params, batch_size)
        if allow_argmax and values[3]:
            return None
        k, p, temperature, _ = values
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return (
            ttnn.from_torch(
                torch.tensor(k, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(p, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(temperature, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )

    def _refresh_kpt(self, device_kpt, sampling_params, batch_size, force_topk):
        host_kpt = self._make_host_kpt(sampling_params, batch_size, force_topk)
        if (host_kpt is None) != (device_kpt is None):
            raise RuntimeError("sampling parameters changed the compiled sampling path")
        if host_kpt is None:
            return
        _copy_host_to_device(host_kpt, device_kpt)

    def _decode_sampling_path(self, sampling_params, batch_size):
        if sampling_params is None:
            return "logits"
        values = _formatted_sampling_values(sampling_params, batch_size)
        if bool(self.model.sampling.config.allow_force_argmax) and values[3]:
            return "argmax"
        return "topk"

    def _decode_graph_key(self, page_table, sampling_path):
        return GraphKey(
            mode="decode",
            batch_size=int(self.model.config.max_batch_size),
            page_table_width=int(page_table.shape[-1]),
            sampling_path=sampling_path,
        )

    def _prefill_sampling_batch_size(self, request):
        if not self.config.device_sampling_enabled:
            return request.padded_batch_size
        return int(self.model.sampling.config.max_batch_size)

    def _ensure_sampling_buffers(self):
        if self._sampling_buffers_loaded:
            return
        if self._traces_captured:
            raise RuntimeError("cannot materialize sampling buffers after trace activation")
        self.model.sampling.load_device_buffers()
        self._sampling_buffers_loaded = True

    def _use_device_feedback(self, sampling_params):
        return (
            self.device_decode_feedback_enabled
            and sampling_params is not None
            and hasattr(self.model, "increment_positions")
        )

    def _validate_sampling_request(self, sampling_params):
        if sampling_params is not None and not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")

    def _validate_bound_cache(self, kv_cache):
        if self._kv_manager.bound_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before execution")
        if kv_cache is not None:
            self._kv_manager.validate_borrowed_handle(kv_cache)

    def _validate_decode_inputs(self, tokens, start_pos, page_table):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ValueError("decode tokens must be a rank-1 torch.Tensor")
        if not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1:
            raise ValueError("decode start_pos must be a rank-1 torch.Tensor")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
            raise ValueError("decode page_table must be a rank-2 torch.Tensor")
        expected = int(self.model.config.max_batch_size)
        if int(tokens.shape[0]) != expected:
            raise ValueError(f"decode batch {tokens.shape[0]} must equal lane capacity {expected}")
        if int(start_pos.shape[0]) != expected or int(page_table.shape[0]) != expected:
            raise ValueError("decode tokens, start_pos, and page_table batches must match")

    def _validate_warmup_hints(self, mode, enable_trace, can_sample_on_device):
        if not isinstance(enable_trace, bool) or not isinstance(can_sample_on_device, bool):
            raise TypeError("warmup trace and sampling hints must be bool")
        configured_trace = self.config.trace.prefill_enabled if mode == "prefill" else self.config.trace.decode_enabled
        if enable_trace and not configured_trace:
            raise ValueError(f"enable_trace=True disagrees with static {mode} trace policy")
        if can_sample_on_device and not self.config.device_sampling_enabled:
            raise ValueError("can_sample_on_device=True disagrees with static sampling policy")

    def _resolve_trace_hint(self, mode, enable_trace):
        configured_trace = self.config.trace.prefill_enabled if mode == "prefill" else self.config.trace.decode_enabled
        if enable_trace is None:
            return configured_trace
        if not isinstance(enable_trace, bool):
            raise TypeError("enable_trace must be bool or None")
        if enable_trace and not configured_trace:
            raise ValueError(f"enable_trace=True disagrees with static {mode} trace policy")
        return enable_trace

    def _warm_eager_invocation(self, invoke):
        output = None
        try:
            output = invoke()
            ttnn.synchronize_device(self.mesh_device)
        except BaseException as primary:
            failures = self._release_or_retain_transient(output)
            try:
                ttnn.synchronize_device(self.mesh_device)
            except BaseException as error:
                failures.append(error)
            _attach_cleanup_failures(primary, failures)
            raise

        failures = self._release_or_retain_transient(output)
        try:
            ttnn.synchronize_device(self.mesh_device)
        except BaseException as primary:
            _attach_cleanup_failures(primary, failures)
            raise
        if failures:
            _raise_cleanup_failures(failures)

    def _maybe_capture_traces(self, mode):
        if self._graph_compiler.trace_active:
            return
        if mode not in ("prefill", "decode"):
            raise ValueError(f"Unknown trace warmup mode {mode!r}")
        if mode == "prefill" and self.config.trace.decode_enabled:
            return
        try:
            self._graph_compiler.capture_all(self._trace_plans)
        finally:
            self._traces_captured = self._graph_compiler.trace_active

    def _validation_context(self, mode):
        return validate_module_input_configs(
            model=self.model,
            iter_named_modules=lambda model: model.iter_executor_named_modules(),
            mode=mode,
        )

    def _normalize_decode_host_output(self, host_output, *, is_tokens):
        output, log_probs = _split_output(host_output)
        batch_size = int(self.model.config.max_batch_size)
        if is_tokens:
            tokens = _process_output_tokens(output, batch_size, self.cluster_shape)
            return tokens.to(torch.int32), log_probs
        logits = _process_output_decode_logits(
            output,
            batch_size,
            int(self.model.vocab_size),
            int(self.model.num_devices),
            self.cluster_shape,
        )
        return logits, log_probs

    def _release_external_record(self, record):
        if record is None or record.released:
            return
        if record.pending is not None:
            self._output_reader.complete(record.pending)
        failures = _best_effort_deallocate_owned_ttnn(
            (record.raw_value, record.owned_values),
            record.deallocated_tensor_ids,
        )
        if failures:
            _raise_cleanup_failures(failures)
        record.released = True
        self._external_by_raw_id.pop(id(record.raw_value), None)
        if record.host_value is not None:
            self._external_by_host_id.pop(id(record.host_value), None)

    def _drain_external_decode_outputs(self):
        failures = []
        records = tuple(self._external_by_raw_id.values())
        for record in records:
            try:
                if record.pending is None:
                    ttnn.synchronize_device(self.mesh_device)
                self._release_external_record(record)
            except BaseException as error:
                failures.append(error)
        return failures

    def _release_or_retain_transient(self, values):
        orphan = _TransientOrphan(values)
        failures = _best_effort_deallocate_owned_ttnn(
            orphan.values,
            orphan.deallocated_tensor_ids,
        )
        if failures:
            self._transient_orphans.append(orphan)
        return failures

    def _release_transient_orphans(self):
        failures = []
        remaining = []
        for orphan in self._transient_orphans:
            orphan_failures = _best_effort_deallocate_owned_ttnn(
                orphan.values,
                orphan.deallocated_tensor_ids,
            )
            if orphan_failures:
                failures.extend(orphan_failures)
                remaining.append(orphan)
        self._transient_orphans = remaining
        return failures

    def _ensure_active(self):
        if self._terminal:
            raise RuntimeError("LLMExecutor is terminal; construct a new executor")
        if self._transient_orphans:
            raise RuntimeError("LLMExecutor has unreleased transient resources; clean up this executor")


def _plan_prefill_requests(
    *,
    tokens,
    page_table,
    prompt_lens,
    empty_slots,
    start_pos,
    block_size,
    max_batch_size,
    max_prefill_chunk_size,
    can_enable_trace,
    max_actual_page_table_width=None,
    canonical_page_table_width=None,
):
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("prefill tokens must be a rank-2 torch.Tensor")
    if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
        raise ValueError("prefill page_table must be a rank-2 torch.Tensor")
    batch_size, token_width = map(int, tokens.shape)
    if int(page_table.shape[0]) != batch_size:
        raise ValueError("prefill token and page-table batches must match")

    if prompt_lens is None:
        prompt_lens = torch.full((batch_size,), token_width, dtype=torch.long)
    if not isinstance(prompt_lens, torch.Tensor) or prompt_lens.ndim != 1:
        raise ValueError("prompt_lens must be a rank-1 torch.Tensor")
    if int(prompt_lens.shape[0]) != batch_size:
        raise ValueError("prompt_lens batch must match tokens")

    if start_pos is None:
        start_pos = torch.zeros(batch_size, dtype=torch.long)
    if not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1:
        raise ValueError("start_pos must be a rank-1 torch.Tensor")
    if int(start_pos.shape[0]) != batch_size:
        raise ValueError("start_pos batch must match tokens")

    slots = list(range(batch_size)) if empty_slots is None else [int(slot) for slot in empty_slots]
    if len(slots) != batch_size:
        raise ValueError("empty_slots length must match prefill batch")
    if len(set(slots)) != len(slots) or any(slot < 0 or slot >= max_batch_size for slot in slots):
        raise ValueError("empty_slots must contain unique lane-local slots")
    if (max_actual_page_table_width is None) != (canonical_page_table_width is None):
        raise ValueError("canonical page-table widths must be provided together")
    if max_actual_page_table_width is not None:
        if max_actual_page_table_width <= 0 or canonical_page_table_width < max_actual_page_table_width:
            raise ValueError("invalid canonical page-table widths")
        if canonical_page_table_width % _PAGE_TABLE_WIDTH_ALIGNMENT:
            raise ValueError("canonical page-table width must be 8-entry aligned")

    lengths = [int(value) for value in prompt_lens]
    cached = [int(value) for value in start_pos]
    for row, (length, prefix) in enumerate(zip(lengths, cached)):
        if prefix < 0 or length < 0 or prefix > length or length > token_width:
            raise ValueError(f"invalid prompt/prefix lengths for prefill row {row}")
        if prefix % block_size:
            raise ValueError(f"cached prefill start for row {row} must be block aligned")
    new_lengths = [length - prefix for length, prefix in zip(lengths, cached)]
    padded_lengths = [_padded_prefill_length(length) if length > 0 else 0 for length in new_lengths]

    padded_batch = None
    if len(set(new_lengths)) == 1:
        padded_batch = _batched_prefill_size(
            batch_size,
            padded_lengths,
            cached,
            max_batch_size,
            max_prefill_chunk_size,
        )
    if padded_batch is not None and slots == list(range(batch_size)):
        sequence_length = padded_lengths[0]
        request_tokens = torch.zeros(
            (padded_batch, sequence_length),
            dtype=torch.long,
            device=tokens.device,
        )
        last_indices = []
        for source_row, slot in enumerate(slots):
            length = lengths[source_row]
            request_tokens[slot, :length] = tokens[source_row, :length]
            last_indices.append(length - 1)
        actual_width = max(_num_blocks(length, block_size) for length in lengths)
        page_width = canonical_page_table_width or _num_blocks(sequence_length, block_size)
        if max_actual_page_table_width is not None and actual_width > max_actual_page_table_width:
            raise ValueError("prefill prompt exceeds the configured paged-KV capacity")
        if int(page_table.shape[-1]) < actual_width:
            raise ValueError("page table is too narrow for batched prefill")
        request_page_table = torch.full(
            (padded_batch, page_width),
            -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        for source_row, slot in enumerate(slots):
            row_width = _num_blocks(lengths[source_row], block_size)
            request_page_table[slot, :row_width] = page_table[source_row, :row_width].to(torch.int32)
        return (
            _PrefillRequest(
                kind="batched",
                source_rows=tuple(range(batch_size)),
                slots=tuple(slots),
                tokens=request_tokens,
                page_table=request_page_table,
                prompt_lengths=tuple(lengths),
                cached_tokens=tuple(cached),
                last_token_indices=tuple(last_indices),
                padded_sequence_length=sequence_length,
                padded_batch_size=padded_batch,
                chunk_page_table_width=None,
                trace_eligible=bool(can_enable_trace(sequence_length, 0)),
            ),
        )

    requests = []
    for source_row, slot in enumerate(slots):
        new_length = new_lengths[source_row]
        if new_length <= 0:
            continue
        sequence_length = padded_lengths[source_row]
        request_tokens = torch.zeros(
            (1, sequence_length),
            dtype=torch.long,
            device=tokens.device,
        )
        request_tokens[0, :new_length] = tokens[
            source_row,
            cached[source_row] : lengths[source_row],
        ]
        trace_eligible = cached[source_row] == 0 and bool(can_enable_trace(sequence_length, cached[source_row]))
        actual_width = _num_blocks(lengths[source_row], block_size)
        page_width = canonical_page_table_width or _num_blocks(
            sequence_length if trace_eligible else sequence_length + cached[source_row],
            block_size,
        )
        if max_actual_page_table_width is not None and actual_width > max_actual_page_table_width:
            raise ValueError(f"prefill row {source_row} exceeds the configured paged-KV capacity")
        if cached[source_row] + sequence_length > page_width * block_size:
            raise ValueError(f"padded prefill row {source_row} exceeds the canonical page-table capacity")
        if int(page_table.shape[-1]) < actual_width:
            raise ValueError(f"page table is too narrow for prefill row {source_row}")
        use_chunk = sequence_length > max_prefill_chunk_size or cached[source_row] > 0
        request_page_table = torch.full(
            (1, page_width),
            0 if use_chunk else -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        request_page_table[0, :actual_width] = page_table[source_row, :actual_width].to(torch.int32)
        chunk_width = None
        if use_chunk:
            chunk_size = (
                _max_prefill_chunk_size(sequence_length, max_prefill_chunk_size)
                if sequence_length > max_prefill_chunk_size
                else sequence_length
            )
            chunk_width = _num_blocks(chunk_size, block_size)
        requests.append(
            _PrefillRequest(
                kind="single",
                source_rows=(source_row,),
                slots=(slot,),
                tokens=request_tokens,
                page_table=request_page_table,
                prompt_lengths=(lengths[source_row],),
                cached_tokens=(cached[source_row],),
                last_token_indices=(lengths[source_row] - 1,),
                padded_sequence_length=sequence_length,
                padded_batch_size=1,
                chunk_page_table_width=chunk_width,
                trace_eligible=trace_eligible,
            )
        )
    return tuple(requests)


def _copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    if device_tensors is None:
        if mesh_device is None:
            raise ValueError("mesh_device is required for device allocation")
        allocated = []
        try:
            for host_tensor in host_tensors:
                allocated.append(ttnn.to_device(host_tensor, device=mesh_device) if host_tensor is not None else None)
        except BaseException as primary:
            failures = _best_effort_deallocate_owned_ttnn(allocated)
            _attach_cleanup_failures(primary, failures)
            raise
        return allocated
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        if host_tensor is None:
            if device_tensor is not None:
                raise ValueError("host/device optional tensor structure changed")
            continue
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
    return device_tensors


def _owned_invocation_output_spec(value):
    return OutputSpec.from_value(_primary_output(value["output"]))


def _primary_output(value):
    if isinstance(value, tuple):
        for item in value:
            if item is not None:
                return item
    return value


def _split_output(value):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("runtime output tuple must contain (output, log_probs)")
        return value
    return value, None


def _slice_sampling_params(sampling_params, source_rows):
    if sampling_params is None:
        return None
    if not dataclasses.is_dataclass(sampling_params):
        raise TypeError("sampling_params must be a dataclass")

    def slice_value(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            selected_rows = (0,) * len(source_rows) if int(value.shape[0]) == 1 else source_rows
            indices = torch.tensor(selected_rows, dtype=torch.long, device=value.device)
            return value.index_select(0, indices)
        if isinstance(value, list):
            if len(value) == 1:
                return [value[0] for _ in source_rows]
            return [value[row] for row in source_rows]
        if isinstance(value, tuple):
            if len(value) == 1:
                return tuple(value[0] for _ in source_rows)
            return tuple(value[row] for row in source_rows)
        return value

    updates = {
        field.name: slice_value(getattr(sampling_params, field.name)) for field in dataclasses.fields(sampling_params)
    }
    return dataclasses.replace(sampling_params, **updates)


def _formatted_sampling_values(sampling_params, batch_size):
    formatted_size = ((int(batch_size) + 31) // 32) * 32
    formatted = format_sampling_params(sampling_params, formatted_size)
    k = tuple(int(value) for value in list(formatted.top_k)[:batch_size])
    p = tuple(float(value) for value in list(formatted.top_p)[:batch_size])
    temperature = tuple(float(value) for value in list(formatted.temperature)[:batch_size])
    greedy = (
        all(value == 1 for value in k) and all(value == 0 for value in p) and all(value == 1 for value in temperature)
    )
    return k, p, temperature, greedy


def _greedy_sampling_params(batch_size):
    return SamplingParams(
        temperature=[0.0] * batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
    )


def _topk_sampling_params(batch_size):
    return SamplingParams(
        temperature=[1.0] * batch_size,
        top_k=[32] * batch_size,
        top_p=[0.08] * batch_size,
    )


def _padded_prefill_length(sequence_length):
    if sequence_length <= 128:
        return 128
    if sequence_length <= 1024:
        return 1024
    return 1 << (sequence_length - 1).bit_length()


def _batched_prefill_size(
    batch_size,
    padded_lengths,
    cached_tokens,
    max_batch_size,
    max_prefill_chunk_size,
):
    if batch_size <= 1 or not padded_lengths or len(set(padded_lengths)) != 1:
        return None
    if any(value != 0 for value in cached_tokens):
        return None
    if padded_lengths[0] != 128 or padded_lengths[0] > max_prefill_chunk_size:
        return None
    padded_batch = next(
        (
            candidate
            for candidate in _SUPPORTED_PREFILL_BATCH_SIZES
            if candidate >= batch_size and candidate <= max_batch_size
        ),
        None,
    )
    if padded_batch is None and batch_size <= max_batch_size:
        padded_batch = max_batch_size
    if padded_batch is None or padded_batch * padded_lengths[0] >= _MAX_BATCHED_PREFILL_TOKENS:
        return None
    return padded_batch


def _max_prefill_chunk_size(sequence_length, maximum):
    minimum_chunk = 2048
    if sequence_length <= 0 or maximum <= 0:
        raise ValueError("prefill chunk lengths must be positive")
    if sequence_length % minimum_chunk or maximum % minimum_chunk:
        raise ValueError("prefill chunk lengths must be multiples of 2048")
    for chunk_size in range(min(sequence_length, maximum), 0, -minimum_chunk):
        if sequence_length % chunk_size == 0:
            return chunk_size
    raise ValueError("no valid prefill chunk size")


def _num_blocks(sequence_length, block_size):
    return math.ceil(int(sequence_length) / int(block_size))


def _round_up(value, alignment):
    return math.ceil(int(value) / int(alignment)) * int(alignment)


def _pad_prefill_logits(logits, sampler):
    target_batch = int(sampler.config.max_batch_size)
    current_batch = int(logits.shape[2])
    if current_batch >= target_batch:
        return logits
    return ttnn.pad(
        logits,
        [(0, 0), (0, 0), (0, target_batch - current_batch), (0, 0)],
        value=0.0,
    )


def _concat_host_output(value, cluster_shape):
    if isinstance(value, torch.Tensor):
        return value
    tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(value)]
    rows, columns = cluster_shape
    mesh = [tensors[index : index + columns] for index in range(0, len(tensors), columns)]
    return torch.cat([torch.cat(row, dim=-1) for row in mesh], dim=1)


def _process_output_prefill(value, row, vocab_size, cluster_shape):
    if isinstance(value, ttnn.Tensor) and value.storage_type() != ttnn.StorageType.HOST:
        raise ValueError("prefill output must be on host")
    output = _concat_host_output(value, cluster_shape)
    return output[0, 0, int(row), :vocab_size].float()


def _process_output_decode_logits(value, batch_size, vocab_size, num_devices, cluster_shape):
    if isinstance(value, torch.Tensor):
        output = value.float()
    elif num_devices > 1:
        output = _concat_host_output(value, cluster_shape).float()
    else:
        output = ttnn.to_torch(value).float()
    return output[:, :, :batch_size, :vocab_size].contiguous().view(batch_size, 1, -1)


def _process_output_tokens(value, batch_size, cluster_shape):
    output = _concat_host_output(value, cluster_shape)
    if output.ndim >= 4:
        if int(output.shape[2]) >= batch_size:
            output = output[0, 0, :batch_size, 0]
        elif int(output.shape[3]) >= batch_size:
            output = output[0, 0, 0, :batch_size]
    return output.reshape(-1)[:batch_size].to(torch.int32)


def _merge_log_probs(row_payloads, batch_size):
    if not row_payloads:
        return None
    if len(row_payloads) == 1 and row_payloads[0][0] == tuple(range(batch_size)):
        return row_payloads[0][1]
    ordered = [None] * batch_size
    for rows, payload in row_payloads:
        if isinstance(payload, torch.Tensor) and payload.shape[0] == len(rows):
            for local_row, source_row in enumerate(rows):
                ordered[source_row] = payload[local_row]
        else:
            for source_row in rows:
                ordered[source_row] = payload
    return ordered


def _deallocate_owned_ttnn(value, seen=None):
    if value is None:
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, ttnn.Tensor):
        ttnn.deallocate(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _deallocate_owned_ttnn(nested, seen)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            _deallocate_owned_ttnn(nested, seen)


def _best_effort_deallocate_owned_ttnn(value, completed=None):
    if completed is None:
        completed = set()
    failures = []
    visiting = set()

    def visit(item):
        if item is None:
            return
        identity = id(item)
        if isinstance(item, ttnn.Tensor):
            if identity in completed:
                return
            try:
                ttnn.deallocate(item)
            except BaseException as error:
                failures.append(error)
            else:
                completed.add(identity)
            return
        if identity in visiting:
            return
        if isinstance(item, dict):
            visiting.add(identity)
            for nested in item.values():
                visit(nested)
            visiting.remove(identity)
        elif isinstance(item, (list, tuple, set)):
            visiting.add(identity)
            for nested in item:
                visit(nested)
            visiting.remove(identity)

    visit(value)
    return failures


def _attach_cleanup_failures(primary, failures):
    if not failures:
        return
    previous = tuple(getattr(primary, "cleanup_failures", ()))
    primary.cleanup_failures = previous + tuple(failures)
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(f"cleanup also encountered {len(failures)} failure(s)")


def _raise_cleanup_failures(failures):
    primary = failures[0]
    if len(failures) > 1:
        primary.cleanup_failures = tuple(failures[1:])
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(f"cleanup also encountered {len(failures) - 1} additional failure(s)")
    raise primary

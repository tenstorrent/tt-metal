# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill invocation, trace hooks, and result handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

import ttnn
from models.common.llm_runtime.module_input_validation import suspend_module_input_validation
from models.common.llm_runtime.prefill.plan import _PAGE_TABLE_WIDTH_ALIGNMENT, PrefillRequest, _plan_prefill_requests
from models.common.llm_runtime.prefill.sampling_helpers import (
    _TILE_SIZE,
    SamplingPath,
    _formatted_sampling_values,
    _merge_log_probs,
    _select_sample_log_prob,
    _slice_sampling_params,
)
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.sampling import SamplingParams

PrefillVariant = Literal["regular-single", "regular-batched", "chunked"]


@dataclass(frozen=True)
class PrefillProgramSignature:
    """Material values selecting one eager prefill program variant."""

    operation_variant: PrefillVariant
    padded_batch_size: int
    invocation_sequence_length: int
    page_table_width: int
    chunk_page_table_width: int | None
    sampling_path: SamplingPath
    last_token_tile_start: int | None = None

    def key_material(self) -> tuple[tuple[str, str | int | None], ...]:
        return (
            ("operation_variant", self.operation_variant),
            ("padded_batch_size", self.padded_batch_size),
            ("invocation_sequence_length", self.invocation_sequence_length),
            ("page_table_width", self.page_table_width),
            ("chunk_page_table_width", self.chunk_page_table_width),
            ("sampling_path", self.sampling_path),
            ("last_token_tile_start", self.last_token_tile_start),
        )


@dataclass(frozen=True)
class PrefillTraceSignature:
    """Identity of the regular prefill hidden body and persistent schema."""

    padded_batch_size: int
    padded_sequence_length: int
    page_table_width: int

    def key_material(self) -> tuple[tuple[str, str | int | None], ...]:
        return (
            ("padded_batch_size", self.padded_batch_size),
            ("padded_sequence_length", self.padded_sequence_length),
            ("page_table_width", self.page_table_width),
        )


@dataclass(frozen=True)
class PreparedPrefill:
    """A request classified once for eager compilation or traced dispatch."""

    request: PrefillRequest
    sampling_params: SamplingParams | None
    sampling_path: SamplingPath
    program_signatures: tuple[PrefillProgramSignature, ...]
    trace_signature: PrefillTraceSignature | None

    @property
    def trace_eligible(self) -> bool:
        return self.trace_signature is not None


@dataclass(frozen=True)
class PrefillHostInputs:
    tokens: Any
    position_indices: Any
    page_table: Any
    chunk_page_table: Any | None
    chunk_start_idx: Any | None

    def values(self) -> tuple[Any, ...]:
        return (
            self.tokens,
            self.position_indices,
            self.page_table,
            self.chunk_page_table,
            self.chunk_start_idx,
        )


@dataclass(frozen=True)
class PrefillDeviceInputs:
    tokens: Any
    rotary_cos: Any
    rotary_sin: Any
    page_table: Any
    chunk_page_table: Any | None
    position_indices: Any
    chunk_start_idx: Any | None

    def model_values(self) -> tuple[Any, ...]:
        return (
            self.tokens,
            self.rotary_cos,
            self.rotary_sin,
            self.page_table,
            self.chunk_page_table,
            self.position_indices,
            self.chunk_start_idx,
        )

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.model_values()


@dataclass(frozen=True)
class PrefillPositionInputs:
    slice_start: Any
    slice_end: Any
    row_index: Any

    def values(self) -> tuple[Any, ...]:
        return self.slice_start, self.slice_end, self.row_index

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.values()


@dataclass(frozen=True)
class InvocationResult:
    value: Any
    owned: Any


@dataclass(frozen=True)
class PrefillPersistentInputs:
    device_inputs: PrefillDeviceInputs
    position_inputs: PrefillPositionInputs
    kpt: tuple[Any, Any, Any] | None
    sampled_output: Any | None = None
    position_signature: list[int] | None = None
    kpt_signature: list[Any] | None = None

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.device_inputs.model_values(), self.position_inputs.values(), self.kpt, self.sampled_output


@dataclass(frozen=True)
class PrefillCapturePlan:
    """Operation hooks consumed by the trace compiler."""

    signature: PrefillTraceSignature
    prepare_inputs: Callable[[], PrefillPersistentInputs]
    capture: Callable[[PrefillPersistentInputs], Any]
    refresh: Callable[[PrefillPersistentInputs], None]
    refresh_fields: tuple[str, ...] = ("tokens", "page_table", "last_token", "sampling")


class PrefillRuntime:
    """Own the complete prefill operation slice and its transient resources."""

    def __init__(
        self,
        *,
        model: Any,
        mesh_device: Any,
        output_reader: Any,
        page_table_layout: Any,
        max_batch_size: int,
        max_prefill_chunk_size: int,
        cluster_shape: Sequence[int],
        device_sampling_enabled: bool,
        can_enable_trace: Callable[[int, int], bool],
    ) -> None:
        block_size = int(page_table_layout.block_size)
        max_actual_page_table_width = int(page_table_layout.raw_capacity_width)
        canonical_page_table_width = int(page_table_layout.prefill_width)
        if block_size <= 0 or max_batch_size <= 0 or max_prefill_chunk_size <= 0:
            raise ValueError("prefill runtime dimensions must be positive")
        if max_actual_page_table_width <= 0:
            raise ValueError("max_actual_page_table_width must be positive")
        if canonical_page_table_width < max_actual_page_table_width:
            raise ValueError("canonical page-table width cannot be smaller than physical width")
        if canonical_page_table_width % _PAGE_TABLE_WIDTH_ALIGNMENT:
            raise ValueError("canonical page-table width must be 8-entry aligned")
        if len(cluster_shape) != 2:
            raise ValueError("cluster_shape must contain rows and columns")
        if not callable(can_enable_trace):
            raise TypeError("can_enable_trace must be callable")

        self.model = model
        self.mesh_device = mesh_device
        self.output_reader = output_reader
        self.page_table_layout = page_table_layout
        self.block_size = int(block_size)
        self.max_batch_size = int(max_batch_size)
        self.max_prefill_chunk_size = int(max_prefill_chunk_size)
        self.max_actual_page_table_width = int(max_actual_page_table_width)
        self.canonical_page_table_width = int(canonical_page_table_width)
        self.cluster_shape = tuple(int(value) for value in cluster_shape)
        self.device_sampling_enabled = bool(device_sampling_enabled)
        self.can_enable_trace = can_enable_trace
        self._transient_orphans: list[TensorResourceOrphan] = []

    @property
    def transient_orphan_count(self) -> int:
        return len(self._transient_orphans)

    def plan(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        start_pos: torch.Tensor | None = None,
    ) -> tuple[PrefillRequest, ...]:
        return _plan_prefill_requests(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            block_size=self.block_size,
            max_batch_size=self.max_batch_size,
            max_prefill_chunk_size=self.max_prefill_chunk_size,
            max_actual_page_table_width=self.max_actual_page_table_width,
            canonical_page_table_width=self.canonical_page_table_width,
        )

    def prepare(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        start_pos: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> tuple[PreparedPrefill, ...]:
        self._ensure_usable()
        self._validate_sampling_request(sampling_params)
        requests = self.plan(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
        )
        prepared = []
        for request in requests:
            request_sampling = _slice_sampling_params(sampling_params, request.source_rows)
            sampling_path = self._sampling_path(request_sampling, request)
            signatures = tuple(self.program_signatures(request, sampling_path))
            trace_signature = self.trace_signature(request)
            prepared.append(
                PreparedPrefill(
                    request=request,
                    sampling_params=request_sampling,
                    sampling_path=sampling_path,
                    program_signatures=signatures,
                    trace_signature=trace_signature,
                )
            )
        return tuple(prepared)

    def program_signatures(
        self,
        request: PrefillRequest,
        sampling_path: SamplingPath,
    ) -> tuple[PrefillProgramSignature, ...]:
        variant: PrefillVariant
        if request.uses_chunked_prefill:
            variant = "chunked"
        elif request.kind == "batched":
            variant = "regular-batched"
        else:
            variant = "regular-single"
        signatures = []
        for chunk in request.chunks:
            last_token_tile_start = None
            if self._uses_q128_tiled_sample(request, sampling_path):
                relative_last = request.last_token_indices[0] - request.cached_tokens[0]
                last_token_tile_start = (relative_last // _TILE_SIZE) * _TILE_SIZE
            signatures.append(
                PrefillProgramSignature(
                    operation_variant=variant,
                    padded_batch_size=request.padded_batch_size,
                    invocation_sequence_length=chunk.chunk_size,
                    page_table_width=request.page_table_width,
                    chunk_page_table_width=(
                        int(chunk.chunk_page_table.shape[-1]) if chunk.chunk_page_table is not None else None
                    ),
                    sampling_path=sampling_path,
                    last_token_tile_start=last_token_tile_start,
                )
            )
        return tuple(dict.fromkeys(signatures))

    def trace_signature(self, request: PrefillRequest) -> PrefillTraceSignature | None:
        if request.uses_chunked_prefill:
            return None
        if any(request.cached_tokens):
            return None
        if not self.can_enable_trace(request.padded_sequence_length, 0):
            return None
        return PrefillTraceSignature(
            padded_batch_size=request.padded_batch_size,
            padded_sequence_length=request.padded_sequence_length,
            page_table_width=request.page_table_width,
        )

    def invoke(self, prepared: PreparedPrefill) -> InvocationResult:
        """Run a prepared request eagerly without replanning or reclassification."""

        self._ensure_usable()
        request = prepared.request
        if request.uses_chunked_prefill:
            return self._run_chunked_prefill(prepared)
        return self._run_regular_prefill(prepared)

    def capture_plan(self, prepared: PreparedPrefill) -> PrefillCapturePlan:
        if prepared.trace_signature is None:
            raise ValueError("cached and multi-chunk prefill requests are trace-ineligible")

        def prepare_inputs() -> PrefillPersistentInputs:
            return self._prepare_persistent_inputs(prepared)

        def capture(persistent: PrefillPersistentInputs) -> Any:
            with suspend_module_input_validation():
                return self._run_hidden_body(prepared.request, persistent.device_inputs)

        def refresh(persistent: PrefillPersistentInputs) -> None:
            self.refresh_trace(prepared, persistent)

        return PrefillCapturePlan(
            signature=prepared.trace_signature,
            prepare_inputs=prepare_inputs,
            capture=capture,
            refresh=refresh,
        )

    def refresh_trace(self, prepared: PreparedPrefill, persistent: PrefillPersistentInputs) -> None:
        """Refresh borrowed persistent inputs for one replay."""

        request = prepared.request
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        # Trace-eligible prefill fixes rotary positions and has no chunk inputs.
        # Rebuilding those capture-owned host tensors on every replay adds TTFT
        # without refreshing any device input; only tokens and page table vary.
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        host_tokens = ttnn.from_torch(
            request.tokens.reshape(1, 1, 1, -1),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        host_page_table = ttnn.from_torch(
            request.page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_tokens, persistent.device_inputs.tokens)
        ttnn.copy_host_to_device_tensor(host_page_table, persistent.device_inputs.page_table)
        if not self._uses_static_q128_topk(request, prepared.sampling_path):
            position_value = relative_last
            position_signature = persistent.position_signature
            if position_signature is None or position_signature[0] != position_value:
                position_inputs = self._prepare_position_inputs_host(relative_last, request.padded_sequence_length)
                _copy_host_to_device(position_inputs.values(), persistent.position_inputs.values())
                if position_signature is not None:
                    position_signature[0] = position_value
        if prepared.sampling_path == "topk":
            sampling_batch_size = self._sampling_parameter_batch_size(prepared)
            kpt_value = self._kpt_signature(prepared.sampling_params, sampling_batch_size)
            kpt_signature = persistent.kpt_signature
            if kpt_signature is None or kpt_signature[0] != kpt_value:
                self._refresh_kpt(
                    persistent.kpt,
                    prepared.sampling_params,
                    sampling_batch_size,
                    force_topk=True,
                )
                if kpt_signature is not None:
                    kpt_signature[0] = kpt_value

    def finish_trace(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        persistent: PrefillPersistentInputs,
    ) -> InvocationResult:
        output = self._finish_regular_prefill(
            prepared,
            hidden,
            persistent.kpt if prepared.sampling_path == "topk" else None,
            persistent.position_inputs,
            sampled_output=persistent.sampled_output,
        )
        owned = () if persistent.sampled_output is not None else (output,)
        return InvocationResult(value=output, owned=owned)

    def assemble(
        self,
        prepared_results: Sequence[tuple[PreparedPrefill, InvocationResult]],
        *,
        batch_size: int,
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Read phase outputs, restore source-row order, and release transients."""

        sampled = sampling_params is not None
        if prepared_results and sampled != (prepared_results[0][0].sampling_params is not None):
            raise ValueError("prefill result sampling path disagrees with the public request")
        output_logits = torch.zeros(batch_size, 1, int(self.model.vocab_size))
        output_tokens = torch.zeros(batch_size, dtype=torch.int64)
        row_log_probs: list[tuple[tuple[int, ...], Any]] = []

        for prepared, result in prepared_results:
            request = prepared.request
            try:
                host_output = self.output_reader.read_synchronized(result.value)
                host_primary, host_log_probs = _split_output(host_output)
                if sampled:
                    output_rows = (
                        _TILE_SIZE
                        if self._uses_static_q128_topk(request, prepared.sampling_path)
                        else self._sampling_batch_size(request)
                    )
                    sampled_tokens = _process_output_tokens(
                        host_primary,
                        output_rows,
                        self.cluster_shape,
                    )
                    for source_row, slot in zip(request.source_rows, request.slots):
                        if request.kind == "batched":
                            token_index = slot
                        elif self._uses_static_q128_topk(request, prepared.sampling_path):
                            token_index = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                        else:
                            token_index = 0
                        output_tokens[source_row] = sampled_tokens.reshape(-1)[token_index].to(torch.int64)
                    if host_log_probs is not None:
                        if self._uses_static_q128_topk(request, prepared.sampling_path):
                            host_log_probs = _select_sample_log_prob(host_log_probs, token_index)
                        row_log_probs.append((request.source_rows, host_log_probs))
                elif request.kind == "batched":
                    for source_row, slot in zip(request.source_rows, request.slots):
                        output_logits[source_row] = _process_output_prefill(
                            host_primary,
                            slot,
                            int(self.model.vocab_size),
                            self.cluster_shape,
                        )
                else:
                    relative_last = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                    output_logits[request.source_rows[0]] = _process_output_prefill(
                        host_primary,
                        relative_last,
                        int(self.model.vocab_size),
                        self.cluster_shape,
                    )
            except BaseException as primary:
                failures = self._release_or_retain_transient(result.owned)
                attach_cleanup_failures(primary, failures)
                raise
            failures = self._release_or_retain_transient(result.owned)
            if failures:
                raise_cleanup_failures(failures)

        if sampled:
            return output_tokens, _merge_log_probs(row_log_probs, batch_size)
        return output_logits

    def cleanup(self) -> None:
        failures = self._release_transient_orphans()
        if failures:
            raise_cleanup_failures(failures)

    def _run_regular_prefill(self, prepared: PreparedPrefill) -> InvocationResult:
        request = prepared.request
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        host_inputs = self._prepare_inputs_host(
            request.tokens,
            request.page_table,
            last_token_idx=max(request.last_token_indices),
        )
        device_inputs = None
        position_inputs = None
        kpt = None
        hidden = None
        output = None
        sampled_output = None
        try:
            device_inputs, position_inputs, kpt = self._stage_inputs_and_kpt(
                host_inputs,
                prepared.sampling_params,
                self._sampling_parameter_batch_size(prepared),
                relative_last=relative_last,
                sequence_length=request.padded_sequence_length,
                force_topk=prepared.sampling_path == "topk",
            )
            hidden = self._run_hidden_body(request, device_inputs)
            if prepared.sampling_path == "topk":
                sampled_output = self._make_sampling_output(self._sampling_output_rows(prepared))
            output = self._finish_regular_prefill(
                prepared,
                hidden,
                kpt,
                position_inputs,
                sampled_output=sampled_output,
            )
        except BaseException as primary:
            failures = self._release_or_retain_transient(
                (output, sampled_output, hidden, device_inputs, position_inputs, kpt)
            )
            attach_cleanup_failures(primary, failures)
            raise
        extra_sampled_output = () if sampled_output is output else (sampled_output,)
        return InvocationResult(
            value=output,
            owned=(output, *extra_sampled_output, hidden, device_inputs, position_inputs, kpt),
        )

    def _run_chunked_prefill(self, prepared: PreparedPrefill) -> InvocationResult:
        """Execute already-planned cached or multi-chunk prefill invocations."""

        request = prepared.request
        owned_inputs: list[Any] = []
        kpt = self._make_device_kpt(
            prepared.sampling_params,
            self._sampling_batch_size(request),
            force_topk=prepared.sampling_path == "topk",
        )
        output = None
        try:
            last_chunk = request.chunks[-1]
            relative_last = (request.last_token_indices[0] - last_chunk.chunk_start_idx) % last_chunk.chunk_size
            for chunk in request.chunks:
                chunk_tokens = request.tokens[:, chunk.token_slice]
                host_inputs = self._prepare_inputs_host(
                    chunk_tokens,
                    request.page_table,
                    start_pos=chunk.chunk_start_idx,
                    chunk_page_table=chunk.chunk_page_table,
                    chunk_start_idx=chunk.chunk_start_idx,
                    last_token_idx=request.last_token_indices[0],
                )
                device_inputs = self._stage_device_inputs(host_inputs)
                owned_inputs.append(device_inputs)
                position_inputs = _copy_host_to_device(
                    self._prepare_position_inputs_host(relative_last, chunk.chunk_size).values(),
                    mesh_device=self.mesh_device,
                )
                position_inputs = PrefillPositionInputs(*position_inputs)
                owned_inputs.append(position_inputs)
                output = self.model.prefill_forward(
                    self.model.embed_prefill(device_inputs.tokens),
                    [device_inputs.rotary_cos, device_inputs.rotary_sin],
                    user_id=0,
                    page_table=device_inputs.page_table,
                    chunk_page_table=device_inputs.chunk_page_table,
                    chunk_start_idx=chunk.chunk_start_idx,
                    get_last_token=-1,
                    chunk_start_idx_tensor=device_inputs.chunk_start_idx,
                    last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
                    last_token_index=(position_inputs.row_index if prepared.sampling_params is not None else None),
                )
                if chunk.contains_last_token:
                    break
                failures = self._release_or_retain_transient(output)
                output = None
                if failures:
                    raise_cleanup_failures(failures)

            if prepared.sampling_params is not None:
                selected = _pad_prefill_logits(output, self.model.sampling)
                output = self._sample_device(selected, kpt)
            else:
                output = ttnn.untilize(output, use_multicore=True)
        except BaseException as primary:
            failures = self._release_or_retain_transient((output, owned_inputs, kpt))
            attach_cleanup_failures(primary, failures)
            raise
        return InvocationResult(value=output, owned=(output, owned_inputs, kpt))

    def _prepare_persistent_inputs(self, prepared: PreparedPrefill) -> PrefillPersistentInputs:
        request = prepared.request
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        host_inputs = self._prepare_inputs_host(
            request.tokens,
            request.page_table,
            last_token_idx=max(request.last_token_indices),
        )
        device_inputs = None
        position_inputs = None
        kpt = None
        sampled_output = None
        try:
            sampling_batch_size = self._sampling_parameter_batch_size(prepared)
            device_inputs, position_inputs, kpt = self._stage_inputs_and_kpt(
                host_inputs,
                prepared.sampling_params,
                sampling_batch_size,
                relative_last=relative_last,
                sequence_length=request.padded_sequence_length,
                force_topk=prepared.sampling_path == "topk",
            )
            if prepared.sampling_params is not None:
                sampled_output = self._make_sampling_output(self._sampling_output_rows(prepared))
        except BaseException as primary:
            failures = self._release_or_retain_transient((device_inputs, position_inputs, kpt, sampled_output))
            attach_cleanup_failures(primary, failures)
            raise
        return PrefillPersistentInputs(
            device_inputs=device_inputs,
            position_inputs=position_inputs,
            kpt=kpt,
            sampled_output=sampled_output,
            position_signature=[relative_last],
            kpt_signature=[self._kpt_signature(prepared.sampling_params, sampling_batch_size)],
        )

    def _run_hidden_body(self, request: PrefillRequest, device_inputs: PrefillDeviceInputs) -> Any:
        return self.model.prefill_forward(
            self.model.embed_prefill(device_inputs.tokens),
            [device_inputs.rotary_cos, device_inputs.rotary_sin],
            user_id=list(range(request.padded_batch_size)) if request.kind == "batched" else 0,
            page_table=device_inputs.page_table,
            chunk_page_table=device_inputs.chunk_page_table,
            get_last_token=-1,
            batch_size=request.padded_batch_size,
            chunk_start_idx_tensor=device_inputs.chunk_start_idx,
        )

    def _finish_regular_prefill(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        kpt: tuple[Any, Any, Any] | None,
        position_inputs: PrefillPositionInputs,
        *,
        sampled_output: Any | None = None,
    ) -> Any:
        request = prepared.request
        relative_last = [last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)]
        if request.kind == "batched":
            padded_last = list(relative_last) + [0] * (request.padded_batch_size - len(relative_last))
            logits = self.model.post_process_batched_prefill_output(
                hidden,
                padded_last,
                request.padded_batch_size,
                request.padded_sequence_length,
                last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
                last_token_index=position_inputs.row_index,
            )
        elif self._uses_static_q128_topk(request, prepared.sampling_path):
            logits = self.model.post_process_prefill_output(hidden, relative_last[0])
        else:
            logits = self.model.post_process_prefill_output(
                hidden,
                relative_last[0],
                last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
                last_token_index=(position_inputs.row_index if prepared.sampling_params is not None else None),
            )
        if prepared.sampling_params is not None:
            logits = _pad_prefill_logits(logits, self.model.sampling)
            return self._sample_device(logits, kpt, sampled_output)
        return ttnn.untilize(logits, use_multicore=True)

    def _prepare_inputs_host(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        start_pos: int = 0,
        chunk_page_table: torch.Tensor | None = None,
        chunk_start_idx: int | None = None,
        last_token_idx: int | None = None,
    ) -> PrefillHostInputs:
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
        if start_pos < 0:
            raise ValueError("prefill start position must be nonnegative")
        if last_token_idx is not None and int(last_token_idx) + 1 > matrix_length:
            raise ValueError(f"Sequence length {int(last_token_idx) + 1} exceeds rotary capacity {matrix_length}")
        position_indices = torch.arange(start_pos, start_pos + sequence_length, dtype=torch.long).clamp(
            max=matrix_length - 1
        )
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
        chunk_tt = (
            ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            if chunk_page_table is not None
            else None
        )
        chunk_start_tt = (
            ttnn.from_torch(
                torch.tensor([int(chunk_start_idx)], dtype=torch.int32),
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            if chunk_start_idx is not None
            else None
        )
        return PrefillHostInputs(tokens_tt, position_indices_tt, page_table_tt, chunk_tt, chunk_start_tt)

    def _prepare_position_inputs_host(self, relative_last: int, sequence_length: int) -> PrefillPositionInputs:
        relative_last = int(relative_last)
        sequence_length = int(sequence_length)
        if relative_last < 0 or relative_last >= sequence_length:
            raise ValueError("prefill last-token position must fall within the padded sequence")
        block_start = (relative_last // _TILE_SIZE) * _TILE_SIZE
        hidden_width = int(self.model.config.dim)
        bounds = ((0, 0, block_start, 0), (1, 1, block_start + _TILE_SIZE, hidden_width))
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
            torch.tensor([[relative_last % _TILE_SIZE]], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        return PrefillPositionInputs(slice_bounds[0], slice_bounds[1], row_index)

    def _stage_device_inputs(self, host_inputs: PrefillHostInputs) -> PrefillDeviceInputs:
        raw_inputs = None
        rot_mats = None
        try:
            raw_inputs = _copy_host_to_device(host_inputs.values(), mesh_device=self.mesh_device)
            prepare_rot_mats = getattr(self.model, "prepare_prefill_rot_mats", None)
            if not callable(prepare_rot_mats):
                raise TypeError("model must provide prepare_prefill_rot_mats()")
            rot_mats = tuple(prepare_rot_mats(raw_inputs[1]))
            if len(rot_mats) != 2:
                raise ValueError("prepare_prefill_rot_mats() must return cosine and sine tensors")
        except BaseException as primary:
            failures = self._release_or_retain_transient((rot_mats, raw_inputs))
            attach_cleanup_failures(primary, failures)
            raise
        return PrefillDeviceInputs(
            tokens=raw_inputs[0],
            rotary_cos=rot_mats[0],
            rotary_sin=rot_mats[1],
            page_table=raw_inputs[2],
            chunk_page_table=raw_inputs[3],
            position_indices=raw_inputs[1],
            chunk_start_idx=raw_inputs[4],
        )

    def _stage_inputs_and_kpt(
        self,
        host_inputs: PrefillHostInputs,
        sampling_params: SamplingParams | None,
        batch_size: int,
        *,
        relative_last: int,
        sequence_length: int,
        force_topk: bool,
    ) -> tuple[PrefillDeviceInputs, PrefillPositionInputs, tuple[Any, Any, Any] | None]:
        device_inputs = None
        position_inputs = None
        kpt = None
        try:
            device_inputs = self._stage_device_inputs(host_inputs)
            position_values = _copy_host_to_device(
                self._prepare_position_inputs_host(relative_last, sequence_length).values(),
                mesh_device=self.mesh_device,
            )
            position_inputs = PrefillPositionInputs(*position_values)
            kpt = self._make_device_kpt(sampling_params, batch_size, force_topk)
        except BaseException as primary:
            failures = self._release_or_retain_transient((device_inputs, position_inputs, kpt))
            attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, position_inputs, kpt

    def _sampling_batch_size(self, request: PrefillRequest) -> int:
        if not self.device_sampling_enabled:
            return request.padded_batch_size
        return int(self.model.sampling.config.max_batch_size)

    def _sampling_output_rows(self, prepared: PreparedPrefill) -> int:
        if self._uses_static_q128_topk(prepared.request, prepared.sampling_path):
            return _TILE_SIZE
        return self._sampling_batch_size(prepared.request)

    def _uses_static_q128_topk(self, request: PrefillRequest, sampling_path: SamplingPath) -> bool:
        return (
            int(self.model.sampling.config.max_batch_size) >= _TILE_SIZE
            and sampling_path == "topk"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        )

    def _uses_q128_tiled_sample(self, request: PrefillRequest, sampling_path: SamplingPath) -> bool:
        return self._uses_static_q128_topk(request, sampling_path) or (
            sampling_path == "argmax"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        )

    def _sampling_parameter_batch_size(self, prepared: PreparedPrefill) -> int:
        # TT sampling validates K/P/T against the physical logits row count.
        # The static Q128 path retains one complete tile and selects the exact
        # logical row on the host, so its sampling tensors must also span that
        # tile even for a single logical user.
        return self._sampling_output_rows(prepared)

    def _sampling_path(
        self,
        sampling_params: SamplingParams | None,
        request: PrefillRequest,
    ) -> SamplingPath:
        if sampling_params is None:
            return "logits"
        if request.kind == "single" and bool(self.model.sampling.config.allow_force_argmax):
            values = _formatted_sampling_values(sampling_params, self._sampling_batch_size(request))
            if values[3]:
                return "argmax"
        return "topk"

    def _make_device_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
        host = self._make_host_kpt(sampling_params, batch_size, force_topk)
        if host is None:
            return None
        return tuple(_copy_host_to_device(host, mesh_device=self.mesh_device))

    def _make_host_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
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

    def _kpt_signature(self, sampling_params: SamplingParams | None, batch_size: int) -> Any:
        if sampling_params is None:
            return None
        k, p, temperature, _ = _formatted_sampling_values(sampling_params, batch_size)
        return k, p, temperature

    def _refresh_kpt(
        self,
        device_kpt: tuple[Any, Any, Any] | None,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> None:
        host_kpt = self._make_host_kpt(sampling_params, batch_size, force_topk)
        if (host_kpt is None) != (device_kpt is None):
            raise RuntimeError("sampling parameters changed the compiled sampling path")
        if host_kpt is not None:
            _copy_host_to_device(host_kpt, device_kpt)

    def _make_sampling_output(self, batch_size: int) -> Any:
        return ttnn.from_torch(
            torch.zeros((1, 1, 1, int(batch_size)), dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _sample_device(
        self,
        logits: Any,
        kpt: tuple[Any, Any, Any] | None,
        sampled_output: Any | None = None,
    ) -> Any:
        if kpt is None:
            return self.model.sampling.decode_forward(logits, tt_out_tok=sampled_output)
        return self.model.sampling.decode_forward(
            logits,
            k=kpt[0],
            p=kpt[1],
            temp=kpt[2],
            tt_out_tok=sampled_output,
        )

    def _validate_sampling_request(self, sampling_params: SamplingParams | None) -> None:
        if sampling_params is not None and not self.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")

    def _release_or_retain_transient(self, values: Any) -> list[BaseException]:
        orphan = TensorResourceOrphan(values)
        failures = best_effort_deallocate_owned_tensors(orphan.values, orphan.deallocated_tensor_ids)
        if failures:
            self._transient_orphans.append(orphan)
        return failures

    def _release_transient_orphans(self) -> list[BaseException]:
        return release_orphans(self._transient_orphans)

    def _ensure_usable(self) -> None:
        if self._transient_orphans:
            raise RuntimeError("PrefillRuntime has unreleased transient resources; cleanup is required")


def _copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    if device_tensors is None:
        if mesh_device is None:
            raise ValueError("mesh_device is required for device allocation")
        allocated = []
        try:
            for host_tensor in host_tensors:
                allocated.append(ttnn.to_device(host_tensor, device=mesh_device) if host_tensor is not None else None)
        except BaseException as primary:
            failures = best_effort_deallocate_owned_tensors(allocated)
            attach_cleanup_failures(primary, failures)
            raise
        return allocated
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        if host_tensor is None:
            if device_tensor is not None:
                raise ValueError("host/device optional tensor structure changed")
            continue
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
    return device_tensors


def _pad_prefill_logits(logits, sampler):
    target_batch = int(sampler.config.max_batch_size)
    current_batch = int(logits.shape[2])
    if current_batch >= target_batch:
        return logits
    return ttnn.pad(logits, [(0, 0), (0, 0), (0, target_batch - current_batch), (0, 0)], value=0.0)


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


def _process_output_tokens(value, batch_size, cluster_shape):
    if isinstance(value, ttnn.Tensor):
        replicas = ttnn.get_device_tensors(value)
        if not replicas:
            raise ValueError("sampled prefill output has no device tensors")
        # Sampling outputs are replicated. The prior mesh concatenation also
        # selected only replica zero below, but converted every unused replica
        # to torch first.
        output = ttnn.to_torch(replicas[0])
    else:
        output = value
    if output.ndim >= 4:
        if int(output.shape[2]) >= batch_size:
            output = output[0, 0, :batch_size, 0]
        elif int(output.shape[3]) >= batch_size:
            output = output[0, 0, 0, :batch_size]
    return output.reshape(-1)[:batch_size].to(torch.int64)


def _split_output(value):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("runtime output tuple must contain (output, log_probs)")
        return value
    return value, None

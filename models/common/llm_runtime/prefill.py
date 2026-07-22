# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill request planning, invocation, trace hooks, and result handling.

The values in this module deliberately describe one prepared semantic request.
Program compilation, trace selection, warmup policy, and paged-KV ownership
belong to their respective runtime components.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

import ttnn
from models.common.llm_runtime.module_input_validation import suspend_module_input_validation
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.sampling import SamplingParams, format_sampling_params

_SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
_MAX_BATCHED_PREFILL_TOKENS = 128 * 1024
_PAGE_TABLE_WIDTH_ALIGNMENT = 8
_TILE_SIZE = 32

PrefillKind = Literal["single", "batched"]
PrefillVariant = Literal["regular-single", "regular-batched", "chunked"]
SamplingPath = Literal["logits", "argmax", "topk"]


@dataclass(frozen=True)
class PrefillChunk:
    """One planned invocation over a slice of a request's uncached tokens."""

    token_slice: slice
    chunk_start_idx: int
    chunk_size: int
    chunk_page_table: torch.Tensor | None
    contains_last_token: bool

    def __post_init__(self) -> None:
        if self.token_slice.step not in (None, 1):
            raise ValueError("prefill chunk slices must be contiguous")
        if self.token_slice.start is None or self.token_slice.stop is None:
            raise ValueError("prefill chunk slices must have explicit bounds")
        if self.token_slice.start < 0 or self.token_slice.stop <= self.token_slice.start:
            raise ValueError("prefill chunk slices must be non-empty and nonnegative")
        if self.token_slice.stop - self.token_slice.start != self.chunk_size:
            raise ValueError("prefill chunk slice and chunk_size disagree")
        if self.chunk_start_idx < 0 or self.chunk_size <= 0:
            raise ValueError("prefill chunk positions must be nonnegative and non-empty")
        if self.chunk_page_table is not None:
            if not isinstance(self.chunk_page_table, torch.Tensor) or self.chunk_page_table.ndim != 2:
                raise ValueError("chunk_page_table must be a rank-2 torch.Tensor")


@dataclass(frozen=True)
class PrefillRequest:
    """One immutable planned prefill unit with all chunk decisions retained."""

    kind: PrefillKind
    source_rows: tuple[int, ...]
    slots: tuple[int, ...]
    tokens: torch.Tensor
    page_table: torch.Tensor
    prompt_lengths: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    last_token_indices: tuple[int, ...]
    padded_sequence_length: int
    padded_batch_size: int
    chunks: tuple[PrefillChunk, ...]
    uses_chunked_prefill: bool

    def __post_init__(self) -> None:
        row_count = len(self.source_rows)
        if self.kind not in ("single", "batched"):
            raise ValueError(f"unsupported prefill request kind {self.kind!r}")
        if row_count == 0 or not (
            len(self.slots)
            == len(self.prompt_lengths)
            == len(self.cached_tokens)
            == len(self.last_token_indices)
            == row_count
        ):
            raise ValueError("prefill request row metadata must be non-empty and aligned")
        if self.kind == "single" and row_count != 1:
            raise ValueError("single prefill requests must describe exactly one source row")
        if not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 2:
            raise ValueError("planned prefill tokens must be a rank-2 torch.Tensor")
        if not isinstance(self.page_table, torch.Tensor) or self.page_table.ndim != 2:
            raise ValueError("planned prefill page_table must be a rank-2 torch.Tensor")
        if int(self.tokens.shape[0]) != self.padded_batch_size:
            raise ValueError("planned token batch does not match padded_batch_size")
        if int(self.tokens.shape[1]) != self.padded_sequence_length:
            raise ValueError("planned token width does not match padded_sequence_length")
        if int(self.page_table.shape[0]) != self.padded_batch_size:
            raise ValueError("planned page-table batch does not match padded_batch_size")
        if not self.chunks:
            raise ValueError("a prefill request must contain at least one planned chunk")
        if sum(chunk.contains_last_token for chunk in self.chunks) != 1:
            raise ValueError("exactly one planned chunk must contain the actual last token")
        if not self.chunks[-1].contains_last_token:
            raise ValueError("planning must stop at the chunk containing the actual last token")
        if self.uses_chunked_prefill != any(chunk.chunk_page_table is not None for chunk in self.chunks):
            raise ValueError("chunked-prefill classification disagrees with planned chunks")

    @property
    def page_table_width(self) -> int:
        return int(self.page_table.shape[-1])


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


def _plan_prefill_requests(
    *,
    tokens: torch.Tensor,
    page_table: torch.Tensor,
    prompt_lens: torch.Tensor | None,
    empty_slots: Sequence[int] | None,
    start_pos: torch.Tensor | None,
    block_size: int,
    max_batch_size: int,
    max_prefill_chunk_size: int,
    max_actual_page_table_width: int | None = None,
    canonical_page_table_width: int | None = None,
) -> tuple[PrefillRequest, ...]:
    """Plan prefix-caching and chunked-prefill semantics exactly once."""

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
    for row, (length, num_cached_tokens) in enumerate(zip(lengths, cached)):
        if num_cached_tokens < 0 or length < 0 or num_cached_tokens > length or length > token_width:
            raise ValueError(f"invalid prompt/cached-token lengths for prefill row {row}")
        if num_cached_tokens % block_size:
            raise ValueError(f"cached prefill start for row {row} must be block aligned")
    uncached_lengths = [length - num_cached_tokens for length, num_cached_tokens in zip(lengths, cached)]
    padded_lengths = [_padded_prefill_length(length) if length > 0 else 0 for length in uncached_lengths]

    padded_batch = None
    if len(set(padded_lengths)) == 1:
        padded_batch = _batched_prefill_size(
            batch_size,
            padded_lengths,
            cached,
            max_batch_size,
            max_prefill_chunk_size,
        )
    if padded_batch is not None and slots == list(range(batch_size)):
        sequence_length = padded_lengths[0]
        request_tokens = torch.zeros((padded_batch, sequence_length), dtype=torch.long, device=tokens.device)
        last_indices = []
        for source_row, slot in enumerate(slots):
            length = lengths[source_row]
            request_tokens[slot, :length] = tokens[source_row, :length]
            last_indices.append(length - 1)
        actual_width = max(_num_blocks(length, block_size) for length in lengths)
        page_width = canonical_page_table_width or _num_blocks(sequence_length, block_size)
        _validate_page_table_width(
            actual_width,
            page_table,
            max_actual_page_table_width,
            "batched prefill",
        )
        request_page_table = torch.full(
            (padded_batch, page_width),
            -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        for source_row, slot in enumerate(slots):
            row_width = _num_blocks(lengths[source_row], block_size)
            request_page_table[slot, :row_width] = page_table[source_row, :row_width].to(torch.int32)
        chunk = PrefillChunk(
            token_slice=slice(0, sequence_length),
            chunk_start_idx=0,
            chunk_size=sequence_length,
            chunk_page_table=None,
            contains_last_token=True,
        )
        return (
            PrefillRequest(
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
                chunks=(chunk,),
                uses_chunked_prefill=False,
            ),
        )

    requests = []
    for source_row, slot in enumerate(slots):
        uncached_length = uncached_lengths[source_row]
        # Gate 1 remains intentionally behavior-preserving until its public
        # cache-hit output contract is decided.
        if uncached_length <= 0:
            continue
        sequence_length = padded_lengths[source_row]
        request_tokens = torch.zeros((1, sequence_length), dtype=torch.long, device=tokens.device)
        request_tokens[0, :uncached_length] = tokens[
            source_row,
            cached[source_row] : lengths[source_row],
        ]
        actual_width = _num_blocks(lengths[source_row], block_size)
        page_width = canonical_page_table_width or _num_blocks(sequence_length + cached[source_row], block_size)
        _validate_page_table_width(
            actual_width,
            page_table,
            max_actual_page_table_width,
            f"prefill row {source_row}",
        )
        if cached[source_row] + sequence_length > page_width * block_size:
            raise ValueError(f"padded prefill row {source_row} exceeds the canonical page-table capacity")
        uses_chunked_prefill = sequence_length > max_prefill_chunk_size or cached[source_row] > 0
        request_page_table = torch.full(
            (1, page_width),
            0 if uses_chunked_prefill else -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        request_page_table[0, :actual_width] = page_table[source_row, :actual_width].to(torch.int32)
        chunks = _plan_chunks(
            padded_sequence_length=sequence_length,
            actual_uncached_length=uncached_length,
            num_cached_tokens=cached[source_row],
            prompt_length=lengths[source_row],
            page_table=request_page_table,
            block_size=block_size,
            max_prefill_chunk_size=max_prefill_chunk_size,
            uses_chunked_prefill=uses_chunked_prefill,
        )
        requests.append(
            PrefillRequest(
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
                chunks=chunks,
                uses_chunked_prefill=uses_chunked_prefill,
            )
        )
    return tuple(requests)


def _plan_chunks(
    *,
    padded_sequence_length: int,
    actual_uncached_length: int,
    num_cached_tokens: int,
    prompt_length: int,
    page_table: torch.Tensor,
    block_size: int,
    max_prefill_chunk_size: int,
    uses_chunked_prefill: bool,
) -> tuple[PrefillChunk, ...]:
    if not uses_chunked_prefill:
        return (
            PrefillChunk(
                token_slice=slice(0, padded_sequence_length),
                chunk_start_idx=0,
                chunk_size=padded_sequence_length,
                chunk_page_table=None,
                contains_last_token=True,
            ),
        )

    chunk_size = (
        _max_prefill_chunk_size(padded_sequence_length, max_prefill_chunk_size)
        if padded_sequence_length > max_prefill_chunk_size
        else padded_sequence_length
    )
    relative_last = actual_uncached_length - 1
    chunks = []
    for relative_start in range(0, padded_sequence_length, chunk_size):
        absolute_start = num_cached_tokens + relative_start
        chunk_start_block = absolute_start // block_size
        chunk_width = _num_blocks(chunk_size, block_size)
        mapped_blocks = min(
            chunk_width,
            max(0, _num_blocks(prompt_length, block_size) - chunk_start_block),
        )
        chunk_page_table = torch.full(
            (int(page_table.shape[0]), chunk_width),
            -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        if mapped_blocks:
            chunk_page_table[:, :mapped_blocks] = page_table[:, chunk_start_block : chunk_start_block + mapped_blocks]
        contains_last_token = relative_start <= relative_last < relative_start + chunk_size
        chunks.append(
            PrefillChunk(
                token_slice=slice(relative_start, relative_start + chunk_size),
                chunk_start_idx=absolute_start,
                chunk_size=chunk_size,
                chunk_page_table=chunk_page_table,
                contains_last_token=contains_last_token,
            )
        )
        if contains_last_token:
            break
    return tuple(chunks)


def _validate_page_table_width(
    actual_width: int,
    page_table: torch.Tensor,
    max_actual_page_table_width: int | None,
    label: str,
) -> None:
    if max_actual_page_table_width is not None and actual_width > max_actual_page_table_width:
        raise ValueError(f"{label} exceeds the configured paged-KV capacity")
    if int(page_table.shape[-1]) < actual_width:
        raise ValueError(f"page table is too narrow for {label}")


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
            return [value[0] for _ in source_rows] if len(value) == 1 else [value[row] for row in source_rows]
        if isinstance(value, tuple):
            return tuple(value[0] for _ in source_rows) if len(value) == 1 else tuple(value[row] for row in source_rows)
        return value

    updates = {
        field.name: slice_value(getattr(sampling_params, field.name)) for field in dataclasses.fields(sampling_params)
    }
    return dataclasses.replace(sampling_params, **updates)


def _formatted_sampling_values(sampling_params, batch_size):
    sampling_params = _tensor_sampling_fields_to_python(sampling_params)
    formatted_size = _round_up(int(batch_size), _TILE_SIZE)
    formatted = format_sampling_params(sampling_params, formatted_size)
    k = tuple(int(value) for value in formatted.top_k[:batch_size])
    p = tuple(float(value) for value in formatted.top_p[:batch_size])
    temperature = tuple(float(value) for value in formatted.temperature[:batch_size])
    greedy = (
        all(value == 1 for value in k) and all(value == 0 for value in p) and all(value == 1 for value in temperature)
    )
    return k, p, temperature, greedy


def _tensor_sampling_fields_to_python(sampling_params):
    updates = {}
    for field in dataclasses.fields(sampling_params):
        value = getattr(sampling_params, field.name)
        if isinstance(value, torch.Tensor):
            updates[field.name] = value.item() if value.ndim == 0 else value.tolist()
    return dataclasses.replace(sampling_params, **updates) if updates else sampling_params


def _padded_prefill_length(sequence_length: int) -> int:
    if sequence_length <= 128:
        return 128
    if sequence_length <= 1024:
        return 1024
    return 1 << (sequence_length - 1).bit_length()


def _batched_prefill_size(batch_size, padded_lengths, cached_tokens, max_batch_size, max_prefill_chunk_size):
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


def _max_prefill_chunk_size(sequence_length: int, maximum: int) -> int:
    minimum_chunk = 2048
    if sequence_length <= 0 or maximum <= 0:
        raise ValueError("prefill chunk lengths must be positive")
    if sequence_length % minimum_chunk or maximum % minimum_chunk:
        raise ValueError("prefill chunk lengths must be multiples of 2048")
    for chunk_size in range(min(sequence_length, maximum), 0, -minimum_chunk):
        if sequence_length % chunk_size == 0:
            return chunk_size
    raise ValueError("no valid prefill chunk size")


def _num_blocks(sequence_length: int, block_size: int) -> int:
    return math.ceil(int(sequence_length) / int(block_size))


def _round_up(value: int, alignment: int) -> int:
    return math.ceil(int(value) / int(alignment)) * int(alignment)


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


def _select_sample_log_prob(value, row):
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[int(row)]
    if isinstance(value, ttnn.Tensor):
        first_replica = ttnn.get_device_tensors(value)[0]
        return ttnn.to_torch(first_replica).reshape(-1)[int(row)]
    return value


def _split_output(value):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("runtime output tuple must contain (output, log_probs)")
        return value
    return value, None


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


__all__ = [
    "InvocationResult",
    "PrefillCapturePlan",
    "PrefillChunk",
    "PrefillDeviceInputs",
    "PrefillHostInputs",
    "PrefillPersistentInputs",
    "PrefillPositionInputs",
    "PrefillProgramSignature",
    "PrefillRequest",
    "PrefillRuntime",
    "PrefillTraceSignature",
    "PreparedPrefill",
]

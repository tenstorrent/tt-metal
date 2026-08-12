# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill invocation, trace hooks, and result handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import torch

import ttnn
from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.inputs import (
    PrefillDeviceInputs,
    PrefillInputStager,
    PrefillPositionInputs,
    allocate_device_tensors,
    copy_into_device_tensors,
)
from models.common.llm_runtime.prefill.plan import (
    PrefillChunk,
    PrefillRequest,
    _max_prefill_chunk_size,
    _padded_prefill_length,
    _plan_prefill_requests,
)
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
    """Material values selecting one eager prefill program variant.

    Active row count is deliberately absent: regular paged fill uses ``-1``
    skip rows, so all active counts with the same padded geometry execute the
    same compiled program. Sampling remains material because it changes the
    postprocessing program family.
    """

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
    """Identity of one static prefill body and persistent input geometry."""

    operation_variant: PrefillVariant
    padded_batch_size: int
    padded_sequence_length: int
    page_table_width: int
    chunk_page_table_width: int | None

    def key_material(self) -> tuple[tuple[str, str | int | None], ...]:
        return (
            ("operation_variant", self.operation_variant),
            ("padded_batch_size", self.padded_batch_size),
            ("padded_sequence_length", self.padded_sequence_length),
            ("page_table_width", self.page_table_width),
            ("chunk_page_table_width", self.chunk_page_table_width),
        )


@dataclass(frozen=True)
class PreparedPrefill:
    """A request classified once for eager compilation or traced dispatch."""

    request: PrefillRequest
    sampling_params: SamplingParams | None
    sampling_path: SamplingPath
    program_signatures: tuple[PrefillProgramSignature, ...]
    trace_signature: PrefillTraceSignature | None


@dataclass(frozen=True)
class InvocationResult:
    value: Any
    owned: Any
    replay_ownership: "PrefillReplayOwnership | None" = None


@dataclass(frozen=True)
class PrefillReplayOwnership:
    """Explicit ownership split for one hidden-trace postprocess result."""

    trace_owned_hidden_output: Any
    nested_persistent_output: Any | None
    new_logprob_output: Any | None
    replay_local_intermediates: tuple[Any, ...]


@dataclass(frozen=True)
class PrefillHiddenPersistentInputs:
    """Canonical trace-owned model inputs; deliberately sampling-free."""

    device_inputs: PrefillDeviceInputs

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.device_inputs.model_values()


@dataclass
class PrefillReplayState:
    """Program-alias-local postprocessing and sampling state."""

    position_inputs: PrefillPositionInputs
    kpt: tuple[Any, Any, Any] | None
    sampled_output: Any | None = None
    position_signature: int | None = None
    kpt_signature: Any = None

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.position_inputs.values(), self.kpt, self.sampled_output


@dataclass(frozen=True)
class PrefillCapturePlan:
    """Operation hooks consumed by the trace compiler."""

    signature: PrefillTraceSignature
    prepare_inputs: Callable[[], PrefillHiddenPersistentInputs]
    capture: Callable[[PrefillHiddenPersistentInputs], Any]
    prepare_workspace: Callable[[], PrefillReplayState]
    schema_fingerprint: Any
    workspace_fingerprint: Any
    refresh_fields: tuple[str, ...] = ("tokens", "page_table", "last_token", "sampling")


class PrefillRuntime:
    """Plan, execute, trace, and assemble prefill for one execution lane.

    The normal eager call chain is
    `EagerExecutor.prefill_forward()` → `prepare` → `invoke` →
    `assemble`. Trace warmup uses `capture_plan`; replay uses
    `refresh_trace` and `finish_trace` before the same
    `assemble` step. Callers pass host request values and never invoke
    the private chunk-sequence, staging, or sampling helpers directly.

    The runtime borrows the model, mesh, and output reader. It owns staged
    prefill tensors and retains failed releases for retry by `cleanup`.
    """

    def __init__(self, config: PrefillRuntimeConfig) -> None:
        if not isinstance(config, PrefillRuntimeConfig):
            raise TypeError("config must be a PrefillRuntimeConfig")
        self.config = config
        self._transient_orphans: list[TensorResourceOrphan] = []
        self.inputs = PrefillInputStager(
            model=config.model,
            mesh_device=config.mesh_device,
            release_transient=self._release_or_retain_transient,
        )

    # Public API

    @property
    def transient_orphan_count(self) -> int:
        """Return the number of failed transient releases awaiting cleanup."""

        return len(self._transient_orphans)

    def configure_page_table_layout(self, layout: PageTableLayout) -> None:
        """Install final physical KV geometry before allocation or execution."""

        self.config = self.config.with_page_table_layout(layout)

    def can_trace(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
    ) -> bool:
        """Classify trace applicability without allocating planned request tensors."""

        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or int(tokens.shape[0]) == 0:
            return False
        batch_size, token_width = map(int, tokens.shape)
        if prompt_lens is not None and (not isinstance(prompt_lens, torch.Tensor) or prompt_lens.ndim != 1):
            return False
        if start_pos is not None and (not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1):
            return False
        lengths = [token_width] * batch_size if prompt_lens is None else [int(value) for value in prompt_lens]
        cached = [0] * batch_size if start_pos is None else [int(value) for value in start_pos]
        if len(lengths) != batch_size or len(cached) != batch_size:
            return False
        for length, num_cached_tokens in zip(lengths, cached):
            if (
                num_cached_tokens < 0
                or num_cached_tokens % self.config.page_table_layout.block_size
                or length <= num_cached_tokens
                or length > token_width
            ):
                return False
            padded_length = _padded_prefill_length(length - num_cached_tokens)
            invocation_length = (
                _max_prefill_chunk_size(padded_length, self.config.max_prefill_chunk_size)
                if padded_length > self.config.max_prefill_chunk_size
                else padded_length
            )
            # Cached/chunk starts are runtime tensors. Static trace capability
            # is therefore checked against invocation geometry, not the
            # request's current cached offset.
            if not self.config.can_enable_trace(invocation_length, 0):
                return False
        return True

    def prepare(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: SamplingParams | None = None,  # ↓ Sampling
    ) -> tuple[PreparedPrefill, ...]:
        """Plan host inputs once and return immutable requests for execution.

        One public request may produce several prepared requests when batching,
        prefix caching, or chunking requires distinct program invocations.
        """

        self._ensure_usable()
        self._validate_sampling_request(sampling_params)
        layout = self.config.page_table_layout
        requests = _plan_prefill_requests(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            block_size=layout.block_size,
            max_batch_size=self.config.max_batch_size,
            max_prefill_chunk_size=self.config.max_prefill_chunk_size,
            supports_batched_prefill=self.config.supports_batched_prefill,
            # Batched prefill makes prefill logits batch-variant (numerics
            # depend on wave composition). On multi-chip Blackhole this is
            # unverified: seeded token-accuracy/eval gates may break. Measure
            # batched on vs off per BH SKU before enabling it there, and check
            # whether https://github.com/tenstorrent/tt-metal/issues/47238
            # (batch-invariant kernel fix) has landed.
            disable_batched_prefill=(
                self.config.disable_batched_prefill
                or bool(os.environ.get("DISABLE_BATCHED_PREFILL"))
                or (sampling_params is not None and not self.config.batched_prefill_batched_extract)
            ),
            max_prefill_batch_size=self.config.max_prefill_batch_size,
            max_actual_page_table_width=layout.raw_capacity_width,
            canonical_page_table_width=layout.prefill_width,
        )
        prepared = []
        for request in requests:
            request_sampling = _slice_sampling_params(sampling_params, request.source_rows)
            sampling_path = self._classify_sampling_path(request, request_sampling)
            signatures = self._program_signatures(request, sampling_path)
            trace_signature = self._trace_signature(request)
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

    def invoke(self, prepared: PreparedPrefill) -> InvocationResult:
        """Run a prepared request eagerly without replanning or reclassification."""

        self._ensure_usable()
        return self._run_prefill_sequence(prepared)

    def capture_plan(self, prepared: PreparedPrefill) -> PrefillCapturePlan:
        """Describe persistent inputs and capture work for one eligible request."""

        self._ensure_usable()
        if prepared.trace_signature is None:
            raise ValueError("prepared prefill request has no configured trace family")

        def prepare_inputs() -> PrefillHiddenPersistentInputs:
            return self._prepare_hidden_persistent_inputs(prepared)

        def prepare_workspace() -> PrefillReplayState:
            return self._prepare_replay_workspace(prepared)

        def capture(persistent: PrefillHiddenPersistentInputs) -> Any:
            if prepared.request.uses_chunked_prefill:
                return self._run_chunk_hidden_body(
                    prepared,
                    prepared.request.chunks[0],
                    persistent.device_inputs,
                    dynamic_start=True,
                )
            # Eager fill touches active rows only. A captured padded identity
            # must record every physical row so replay never depends on which
            # active count registered the shared trace first; planner-owned
            # ``-1`` rows make the additional fills no-ops for KV ownership.
            return self._run_hidden_body(
                prepared.request,
                persistent.device_inputs,
                fill_rows=prepared.request.padded_batch_size,
            )

        return PrefillCapturePlan(
            signature=prepared.trace_signature,
            prepare_inputs=prepare_inputs,
            capture=capture,
            prepare_workspace=prepare_workspace,
            schema_fingerprint=self._capture_schema_fingerprint(prepared),
            workspace_fingerprint=self._workspace_fingerprint(prepared),
        )

    def refresh_trace(
        self,
        prepared: PreparedPrefill,
        persistent: PrefillHiddenPersistentInputs,
        workspace: PrefillReplayState,
        chunk: PrefillChunk | None = None,
    ) -> None:
        """Refresh borrowed persistent inputs for one replay."""

        request = prepared.request
        if request.uses_chunked_prefill:
            if chunk is None:
                chunk = request.chunks[0]
            self._refresh_chunk_trace_inputs(prepared, chunk, persistent, workspace)
            return
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        # Trace-eligible prefill fixes rotary positions and has no chunk inputs.
        # Rebuilding those capture-owned host tensors on every replay adds TTFT
        # without refreshing any device input; only tokens and page table vary.
        self.inputs.refresh_regular_device_inputs(request, persistent.device_inputs)
        uses_static_single_logits = (
            prepared.sampling_params is None and request.kind == "single" and not request.uses_chunked_prefill
        )
        if not self._uses_static_q128_topk(request, prepared.sampling_path) and not uses_static_single_logits:
            position_value = relative_last
            if workspace.position_signature != position_value:
                position_inputs = self.inputs.prepare_position_inputs_host(
                    relative_last, request.padded_sequence_length
                )
                copy_into_device_tensors(position_inputs.values(), workspace.position_inputs.values())
                workspace.position_signature = position_value
        self._refresh_workspace_sampling(prepared, workspace)

    def _refresh_workspace_sampling(
        self,
        prepared: PreparedPrefill,
        workspace: PrefillReplayState,
    ) -> None:
        if prepared.sampling_path == "topk":
            sampling_batch_size = self._sampling_output_rows(prepared)
            if prepared.sampling_params is None:
                kpt_value = None
            else:
                k, p, temperature, _ = _formatted_sampling_values(prepared.sampling_params, sampling_batch_size)
                kpt_value = k, p, temperature
            if workspace.kpt_signature != kpt_value:
                self._refresh_kpt(
                    workspace.kpt,
                    prepared.sampling_params,
                    sampling_batch_size,
                    force_topk=True,
                )
                workspace.kpt_signature = kpt_value

    def finish_trace(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        workspace: PrefillReplayState,
    ) -> InvocationResult:
        """Post-process a replayed hidden-state tensor into a normal result."""

        replay_local: list[Any] = []
        output = self._finish_regular_prefill(
            prepared,
            hidden,
            workspace.kpt if prepared.sampling_path == "topk" else None,
            workspace.position_inputs,
            sampled_output=workspace.sampled_output,
            owned=replay_local,
        )
        new_logprob = _new_logprob_output(output, workspace.sampled_output)
        sampled_output_alias = output[0] if workspace.sampled_output is not None else None
        caller_owned = _without_borrowed(
            replay_local,
            (hidden, workspace.sampled_output, sampled_output_alias),
        )
        ownership = PrefillReplayOwnership(
            trace_owned_hidden_output=hidden,
            nested_persistent_output=workspace.sampled_output,
            new_logprob_output=new_logprob,
            replay_local_intermediates=_without_borrowed(
                replay_local,
                (hidden, workspace.sampled_output, sampled_output_alias, new_logprob),
            ),
        )
        return InvocationResult(value=output, owned=caller_owned, replay_ownership=ownership)

    def assemble(
        self,
        prepared_results: Iterable[tuple[PreparedPrefill, InvocationResult]],
        *,
        batch_size: int,
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Read phase outputs, restore source-row order, and release transients."""

        sampled = sampling_params is not None
        config = self.config
        vocab_size = int(config.model.vocab_size)
        cluster_shape = config.cluster_shape
        output_logits = torch.zeros(batch_size, 1, vocab_size)
        output_tokens = torch.zeros(batch_size, dtype=torch.int64)
        row_log_probs: list[tuple[tuple[int, ...], Any]] = []

        for prepared, result in prepared_results:
            request = prepared.request
            if sampled != (prepared.sampling_params is not None):
                raise ValueError("prefill result sampling path disagrees with the public request")
            try:
                host_output = config.output_reader.read_synchronized(result.value)
                if isinstance(host_output, tuple):
                    if len(host_output) != 2:
                        raise TypeError("runtime output tuple must contain (output, log_probs)")
                    host_primary, host_log_probs = host_output
                else:
                    host_primary, host_log_probs = host_output, None
                if sampled:
                    uses_static_q128 = self._uses_static_q128_topk(request, prepared.sampling_path)
                    output_rows = _TILE_SIZE if uses_static_q128 else self._sampling_batch_size(request)
                    sampled_tokens = _process_output_tokens(
                        host_primary,
                        output_rows,
                        cluster_shape,
                    )
                    for local_row, source_row in enumerate(request.source_rows):
                        if request.kind == "batched":
                            token_index = local_row
                        elif uses_static_q128:
                            token_index = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                        else:
                            token_index = 0
                        output_tokens[source_row] = sampled_tokens.reshape(-1)[token_index].to(torch.int64)
                    if host_log_probs is not None:
                        if uses_static_q128:
                            host_log_probs = _select_sample_log_prob(host_log_probs, token_index)
                        row_log_probs.append((request.source_rows, host_log_probs))
                elif request.kind == "batched":
                    if isinstance(host_primary, list):
                        for local_row, (source_row, last_token, cached_tokens) in enumerate(
                            zip(request.source_rows, request.last_token_indices, request.cached_tokens)
                        ):
                            output_logits[source_row] = _process_output_prefill(
                                host_primary[local_row],
                                (last_token - cached_tokens) % _TILE_SIZE,
                                vocab_size,
                                cluster_shape,
                            )
                    else:
                        if (
                            isinstance(host_primary, ttnn.Tensor)
                            and host_primary.storage_type() != ttnn.StorageType.HOST
                        ):
                            raise ValueError("prefill output must be on host")
                        combined = _concat_host_output(host_primary, cluster_shape)
                        for local_row, source_row in enumerate(request.source_rows):
                            output_logits[source_row] = combined[0, 0, local_row, :vocab_size].float()
                else:
                    relative_last = (request.last_token_indices[0] - request.cached_tokens[0]) % _TILE_SIZE
                    if request.kind == "single" and not request.uses_chunked_prefill:
                        relative_last = 0
                    output_logits[request.source_rows[0]] = _process_output_prefill(
                        host_primary,
                        relative_last,
                        vocab_size,
                        cluster_shape,
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
        """Retry every transient tensor release that previously failed."""

        failures = release_orphans(self._transient_orphans)
        if failures:
            raise_cleanup_failures(failures)

    # Private implementation

    def _validate_sampling_request(self, sampling_params: SamplingParams | None) -> None:
        if sampling_params is not None and not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")

    def _classify_sampling_path(
        self,
        request: PrefillRequest,
        sampling_params: SamplingParams | None,
    ) -> SamplingPath:
        if sampling_params is None:
            return "logits"
        if self.config.allow_force_argmax and request.kind == "single":
            values = _formatted_sampling_values(sampling_params, self._sampling_batch_size(request))
            if values[3]:
                return "argmax"
        return "topk"

    def _program_signatures(
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
            if self._uses_static_q128_topk(request, sampling_path) or (
                sampling_path == "argmax"
                and request.kind == "single"
                and not request.uses_chunked_prefill
                and request.padded_sequence_length == 128
            ):
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

    def _trace_signature(self, request: PrefillRequest) -> PrefillTraceSignature | None:
        chunk = request.chunks[0]
        invocation_length = chunk.chunk_size
        if not self.config.can_enable_trace(invocation_length, 0):
            return None
        return PrefillTraceSignature(
            operation_variant=(
                "chunked"
                if request.uses_chunked_prefill
                else "regular-batched"
                if request.kind == "batched"
                else "regular-single"
            ),
            padded_batch_size=request.padded_batch_size,
            padded_sequence_length=invocation_length,
            page_table_width=request.page_table_width,
            chunk_page_table_width=(
                int(chunk.chunk_page_table.shape[-1]) if chunk.chunk_page_table is not None else None
            ),
        )

    def _capture_schema_fingerprint(self, prepared: PreparedPrefill) -> tuple[Any, ...]:
        """Describe the sampling-free hidden trace allocation."""

        return (
            "prefill-hidden-v2",
            prepared.trace_signature,
            tuple(field for field, value in prepared.trace_signature.key_material() if value is not None),
        )

    def _workspace_fingerprint(self, prepared: PreparedPrefill) -> tuple[Any, ...]:
        """Describe one program alias's separately owned postprocess state."""

        return (
            "prefill-postprocess-v1",
            prepared.sampling_path,
            self._sampling_output_rows(prepared),
            prepared.sampling_params is not None,
        )

    def _run_prefill_sequence(self, prepared: PreparedPrefill) -> InvocationResult:
        """Execute the request's planned chunks as one eager prefill sequence."""

        request = prepared.request
        final_chunk = request.chunks[-1]
        if request.uses_chunked_prefill:
            final_relative_last = (request.last_token_indices[0] - final_chunk.chunk_start_idx) % final_chunk.chunk_size
        else:
            final_relative_last = max(
                last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)
            )

        owned: list[Any] = []
        kpt = None
        kpt_prepared = False
        final_step_output = None
        final_position_inputs = None
        sampled_output = None
        try:
            if request.uses_chunked_prefill:
                kpt = self._make_device_kpt(
                    prepared.sampling_params,
                    self._sampling_output_rows(prepared),
                    force_topk=prepared.sampling_path == "topk",
                )
                kpt_prepared = True
                _retain_owned(owned, kpt)

            for chunk in request.chunks:
                device_inputs, position_inputs = self.inputs.stage_step(
                    request,
                    chunk,
                    final_relative_last,
                )
                _retain_owned(owned, device_inputs)
                _retain_owned(owned, position_inputs)
                if not kpt_prepared:
                    kpt = self._make_device_kpt(
                        prepared.sampling_params,
                        self._sampling_output_rows(prepared),
                        force_topk=prepared.sampling_path == "topk",
                    )
                    kpt_prepared = True
                    _retain_owned(owned, kpt)
                step_output = self._execute_prefill_step(
                    prepared,
                    chunk,
                    device_inputs,
                    position_inputs,
                )
                if chunk.contains_last_token:
                    final_step_output = step_output
                    final_position_inputs = position_inputs
                    _retain_owned(owned, final_step_output)
                    break
                intermediate_output = step_output
                step_output = None
                failures = self._release_or_retain_transient(intermediate_output)
                if failures:
                    raise_cleanup_failures(failures)

            if final_step_output is None or final_position_inputs is None:
                raise RuntimeError("planned prefill sequence did not produce a final output")
            if not request.uses_chunked_prefill and prepared.sampling_path == "topk":
                sampled_output = self._make_sampling_output(self._sampling_output_rows(prepared))
                _retain_owned(owned, sampled_output)
            output = self._finish_prefill_sequence(
                prepared,
                final_step_output,
                kpt,
                final_position_inputs,
                sampled_output=sampled_output,
                owned=owned,
            )
        except BaseException as primary:
            failures = self._release_or_retain_transient(tuple(owned))
            attach_cleanup_failures(primary, failures)
            raise
        return InvocationResult(value=output, owned=tuple(owned))

    def _execute_prefill_step(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
        position_inputs: PrefillPositionInputs,
    ) -> Any:
        request = prepared.request
        if not request.uses_chunked_prefill:
            return self._run_hidden_body(request, device_inputs)
        return self._run_chunk_body(prepared, chunk, device_inputs, position_inputs)

    def _run_chunk_body(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
        position_inputs: PrefillPositionInputs,
        *,
        dynamic_start: bool = False,
    ) -> Any:
        return self.config.model.prefill_forward(
            self.config.model.embed_prefill(device_inputs.tokens),
            [device_inputs.rotary_cos, device_inputs.rotary_sin],
            user_id=0,
            page_table=device_inputs.page_table,
            chunk_page_table=device_inputs.chunk_page_table,
            chunk_start_idx=None if dynamic_start else chunk.chunk_start_idx,
            get_last_token=-1,
            chunk_start_idx_tensor=device_inputs.chunk_start_idx,
            last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
            last_token_index=(position_inputs.row_index if prepared.sampling_params is not None else None),
        )

    def _run_chunk_hidden_body(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
        *,
        dynamic_start: bool,
    ) -> Any:
        """Run only the shared model body; postprocessing remains alias-local."""

        return self.config.model.prefill_forward(
            self.config.model.embed_prefill(device_inputs.tokens),
            [device_inputs.rotary_cos, device_inputs.rotary_sin],
            user_id=0,
            page_table=device_inputs.page_table,
            chunk_page_table=device_inputs.chunk_page_table,
            chunk_start_idx=None if dynamic_start else chunk.chunk_start_idx,
            get_last_token=-1,
            chunk_start_idx_tensor=device_inputs.chunk_start_idx,
            last_token_slice=None,
            last_token_index=None,
        )

    def _finish_prefill_sequence(
        self,
        prepared: PreparedPrefill,
        final_step_output: Any,
        kpt: tuple[Any, Any, Any] | None,
        position_inputs: PrefillPositionInputs,
        *,
        sampled_output: Any | None,
        owned: list[Any],
    ) -> Any:
        if not prepared.request.uses_chunked_prefill:
            return self._finish_regular_prefill(
                prepared,
                final_step_output,
                kpt,
                position_inputs,
                sampled_output=sampled_output,
                owned=owned,
            )
        if prepared.sampling_params is not None:
            selected = _fit_prefill_sampling_logits(
                final_step_output,
                self._sampling_output_rows(prepared),
            )
            _retain_owned(owned, selected)
            output = self._sample_device(selected, kpt, sampled_output)
        else:
            output = ttnn.untilize(final_step_output, use_multicore=True)
        _retain_owned(owned, output)
        return output

    def _prepare_hidden_persistent_inputs(self, prepared: PreparedPrefill) -> PrefillHiddenPersistentInputs:
        request = prepared.request
        trace_inputs = self.inputs.trace_inputs(request)
        host_inputs = self.inputs.prepare_host_inputs(
            trace_inputs.tokens,
            request.page_table,
            start_pos=trace_inputs.start_pos,
            chunk_page_table=trace_inputs.chunk_page_table,
            chunk_start_idx=trace_inputs.chunk_start_idx,
            last_token_idx=max(request.last_token_indices),
        )
        device_inputs = None
        try:
            device_inputs = self.inputs.stage_device_inputs(host_inputs)
            if prepared.request.uses_chunked_prefill:
                # Prime the exact dynamic rotary-copy programs before trace
                # activation; chunk replay refreshes these buffers in place.
                self.inputs.copy_rotary_inputs(device_inputs)
        except BaseException as primary:
            failures = self._release_or_retain_transient(device_inputs)
            attach_cleanup_failures(primary, failures)
            raise
        return PrefillHiddenPersistentInputs(device_inputs=device_inputs)

    def _prepare_replay_workspace(self, prepared: PreparedPrefill) -> PrefillReplayState:
        trace_inputs = self.inputs.trace_inputs(prepared.request)
        position_inputs = None
        kpt = None
        sampled_output = None
        try:
            sampling_batch_size = self._sampling_output_rows(prepared)
            position_values = allocate_device_tensors(
                self.inputs.prepare_position_inputs_host(
                    trace_inputs.relative_last, trace_inputs.sequence_length
                ).values(),
                mesh_device=self.config.mesh_device,
            )
            position_inputs = PrefillPositionInputs(*position_values)
            kpt = self._make_device_kpt(
                prepared.sampling_params,
                sampling_batch_size,
                force_topk=prepared.sampling_path == "topk",
            )
            if prepared.sampling_params is not None:
                sampled_output = self._make_sampling_output(self._sampling_output_rows(prepared))
        except BaseException as primary:
            failures = self._release_or_retain_transient((position_inputs, kpt, sampled_output))
            attach_cleanup_failures(primary, failures)
            raise
        kpt_signature = None
        if prepared.sampling_params is not None:
            k, p, temperature, _ = _formatted_sampling_values(prepared.sampling_params, sampling_batch_size)
            kpt_signature = k, p, temperature
        return PrefillReplayState(
            position_inputs=position_inputs,
            kpt=kpt,
            sampled_output=sampled_output,
            position_signature=trace_inputs.relative_last,
            kpt_signature=kpt_signature,
        )

    def _refresh_chunk_trace_inputs(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        persistent: PrefillHiddenPersistentInputs,
        workspace: PrefillReplayState,
    ) -> None:
        """Refresh every dynamic chunk input while preserving trace-owned storage."""

        request = prepared.request
        self.inputs.refresh_chunk_device_inputs(request, chunk, persistent.device_inputs)

        final_chunk = request.chunks[-1]
        relative_last = (request.last_token_indices[0] - final_chunk.chunk_start_idx) % final_chunk.chunk_size
        if workspace.position_signature != relative_last:
            position_inputs = self.inputs.prepare_position_inputs_host(relative_last, final_chunk.chunk_size)
            copy_into_device_tensors(position_inputs.values(), workspace.position_inputs.values())
            workspace.position_signature = relative_last
        self._refresh_workspace_sampling(prepared, workspace)

    def _run_hidden_body(
        self,
        request: PrefillRequest,
        device_inputs: PrefillDeviceInputs,
        *,
        fill_rows: int | None = None,
    ) -> Any:
        if fill_rows is None:
            fill_rows = len(request.source_rows)
        if fill_rows < len(request.source_rows) or fill_rows > request.padded_batch_size:
            raise ValueError("fill_rows must cover active rows without exceeding padded batch size")
        return self.config.model.prefill_forward(
            self.config.model.embed_prefill(device_inputs.tokens),
            [device_inputs.rotary_cos, device_inputs.rotary_sin],
            user_id=list(range(fill_rows)) if request.kind == "batched" else 0,
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
        owned: list[Any] | None = None,
    ) -> Any:
        request = prepared.request
        relative_last = [last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens)]
        if request.kind == "batched" and not self.config.batched_prefill_batched_extract:
            hidden = ttnn.reshape(
                hidden,
                [request.padded_batch_size, 1, request.padded_sequence_length, int(hidden.shape[-1])],
            )
            outputs = []
            for local_row, last_token in enumerate(relative_last):
                logits = self.config.model.post_process_prefill_output(
                    hidden[local_row : local_row + 1],
                    last_token,
                )
                _retain_owned(owned, logits)
                output = ttnn.untilize(logits, use_multicore=True)
                _retain_owned(owned, output)
                outputs.append(output)
            return outputs
        if request.kind == "batched":
            padded_last = list(relative_last) + [0] * (request.padded_batch_size - len(relative_last))
            logits = self.config.model.post_process_batched_prefill_output(
                hidden,
                padded_last,
                request.padded_batch_size,
                request.padded_sequence_length,
            )
        elif self._uses_static_q128_topk(request, prepared.sampling_path) or (
            prepared.sampling_params is None and request.kind == "single" and not request.uses_chunked_prefill
        ):
            logits = self.config.model.post_process_prefill_output(hidden, relative_last[0])
        else:
            logits = self.config.model.post_process_prefill_output(
                hidden,
                relative_last[0],
                last_token_slice=(position_inputs.slice_start, position_inputs.slice_end),
                last_token_index=(position_inputs.row_index if prepared.sampling_params is not None else None),
            )
        _retain_owned(owned, logits)
        if prepared.sampling_params is not None:
            selected = _fit_prefill_sampling_logits(
                logits,
                self._sampling_output_rows(prepared),
            )
            _retain_owned(owned, selected)
            output = self._sample_device(selected, kpt, sampled_output)
        else:
            output = ttnn.untilize(logits, use_multicore=True)
            if request.kind == "single" and not request.uses_chunked_prefill:
                _retain_owned(owned, output)
                row = relative_last[0] % _TILE_SIZE
                output = ttnn.slice(output, (0, 0, row, 0), (1, 1, row + 1, int(output.shape[-1])))
        _retain_owned(owned, output)
        return output

    def _sampling_batch_size(self, request: PrefillRequest) -> int:
        if self.config.device_sampling_enabled:
            return self.config.sampling_batch_size
        return request.padded_batch_size

    def _sampling_output_rows(self, prepared: PreparedPrefill) -> int:
        # TT sampling validates K/P/T against the physical logits row count.
        # The static Q128 path retains one complete tile and selects the exact
        # logical row on the host, so its sampling tensors must span that tile.
        if self._uses_static_q128_topk(prepared.request, prepared.sampling_path):
            return _TILE_SIZE
        return self._sampling_batch_size(prepared.request)

    def _uses_static_q128_topk(
        self,
        request: PrefillRequest,
        sampling_path: SamplingPath,
    ) -> bool:
        return (
            self.config.static_q128_topk_supported
            and sampling_path == "topk"
            and request.kind == "single"
            and not request.uses_chunked_prefill
            and request.padded_sequence_length == 128
        )

    def _make_device_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
        host = self._make_host_kpt(sampling_params, batch_size, force_topk)
        if host is None:
            return None
        return tuple(allocate_device_tensors(host, mesh_device=self.config.mesh_device))

    def _make_host_kpt(
        self,
        sampling_params: SamplingParams | None,
        batch_size: int,
        force_topk: bool,
    ) -> tuple[Any, Any, Any] | None:
        if sampling_params is None:
            return None
        values = _formatted_sampling_values(sampling_params, batch_size)
        if self.config.allow_force_argmax and not force_topk and values[3]:
            return None
        k, p, temperature, _ = values
        mapper = ttnn.ReplicateTensorToMesh(self.config.mesh_device)
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
            copy_into_device_tensors(host_kpt, device_kpt)

    def _make_sampling_output(self, batch_size: int) -> Any:
        return ttnn.from_torch(
            torch.zeros((1, 1, 1, int(batch_size)), dtype=torch.int32),
            device=self.config.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.config.mesh_device),
        )

    def _sample_device(
        self,
        logits: Any,
        kpt: tuple[Any, Any, Any] | None,
        sampled_output: Any | None = None,
    ) -> Any:
        if kpt is None:
            return self.config.model.sampling.decode_forward(logits, tt_out_tok=sampled_output)
        return self.config.model.sampling.decode_forward(
            logits,
            k=kpt[0],
            p=kpt[1],
            temp=kpt[2],
            tt_out_tok=sampled_output,
        )

    def _release_or_retain_transient(self, values: Any) -> list[BaseException]:
        orphan = TensorResourceOrphan(values)
        failures = best_effort_deallocate_owned_tensors(orphan.values, orphan.deallocated_tensor_ids)
        if failures:
            self._transient_orphans.append(orphan)
        return failures

    def _ensure_usable(self) -> None:
        if self._transient_orphans:
            raise RuntimeError("PrefillRuntime has unreleased transient resources; cleanup is required")


def _retain_owned(owned: list[Any] | None, value: Any) -> None:
    if owned is None or value is None or any(existing is value for existing in owned):
        return
    owned.append(value)


def _without_borrowed(values: Iterable[Any], borrowed: Iterable[Any]) -> tuple[Any, ...]:
    """Remove trace/workspace-owned leaves from replay-local ownership."""

    borrowed_ids = {id(value) for value in borrowed if value is not None}

    def prune(value):
        if value is None or id(value) in borrowed_ids:
            return None
        if isinstance(value, tuple):
            kept = tuple(item for item in (prune(item) for item in value) if item is not None)
            return kept or None
        if isinstance(value, list):
            kept = [item for item in (prune(item) for item in value) if item is not None]
            return kept or None
        return value

    return tuple(item for item in (prune(value) for value in values) if item is not None)


def _new_logprob_output(output: Any, persistent_sampled_output: Any | None) -> Any | None:
    if persistent_sampled_output is None:
        return None
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("sampled prefill output must contain (tokens, log_probs)")
    return output[1]


def _fit_prefill_sampling_logits(logits, target_batch: int):
    target_batch = int(target_batch)
    if target_batch <= 0:
        raise ValueError("prefill sampling target batch must be positive")
    current_batch = int(logits.shape[2])
    if current_batch == target_batch:
        return logits
    if current_batch > target_batch:
        return ttnn.slice(
            logits,
            (0, 0, 0, 0),
            (logits.shape[0], logits.shape[1], target_batch, logits.shape[3]),
        )
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

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill invocation, trace hooks, and result handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.prefill import assembly as prefill_assembly
from models.common.llm_runtime.prefill import postprocess as prefill_postprocess
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
from models.common.llm_runtime.prefill.sampling_helpers import _formatted_sampling_values, _slice_sampling_params
from models.common.llm_runtime.prefill.signatures import (
    PrefillTraceSignature,
    PreparedPrefill,
    build_program_signatures,
    build_trace_signature,
    capture_schema_fingerprint,
    workspace_fingerprint,
)
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.sampling import SamplingParams


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
        self.postprocessor = prefill_postprocess.PrefillPostprocessor(
            config,
            allocate_device_tensors=lambda values: allocate_device_tensors(
                values,
                mesh_device=self.config.mesh_device,
            ),
            copy_into_device_tensors=copy_into_device_tensors,
        )
        self.assembler = prefill_assembly.PrefillResultAssembler(
            config,
            postprocessor=self.postprocessor,
            release_transient=lambda values: self._release_or_retain_transient(values),
        )

    # Public API

    @property
    def transient_orphan_count(self) -> int:
        """Return the number of failed transient releases awaiting cleanup."""

        return len(self._transient_orphans)

    def configure_page_table_layout(self, layout: PageTableLayout) -> None:
        """Install final physical KV geometry before allocation or execution."""

        self.config = self.config.with_page_table_layout(layout)
        self.postprocessor.configure(self.config)
        self.assembler.configure(self.config)

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
        self.postprocessor.validate_sampling_request(sampling_params)
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
            sampling_path = self.postprocessor.classify_sampling_path(request, request_sampling)
            signatures = build_program_signatures(
                request,
                sampling_path,
                static_q128_topk_supported=self.config.static_q128_topk_supported,
            )
            trace_signature = build_trace_signature(
                request,
                trace_enabled=self.config.can_enable_trace(request.chunks[0].chunk_size, 0),
            )
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

    def invoke(self, prepared: PreparedPrefill) -> prefill_assembly.InvocationResult:
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
            schema_fingerprint=capture_schema_fingerprint(prepared),
            workspace_fingerprint=workspace_fingerprint(
                prepared,
                sampling_output_rows=self.postprocessor.sampling_output_rows(prepared),
            ),
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
        if (
            not self.postprocessor.uses_static_q128_topk(
                request,
                prepared.sampling_path,
            )
            and not uses_static_single_logits
        ):
            position_value = relative_last
            if workspace.position_signature != position_value:
                position_inputs = self.inputs.prepare_position_inputs_host(
                    relative_last, request.padded_sequence_length
                )
                copy_into_device_tensors(position_inputs.values(), workspace.position_inputs.values())
                workspace.position_signature = position_value
        workspace.kpt_signature = self.postprocessor.refresh_workspace_sampling(
            prepared,
            kpt=workspace.kpt,
            kpt_signature=workspace.kpt_signature,
        )

    def finish_trace(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        workspace: PrefillReplayState,
    ) -> prefill_assembly.InvocationResult:
        """Post-process a replayed hidden-state tensor into a normal result."""

        replay_local: list[Any] = []
        output = self.postprocessor.finish_regular_prefill(
            prepared,
            hidden,
            workspace.kpt if prepared.sampling_path == "topk" else None,
            workspace.position_inputs,
            sampled_output=workspace.sampled_output,
            owned=replay_local,
        )
        new_logprob = prefill_postprocess.new_logprob_output(output, workspace.sampled_output)
        sampled_output_alias = output[0] if workspace.sampled_output is not None else None
        caller_owned = prefill_postprocess.without_borrowed(
            replay_local,
            (hidden, workspace.sampled_output, sampled_output_alias),
        )
        ownership = PrefillReplayOwnership(
            trace_owned_hidden_output=hidden,
            nested_persistent_output=workspace.sampled_output,
            new_logprob_output=new_logprob,
            replay_local_intermediates=prefill_postprocess.without_borrowed(
                replay_local,
                (hidden, workspace.sampled_output, sampled_output_alias, new_logprob),
            ),
        )
        return prefill_assembly.InvocationResult(value=output, owned=caller_owned, replay_ownership=ownership)

    def assemble(
        self,
        prepared_results: Iterable[tuple[PreparedPrefill, prefill_assembly.InvocationResult]],
        *,
        batch_size: int,
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Read phase outputs, restore source-row order, and release transients."""

        return self.assembler.assemble(
            prepared_results,
            batch_size=batch_size,
            sampling_params=sampling_params,
        )

    def cleanup(self) -> None:
        """Retry every transient tensor release that previously failed."""

        failures = release_orphans(self._transient_orphans)
        if failures:
            raise_cleanup_failures(failures)

    # Private implementation

    def _run_prefill_sequence(self, prepared: PreparedPrefill) -> prefill_assembly.InvocationResult:
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
                kpt = self.postprocessor.make_device_kpt(
                    prepared.sampling_params,
                    self.postprocessor.sampling_output_rows(prepared),
                    force_topk=prepared.sampling_path == "topk",
                )
                kpt_prepared = True
                prefill_postprocess.retain_owned(owned, kpt)

            for chunk in request.chunks:
                device_inputs, position_inputs = self.inputs.stage_step(
                    request,
                    chunk,
                    final_relative_last,
                )
                prefill_postprocess.retain_owned(owned, device_inputs)
                prefill_postprocess.retain_owned(owned, position_inputs)
                if not kpt_prepared:
                    kpt = self.postprocessor.make_device_kpt(
                        prepared.sampling_params,
                        self.postprocessor.sampling_output_rows(prepared),
                        force_topk=prepared.sampling_path == "topk",
                    )
                    kpt_prepared = True
                    prefill_postprocess.retain_owned(owned, kpt)
                step_output = self._execute_prefill_step(
                    prepared,
                    chunk,
                    device_inputs,
                    position_inputs,
                )
                if chunk.contains_last_token:
                    final_step_output = step_output
                    final_position_inputs = position_inputs
                    prefill_postprocess.retain_owned(owned, final_step_output)
                    break
                intermediate_output = step_output
                step_output = None
                failures = self._release_or_retain_transient(intermediate_output)
                if failures:
                    raise_cleanup_failures(failures)

            if final_step_output is None or final_position_inputs is None:
                raise RuntimeError("planned prefill sequence did not produce a final output")
            if not request.uses_chunked_prefill and prepared.sampling_path == "topk":
                sampled_output = self.postprocessor.make_sampling_output(
                    self.postprocessor.sampling_output_rows(prepared)
                )
                prefill_postprocess.retain_owned(owned, sampled_output)
            output = self.postprocessor.finish_prefill_sequence(
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
        return prefill_assembly.InvocationResult(value=output, owned=tuple(owned))

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
            sampling_batch_size = self.postprocessor.sampling_output_rows(prepared)
            position_values = allocate_device_tensors(
                self.inputs.prepare_position_inputs_host(
                    trace_inputs.relative_last, trace_inputs.sequence_length
                ).values(),
                mesh_device=self.config.mesh_device,
            )
            position_inputs = PrefillPositionInputs(*position_values)
            kpt = self.postprocessor.make_device_kpt(
                prepared.sampling_params,
                sampling_batch_size,
                force_topk=prepared.sampling_path == "topk",
            )
            if prepared.sampling_params is not None:
                sampled_output = self.postprocessor.make_sampling_output(
                    self.postprocessor.sampling_output_rows(prepared)
                )
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
        workspace.kpt_signature = self.postprocessor.refresh_workspace_sampling(
            prepared,
            kpt=workspace.kpt,
            kpt_signature=workspace.kpt_signature,
        )

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

    def _release_or_retain_transient(self, values: Any) -> list[BaseException]:
        orphan = TensorResourceOrphan(values)
        failures = best_effort_deallocate_owned_tensors(orphan.values, orphan.deallocated_tensor_ids)
        if failures:
            self._transient_orphans.append(orphan)
        return failures

    def _ensure_usable(self) -> None:
        if self._transient_orphans:
            raise RuntimeError("PrefillRuntime has unreleased transient resources; cleanup is required")

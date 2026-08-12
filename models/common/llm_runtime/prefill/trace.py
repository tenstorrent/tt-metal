# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill trace capture, refresh, replay state, and ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from models.common.llm_runtime.prefill.inputs import (
    PrefillDeviceInputs,
    PrefillInputStager,
    PrefillPositionInputs,
    allocate_device_tensors,
    copy_into_device_tensors,
)
from models.common.llm_runtime.prefill.plan import PrefillChunk
from models.common.llm_runtime.prefill.postprocess import (
    KPTSignature,
    PrefillPostprocessor,
    new_logprob_output,
    without_borrowed,
)
from models.common.llm_runtime.prefill.result_collector import InvocationResult
from models.common.llm_runtime.prefill.sampling_helpers import _formatted_sampling_values
from models.common.llm_runtime.prefill.signatures import (
    PrefillTraceSignature,
    PreparedPrefill,
    capture_schema_fingerprint,
    workspace_fingerprint,
)
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures


@dataclass(frozen=True)
class PrefillHiddenPersistentInputs:
    """Canonical trace-owned model inputs; deliberately sampling-free."""

    device_inputs: PrefillDeviceInputs

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.device_inputs.model_values()


@dataclass
class PrefillReplayState:
    """Mutable program-alias-local postprocessing and sampling state."""

    position_inputs: PrefillPositionInputs
    kpt: tuple[Any, Any, Any] | None
    sampled_output: Any | None = None
    position_signature: int | None = None
    kpt_signature: KPTSignature = None

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.position_inputs.values(), self.kpt, self.sampled_output


@dataclass(frozen=True)
class PrefillReplayOwnership:
    """Explicit ownership split for one hidden-trace postprocess result."""

    trace_owned_hidden_output: Any
    nested_persistent_output: Any | None
    new_logprob_output: Any | None
    replay_local_intermediates: tuple[Any, ...]


@dataclass(frozen=True)
class PrefillCapturePlan:
    """Operation hooks consumed by the trace compiler."""

    signature: PrefillTraceSignature
    prepare_inputs: Callable[[], PrefillHiddenPersistentInputs]
    capture: Callable[[PrefillHiddenPersistentInputs], Any]
    prepare_workspace: Callable[[], PrefillReplayState]
    schema_fingerprint: tuple[Any, ...]
    workspace_fingerprint: tuple[Any, ...]
    refresh_fields: tuple[str, ...] = ("tokens", "page_table", "last_token", "sampling")


@dataclass(frozen=True)
class PrefillTraceHooks:
    """Narrow input, model-body, postprocess, and cleanup collaborators."""

    input_stager: PrefillInputStager
    postprocessor: PrefillPostprocessor
    run_hidden_body: Callable[..., Any]
    run_chunk_hidden_body: Callable[..., Any]
    release_transient: Callable[[Any], list[BaseException]]


class PrefillTraceLifecycle:
    """Own hidden-trace capture, replay refresh, and replay result ownership."""

    def __init__(self, *, hooks: PrefillTraceHooks) -> None:
        self.hooks = hooks

    def capture_plan(self, prepared: PreparedPrefill) -> PrefillCapturePlan:
        if prepared.trace_signature is None:
            raise ValueError("prepared prefill request has no configured trace family")

        def prepare_inputs() -> PrefillHiddenPersistentInputs:
            return self._prepare_hidden_persistent_inputs(prepared)

        def prepare_workspace() -> PrefillReplayState:
            return self._prepare_replay_state(prepared)

        def capture(persistent: PrefillHiddenPersistentInputs) -> Any:
            if prepared.request.uses_chunked_prefill:
                return self.hooks.run_chunk_hidden_body(
                    prepared,
                    prepared.request.chunks[0],
                    persistent.device_inputs,
                )
            # The captured padded identity must record every physical row so
            # replay does not depend on which active count registered it.
            return self.hooks.run_hidden_body(
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
                sampling_output_rows=self.hooks.postprocessor.sampling_output_rows(prepared),
            ),
        )

    def refresh(
        self,
        prepared: PreparedPrefill,
        persistent: PrefillHiddenPersistentInputs,
        state: PrefillReplayState,
        chunk: PrefillChunk | None = None,
    ) -> None:
        request = prepared.request
        if request.uses_chunked_prefill:
            self._refresh_chunk_inputs(prepared, chunk or request.chunks[0], persistent, state)
            return
        relative_last = max(last - cached for last, cached in zip(request.last_token_indices, request.cached_tokens))
        # Rotary positions are fixed for regular trace families; only tokens
        # and the page table vary on every replay.
        self.hooks.input_stager.refresh_regular_device_inputs(request, persistent.device_inputs)
        uses_static_single_logits = (
            prepared.sampling_params is None and request.kind == "single" and not request.uses_chunked_prefill
        )
        if (
            not self.hooks.postprocessor.uses_static_q128_topk(request, prepared.sampling_path)
            and not uses_static_single_logits
        ):
            if state.position_signature != relative_last:
                position_inputs = self.hooks.input_stager.prepare_position_inputs_host(
                    relative_last,
                    request.padded_sequence_length,
                )
                copy_into_device_tensors(position_inputs.values(), state.position_inputs.values())
                state.position_signature = relative_last
        state.kpt_signature = self.hooks.postprocessor.refresh_workspace_sampling(
            prepared,
            kpt=state.kpt,
            kpt_signature=state.kpt_signature,
        )

    def finish(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        state: PrefillReplayState,
    ) -> InvocationResult:
        replay_local: list[Any] = []
        output = self.hooks.postprocessor.finish_regular_prefill(
            prepared,
            hidden,
            state.kpt if prepared.sampling_path == "topk" else None,
            state.position_inputs,
            sampled_output=state.sampled_output,
            owned=replay_local,
        )
        new_logprob = new_logprob_output(output, state.sampled_output)
        sampled_output_alias = output[0] if state.sampled_output is not None else None
        caller_owned = without_borrowed(
            replay_local,
            (hidden, state.sampled_output, sampled_output_alias),
        )
        ownership = PrefillReplayOwnership(
            trace_owned_hidden_output=hidden,
            nested_persistent_output=state.sampled_output,
            new_logprob_output=new_logprob,
            replay_local_intermediates=without_borrowed(
                replay_local,
                (hidden, state.sampled_output, sampled_output_alias, new_logprob),
            ),
        )
        return InvocationResult(value=output, owned=caller_owned, replay_ownership=ownership)

    def _prepare_hidden_persistent_inputs(self, prepared: PreparedPrefill) -> PrefillHiddenPersistentInputs:
        request = prepared.request
        trace_inputs = self.hooks.input_stager.trace_inputs(request)
        host_inputs = self.hooks.input_stager.prepare_host_inputs(
            trace_inputs.tokens,
            request.page_table,
            start_pos=trace_inputs.start_pos,
            chunk_page_table=trace_inputs.chunk_page_table,
            chunk_start_idx=trace_inputs.chunk_start_idx,
            last_token_idx=max(request.last_token_indices),
        )
        device_inputs = None
        try:
            device_inputs = self.hooks.input_stager.stage_device_inputs(host_inputs)
            if request.uses_chunked_prefill:
                self.hooks.input_stager.copy_rotary_inputs(device_inputs)
        except BaseException as primary:
            failures = self.hooks.release_transient(device_inputs)
            attach_cleanup_failures(primary, failures)
            raise
        return PrefillHiddenPersistentInputs(device_inputs=device_inputs)

    def _prepare_replay_state(self, prepared: PreparedPrefill) -> PrefillReplayState:
        trace_inputs = self.hooks.input_stager.trace_inputs(prepared.request)
        position_inputs = None
        kpt = None
        sampled_output = None
        try:
            sampling_batch_size = self.hooks.postprocessor.sampling_output_rows(prepared)
            position_values = allocate_device_tensors(
                self.hooks.input_stager.prepare_position_inputs_host(
                    trace_inputs.relative_last,
                    trace_inputs.sequence_length,
                ).values(),
                mesh_device=self.hooks.input_stager.mesh_device,
            )
            position_inputs = PrefillPositionInputs(*position_values)
            kpt = self.hooks.postprocessor.make_device_kpt(
                prepared.sampling_params,
                sampling_batch_size,
                force_topk=prepared.sampling_path == "topk",
            )
            if prepared.sampling_params is not None:
                sampled_output = self.hooks.postprocessor.make_sampling_output(
                    self.hooks.postprocessor.sampling_output_rows(prepared)
                )
        except BaseException as primary:
            failures = self.hooks.release_transient((position_inputs, kpt, sampled_output))
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

    def _refresh_chunk_inputs(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        persistent: PrefillHiddenPersistentInputs,
        state: PrefillReplayState,
    ) -> None:
        request = prepared.request
        self.hooks.input_stager.refresh_chunk_device_inputs(request, chunk, persistent.device_inputs)

        final_chunk = request.chunks[-1]
        relative_last = (request.last_token_indices[0] - final_chunk.chunk_start_idx) % final_chunk.chunk_size
        if state.position_signature != relative_last:
            position_inputs = self.hooks.input_stager.prepare_position_inputs_host(
                relative_last,
                final_chunk.chunk_size,
            )
            copy_into_device_tensors(position_inputs.values(), state.position_inputs.values())
            state.position_signature = relative_last
        state.kpt_signature = self.hooks.postprocessor.refresh_workspace_sampling(
            prepared,
            kpt=state.kpt,
            kpt_signature=state.kpt_signature,
        )

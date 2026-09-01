# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Eager prefill chunk-sequence orchestration and ownership."""

from __future__ import annotations

from typing import Any, Callable

from models.common.llm_runtime.prefill.inputs import PrefillDeviceInputs, PrefillInputStager, PrefillPositionInputs
from models.common.llm_runtime.prefill.plan import PrefillChunk, PrefillRequest
from models.common.llm_runtime.prefill.postprocess import PrefillPostprocessor, retain_owned
from models.common.llm_runtime.prefill.result_collector import InvocationResult
from models.common.llm_runtime.prefill.signatures import PreparedPrefill
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures, raise_cleanup_failures


class PrefillSequenceRunner:
    """Execute one prepared eager request while preserving tensor ownership."""

    def __init__(
        self,
        *,
        input_stager: PrefillInputStager,
        postprocessor: PrefillPostprocessor,
        run_hidden_body: Callable[[PrefillRequest, PrefillDeviceInputs], Any],
        run_chunk_body: Callable[[PreparedPrefill, PrefillChunk, PrefillDeviceInputs, PrefillPositionInputs], Any],
        release_transient: Callable[[Any], list[BaseException]],
    ) -> None:
        self.input_stager = input_stager
        self.postprocessor = postprocessor
        self.run_hidden_body = run_hidden_body
        self.run_chunk_body = run_chunk_body
        self.release_transient = release_transient

    def run(self, prepared: PreparedPrefill) -> InvocationResult:
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
                retain_owned(owned, kpt)

            for chunk in request.chunks:
                device_inputs, position_inputs = self.input_stager.stage_step(
                    request,
                    chunk,
                    final_relative_last,
                )
                retain_owned(owned, device_inputs)
                retain_owned(owned, position_inputs)
                if not kpt_prepared:
                    kpt = self.postprocessor.make_device_kpt(
                        prepared.sampling_params,
                        self.postprocessor.sampling_output_rows(prepared),
                        force_topk=prepared.sampling_path == "topk",
                    )
                    kpt_prepared = True
                    retain_owned(owned, kpt)
                step_output = self._execute_step(
                    prepared,
                    chunk,
                    device_inputs,
                    position_inputs,
                )
                if chunk.contains_last_token:
                    final_step_output = step_output
                    final_position_inputs = position_inputs
                    retain_owned(owned, final_step_output)
                    break
                intermediate_output = step_output
                step_output = None
                failures = self.release_transient(intermediate_output)
                if failures:
                    raise_cleanup_failures(failures)

            if final_step_output is None or final_position_inputs is None:
                raise RuntimeError("planned prefill sequence did not produce a final output")
            if not request.uses_chunked_prefill and prepared.sampling_path == "topk":
                sampled_output = self.postprocessor.make_sampling_output(
                    self.postprocessor.sampling_output_rows(prepared)
                )
                retain_owned(owned, sampled_output)
            output = self.postprocessor.finish_prefill_sequence(
                prepared,
                final_step_output,
                kpt,
                final_position_inputs,
                sampled_output=sampled_output,
                owned=owned,
            )
        except BaseException as primary:
            failures = self.release_transient(tuple(owned))
            attach_cleanup_failures(primary, failures)
            raise
        return InvocationResult(value=output, owned=tuple(owned))

    def _execute_step(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
        position_inputs: PrefillPositionInputs,
    ) -> Any:
        if not prepared.request.uses_chunked_prefill:
            return self.run_hidden_body(prepared.request, device_inputs)
        return self.run_chunk_body(prepared, chunk, device_inputs, position_inputs)

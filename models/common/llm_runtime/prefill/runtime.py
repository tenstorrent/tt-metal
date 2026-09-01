# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public prefill orchestration facade and transient cleanup."""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import torch

from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.prefill import postprocess as prefill_postprocess
from models.common.llm_runtime.prefill import result_collector as prefill_result_collector
from models.common.llm_runtime.prefill import sequence_runner as prefill_sequence_runner
from models.common.llm_runtime.prefill import trace as prefill_trace
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
from models.common.llm_runtime.prefill.sampling_helpers import _slice_sampling_params
from models.common.llm_runtime.prefill.signatures import (
    PreparedPrefill,
    build_program_signatures,
    build_trace_signature,
)
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.sampling import SamplingParams


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
        self.assembler = prefill_result_collector.PrefillResultAssembler(
            config,
            postprocessor=self.postprocessor,
            release_transient=lambda values: self._release_or_retain_transient(values),
        )
        self.sequence_runner = prefill_sequence_runner.PrefillSequenceRunner(
            input_stager=self.inputs,
            postprocessor=self.postprocessor,
            run_hidden_body=lambda *args, **kwargs: self._run_hidden_body(*args, **kwargs),
            run_chunk_body=lambda *args, **kwargs: self._run_chunk_body(*args, **kwargs),
            release_transient=lambda values: self._release_or_retain_transient(values),
        )
        self.trace = prefill_trace.PrefillTraceLifecycle(
            hooks=prefill_trace.PrefillTraceHooks(
                input_stager=self.inputs,
                postprocessor=self.postprocessor,
                run_hidden_body=lambda *args, **kwargs: self._run_hidden_body(*args, **kwargs),
                run_chunk_hidden_body=lambda *args, **kwargs: self._run_chunk_hidden_body(*args, **kwargs),
                release_transient=lambda values: self._release_or_retain_transient(values),
            )
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

    def invoke(self, prepared: PreparedPrefill) -> prefill_result_collector.InvocationResult:
        """Run a prepared request eagerly without replanning or reclassification."""

        self._ensure_usable()
        return self.sequence_runner.run(prepared)

    def capture_plan(self, prepared: PreparedPrefill) -> prefill_trace.PrefillCapturePlan:
        """Describe persistent inputs and capture work for one eligible request."""

        self._ensure_usable()
        return self.trace.capture_plan(prepared)

    def refresh_trace(
        self,
        prepared: PreparedPrefill,
        persistent: prefill_trace.PrefillHiddenPersistentInputs,
        workspace: prefill_trace.PrefillReplayState,
        chunk: PrefillChunk | None = None,
    ) -> None:
        """Refresh borrowed persistent inputs for one replay."""

        self.trace.refresh(prepared, persistent, workspace, chunk)

    def finish_trace(
        self,
        prepared: PreparedPrefill,
        hidden: Any,
        workspace: prefill_trace.PrefillReplayState,
    ) -> prefill_result_collector.InvocationResult:
        """Post-process a replayed hidden-state tensor into a normal result."""

        return self.trace.finish(prepared, hidden, workspace)

    def assemble(
        self,
        prepared_results: Iterable[tuple[PreparedPrefill, prefill_result_collector.InvocationResult]],
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

    def _run_chunk_body(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
        position_inputs: PrefillPositionInputs,
    ) -> Any:
        return self.config.model.prefill_forward(
            self.config.model.embed_prefill(device_inputs.tokens),
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

    def _run_chunk_hidden_body(
        self,
        prepared: PreparedPrefill,
        chunk: PrefillChunk,
        device_inputs: PrefillDeviceInputs,
    ) -> Any:
        """Run only the shared model body; postprocessing remains alias-local."""

        return self.config.model.prefill_forward(
            self.config.model.embed_prefill(device_inputs.tokens),
            [device_inputs.rotary_cos, device_inputs.rotary_sin],
            user_id=0,
            page_table=device_inputs.page_table,
            chunk_page_table=device_inputs.chunk_page_table,
            chunk_start_idx=None,
            get_last_token=-1,
            chunk_start_idx_tensor=device_inputs.chunk_start_idx,
            last_token_slice=None,
            last_token_index=None,
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

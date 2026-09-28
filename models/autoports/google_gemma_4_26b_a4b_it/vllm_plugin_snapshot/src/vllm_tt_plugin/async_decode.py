# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

import torch
import ttnn

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists, ModelRunnerOutput
from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL
from vllm_tt_plugin.structured_output import has_structured_outputs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm_tt_plugin.input_batch import CachedRequestState
    from vllm_tt_plugin.model_input import TTModelInput
    from vllm_tt_plugin.model_runner import TTModelRunner


@dataclass(frozen=True)
class TTDecodeSubmission:
    """Carries the raw result of decode submission until finalization."""

    tt_out: Any | None
    read_events: list[Any] | None
    batch_size_per_dp: list[int]
    sampling_params: Any
    perform_device_sampling: bool


@dataclass(frozen=True)
class TTFinalizedDecode:
    """Normalized decode result after TT event waits and host processing."""

    tt_out: torch.Tensor
    tt_log_probs: torch.Tensor | None


def _is_host_decode_output(tt_out: Any) -> bool:
    if isinstance(tt_out, torch.Tensor):
        return True
    if isinstance(tt_out, tuple):
        return all(
            tensor is None or isinstance(tensor, torch.Tensor) for tensor in tt_out
        )
    return False


@dataclass(frozen=True)
class SubmittedStepContext:
    """Immutable snapshot of the host state associated with one decode submit."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    request_states: tuple[CachedRequestState, ...]
    submit_time_ns: int


@dataclass(frozen=True)
class CompletedDecodeStep:
    """Decode output that has completed readback but is not yet applied."""

    sampled_token_ids: torch.Tensor
    logprobs: LogprobsLists | None
    context: SubmittedStepContext
    completion_time_ns: int


class DeferredDecodeOutput(AsyncModelRunnerOutput):
    """Run the deferred device readback exactly once, from whichever caller
    reaches it first.

    Two callers race for the same step from different threads: vLLM's
    ``UniProcExecutor`` resolves it via ``get_output`` on its async-output
    thread when async scheduling is on, while the runner's drain
    (``TTAsyncDecodeController.wait_for_all_pending_async_steps``) resolves it
    via ``ensure_finalized`` on the engine thread. ``_finalize_lock`` makes the
    readback run exactly once across both threads; a second concurrent readback
    of the same device submission corrupts the decode output. The completion
    event is set here, when the readback actually runs, not only inside
    ``get_output``. That is the invariant the drain depends on: vLLM 0.22's
    ``step_with_batch_queue`` schedules the next batch before resolving the
    prior future, so a drain that merely ``event.wait()``-ed would block forever
    on an event nothing else has reached yet.
    """

    _completion_event: threading.Event
    _finalize_lock: threading.Lock
    _finalized: bool
    _cached_output: Any

    def _init_deferred(self) -> None:
        self._finalized = False
        self._cached_output = None
        self._finalize_lock = threading.Lock()

    def ensure_finalized(self) -> Any:
        if self._finalized:
            return self._cached_output
        with self._finalize_lock:
            if not self._finalized:
                self._cached_output = self._get_output_impl()
                self._finalized = True
                self._completion_event.set()
        return self._cached_output

    def is_resolved(self) -> bool:
        return self._completion_event.is_set()

    def get_output(self) -> Any:
        return self.ensure_finalized()

    def _get_output_impl(self) -> Any:
        raise NotImplementedError


class AsyncTTModelRunnerOutput(DeferredDecodeOutput):
    """Wrap a non-blocking single-process TT decode submission plus async read.

    Handles both a plain single-process decode and a lane-DP decode: when
    ``scheduled_rows`` is set the read-back goes through the merged lane batch
    (``TTLaneInputBatch.extract_output``), otherwise through ``_get_output_tokens``.
    """

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        completion_event: threading.Event,
        context: SubmittedStepContext,
        scheduled_rows: list[int] | None = None,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input
        self._completion_event = completion_event
        self._context = context
        self._scheduled_rows = scheduled_rows
        self._init_deferred()

    def set_grammar_bitmask(self, bitmask: torch.Tensor) -> None:
        """Attach a sample-time grammar bitmask before the deferred read.

        The runner reorders the bitmask on the engine thread (where the batch
        layout still matches this step's forward) and calls this; the read on
        the output thread then applies it through ``model_input``.
        """
        self._model_input = replace(self._model_input, grammar_bitmask=[bitmask])

    def _get_output_impl(self) -> ModelRunnerOutput:
        completed = self._controller.complete_decode_step(
            submission=self._submission,
            model_input=self._model_input,
            context=self._context,
            scheduled_rows=self._scheduled_rows,
        )
        self._controller.enqueue_completed_decode_step(completed)
        return self._controller.build_runner_output_from_completed_step(completed)


class AsyncTTDPGatherOutput(DeferredDecodeOutput):
    """Wrap a non-blocking DP decode submission plus async read submit."""

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        completion_event: threading.Event,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input
        self._completion_event = completion_event
        self._init_deferred()

    def _get_output_impl(self) -> tuple[torch.Tensor, list]:
        finalized = self._controller.finalize_decode(self._submission)
        runner = self._controller.runner
        if finalized is None:
            return runner.pack_dp_results(
                [torch.tensor([], dtype=torch.int32)]
                * len(self._submission.batch_size_per_dp),
                [None] * len(self._submission.batch_size_per_dp),
            )

        sampled_token_ids_per_dp, logprobs_per_dp = runner._get_output_tokens(
            tt_out=finalized.tt_out,
            tt_log_probs=finalized.tt_log_probs,
            sampling_params=self._submission.sampling_params,
            model_input=self._model_input,
            batch_size_per_dp=self._submission.batch_size_per_dp,
            perform_device_sampling=self._submission.perform_device_sampling,
            is_decode=True,
        )
        return runner.pack_dp_results(sampled_token_ids_per_dp, logprobs_per_dp)


class TTAsyncDecodeController:
    """Own the TT async decode lifecycle for a `TTModelRunner`."""

    def __init__(self, runner: TTModelRunner):
        self.runner = runner

    def capture_submitted_step_context(
        self, req_ids: list[str] | None = None
    ) -> SubmittedStepContext:
        """Snapshot the submitted requests for deferred async state apply.

        ``req_ids`` is the merged output order: the lane path passes its
        scheduled slots' requests (sparse rows), so the index map is built from
        their position. ``None`` takes the condensed front-packed batch, whose
        ``req_id_to_index`` already equals that position map.
        """
        runner = self.runner
        if req_ids is None:
            num_reqs = runner.input_batch.num_reqs
            req_ids = list(runner.input_batch.req_ids[:num_reqs])
            req_id_to_index = dict(runner.input_batch.req_id_to_index)
        else:
            req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        return SubmittedStepContext(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            request_states=tuple(runner.requests[req_id] for req_id in req_ids),
            submit_time_ns=time.perf_counter_ns(),
        )

    def steady_decode_base_enabled(self, *, dp_gather: bool) -> bool:
        runner = self.runner
        if dp_gather:
            if not runner.scheduler_config.async_scheduling:
                return False
        else:
            if not runner.non_dp_async_scheduling:
                return False
            if runner.parallel_config.data_parallel_size != 1:
                return False
        if runner.trace_mode == "none":  # noqa: SIM103
            return False
        return True

    def steady_decode_scheduler_invariants_met(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        runner = self.runner
        cached_reqs = scheduler_output.scheduled_cached_reqs
        is_prompt = (len(scheduler_output.scheduled_new_reqs) > 0) or bool(
            cached_reqs.resumed_req_ids
        )
        if is_prompt or runner._decode_layout_changed_since_last_decode:
            return False
        # A page-table reset must use host tokens after pending decode has
        # completed. Empty per-group deltas preserve steady decode eligibility.
        if any(block_ids and any(block_ids) for block_ids in cached_reqs.new_block_ids):
            return False
        # Structured outputs are detected from the scheduler state, not from a
        # prepared bitmask: the non-DP/lane paths now defer grammar to sample
        # time and pass ``grammar_output=None`` here, while gathered DP still
        # passes a bitmask. Either signal disables steady decode so the grammar
        # constraint is never skipped by an overlapped step.
        if (
            scheduler_output.pending_structured_output_tokens
            or grammar_output is not None
            or has_structured_outputs(runner.requests, scheduler_output, None)
        ):
            return False
        input_batch = runner.input_batch
        if not input_batch.no_penalties:
            return False
        if not input_batch.no_allowed_token_ids:
            return False
        if input_batch.sampling.bad_words_token_ids:
            return False
        max_num_logprobs = input_batch.max_num_logprobs
        # Treat logprobs=0 as a real logprobs request so decode does not
        # bypass the slower path that preserves per-token logprob metadata.
        if max_num_logprobs is not None:
            return False
        if input_batch.sampling.has_active_logitsprocs():
            return False
        if runner.model_config.logits_processors:
            return False
        return runner.check_perform_device_sampling(
            is_decode=True,
            has_structured_outputs=False,
        )

    def can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=False):
            return False
        return self.steady_decode_scheduler_invariants_met(
            scheduler_output, grammar_output
        )

    def can_attempt_steady_dp_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=True):
            return False
        if scheduler_output is None or scheduler_output.total_num_scheduled_tokens == 0:
            return True
        return self.steady_decode_scheduler_invariants_met(
            scheduler_output, grammar_output
        )

    def can_use_steady_decode_fast_path(self, model_input: TTModelInput) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=False):
            return False
        if model_input.prompt_lens is not None:
            return False
        if not model_input.perform_device_sampling:
            return False
        if model_input.reset_batch:
            return False
        if model_input.grammar_bitmask[0] is not None:
            return False
        if (
            model_input.prompt_tokens is not None
            or model_input.output_tokens is not None
        ):
            return False
        if model_input.allowed_token_ids_mask_list[0] is not None:
            return False
        if model_input.bad_words_token_ids_list[0]:
            return False
        max_num_logprobs = model_input.max_num_logprobs[0]
        if max_num_logprobs is not None:  # noqa: SIM103
            return False
        return True

    def enqueue_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        with self.runner._steady_decode_lock:
            self.runner._completed_decode_steps.append(completed)

    def register_pending_async_step(
        self,
        step: DeferredDecodeOutput,
        *,
        overlap_ok: bool,
    ) -> None:
        with self.runner._steady_decode_lock:
            self.runner._pending_async_steps.append(step)
            self.runner._pending_async_overlap_ok.append(overlap_ok)

    def prune_finished_async_events(self) -> None:
        with self.runner._steady_decode_lock:
            while (
                self.runner._pending_async_steps
                and self.runner._pending_async_steps[0].is_resolved()
            ):
                self.runner._pending_async_steps.popleft()
                self.runner._pending_async_overlap_ok.popleft()

    def drain_completed_decode_steps(self) -> list[CompletedDecodeStep]:
        completed: list[CompletedDecodeStep] = []
        with self.runner._steady_decode_lock:
            while self.runner._completed_decode_steps:
                completed.append(self.runner._completed_decode_steps.popleft())
        return completed

    def apply_ready_completed_decode_steps(self) -> None:
        for completed in self.drain_completed_decode_steps():
            self.apply_completed_decode_step(completed)
        self.prune_finished_async_events()

    def wait_for_all_pending_async_steps(self) -> None:
        # Drive each pending readback to completion here rather than blocking on
        # its event: the engine has not popped these futures yet (and on 0.22
        # will not until after this returns), so nothing else will set the
        # events. ``ensure_finalized`` is idempotent, so the engine's later
        # ``get_output`` on the same step returns the cached result.
        with self.runner._steady_decode_lock:
            steps = list(self.runner._pending_async_steps)
        for step in steps:
            step.ensure_finalized()
        self.apply_ready_completed_decode_steps()

    def must_drain_pending_async_steps(
        self,
        steady_decode_candidate: bool,
    ) -> bool:
        with self.runner._steady_decode_lock:
            if not self.runner._pending_async_steps:
                return False
            if not steady_decode_candidate:
                return True
            return any(
                not overlap_ok for overlap_ok in self.runner._pending_async_overlap_ok
            )

    def complete_decode_step(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        context: SubmittedStepContext,
        scheduled_rows: list[int] | None = None,
    ) -> CompletedDecodeStep:
        """Finalize a single-process async decode read into sampled tokens.

        When ``scheduled_rows`` is given this is a lane-DP step: the merged slot
        batch is read back via ``TTLaneInputBatch.extract_output``. Otherwise it
        is a plain single-process step read back via ``_get_output_tokens``.
        """
        finalized = self.finalize_decode(submission)
        if finalized is None:
            sampled_token_ids = torch.empty((0, 1), dtype=torch.int32)
            logprobs = None
        elif scheduled_rows is not None:
            sampled_token_ids, logprobs = self.runner.lane_batch.extract_output(
                self.runner,
                finalized.tt_out,
                finalized.tt_log_probs,
                model_input,
                scheduled_rows,
                is_decode=True,
            )
        else:
            sampled_token_ids_per_dp, logprobs_per_dp = self.runner._get_output_tokens(
                tt_out=finalized.tt_out,
                tt_log_probs=finalized.tt_log_probs,
                sampling_params=submission.sampling_params,
                model_input=model_input,
                batch_size_per_dp=submission.batch_size_per_dp,
                perform_device_sampling=submission.perform_device_sampling,
                is_decode=True,
            )
            sampled_token_ids = sampled_token_ids_per_dp[0]
            logprobs_tensors = logprobs_per_dp[0] if logprobs_per_dp else None
            logprobs = logprobs_tensors.tolists() if logprobs_tensors else None
        return CompletedDecodeStep(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            context=context,
            completion_time_ns=time.perf_counter_ns(),
        )

    def build_runner_output_from_completed_step(
        self,
        completed: CompletedDecodeStep,
    ) -> ModelRunnerOutput:
        return self.runner._build_runner_output(
            sampled_token_ids=completed.sampled_token_ids,
            logprobs=completed.logprobs,
            req_ids=completed.context.req_ids,
            req_id_to_index=completed.context.req_id_to_index,
        )

    def apply_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        self.runner._apply_sampled_tokens_to_state(
            sampled_token_ids=completed.sampled_token_ids,
            req_ids=completed.context.req_ids,
            request_states=completed.context.request_states,
        )

    def submit_async_non_dp_decode(
        self,
        model_input: TTModelInput,
        *,
        steady_decode_fast_path: bool,
    ) -> AsyncTTModelRunnerOutput:
        event = threading.Event()
        context = self.capture_submitted_step_context()
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
        )
        if submission.tt_out is None:
            event.set()
        step = AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=event,
            context=context,
        )
        self.register_pending_async_step(step, overlap_ok=steady_decode_fast_path)
        return step

    def submit_async_dp_decode(
        self,
        model_input: TTModelInput,
        *,
        allow_decode_overlap: bool = True,
    ) -> AsyncTTDPGatherOutput:
        """Submit a non-blocking gathered-DP decode step.

        ``allow_decode_overlap`` is the caller's veto on overlapping this step's
        device work with the next scheduler step: it only sets ``overlap_ok``
        when both it is True (the default) and ``can_use_steady_decode_fast_path``
        holds. Passing False forces the next step to drain this one regardless of
        the fast-path check.
        """
        overlap_ok = allow_decode_overlap and self.can_use_steady_decode_fast_path(
            model_input
        )
        completion_event = threading.Event()
        submission = self.submit_decode(
            model_input, read_from_device=False, async_read=True
        )
        if submission.tt_out is None:
            completion_event.set()
        step = AsyncTTDPGatherOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=completion_event,
        )
        self.register_pending_async_step(step, overlap_ok=overlap_ok)
        return step

    def submit_async_lane_decode(
        self,
        model_input: TTModelInput,
        context: SubmittedStepContext,
        scheduled_rows: list[int],
    ) -> AsyncTTModelRunnerOutput:
        """Submit a non-blocking single-process multi-lane decode step."""
        overlap_ok = self.can_use_steady_decode_fast_path(model_input)
        completion_event = threading.Event()
        submission = self.submit_decode(
            model_input, read_from_device=False, async_read=True
        )
        if submission.tt_out is None:
            completion_event.set()
        step = AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=completion_event,
            context=context,
            scheduled_rows=scheduled_rows,
        )
        self.register_pending_async_step(step, overlap_ok=overlap_ok)
        return step

    def submit_decode(
        self,
        model_input: TTModelInput,
        *,
        read_from_device: bool,
        async_read: bool = False,
    ) -> TTDecodeSubmission:
        runner = self.runner
        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        if not any(bs > 0 for bs in batch_size_per_dp):
            return TTDecodeSubmission(
                tt_out=None,
                read_events=None,
                batch_size_per_dp=batch_size_per_dp,
                sampling_params=sampling_params,
                perform_device_sampling=perform_device_sampling,
            )

        kwargs: dict[str, Any] = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": runner.kv_caches,
            "start_pos": model_input.input_positions,
        }
        # Hybrid attention models route per-layer block tables; the
        # runner already populated ``block_tables_per_layer`` at
        # submission time when the kv_cache_config has multiple groups.
        # Legacy/uniform models leave it as ``None`` and never see the
        # kwarg.
        if model_input.block_tables_per_layer is not None:
            kwargs["page_tables_per_layer"] = model_input.block_tables_per_layer
        if perform_device_sampling:
            sampling_param_dict = {
                field.name: (
                    getattr(sampling_params, field.name).tolist()
                    if getattr(sampling_params, field.name) is not None
                    else None
                )
                for field in fields(sampling_params)
            }
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            kwargs["sampling_params"] = type(sampling_params)(**sampling_param_dict)
            if model_input.prompt_tokens is not None:
                assert model_input.output_tokens is not None
                kwargs["prompt_tokens"] = model_input.prompt_tokens
                kwargs["output_tokens"] = model_input.output_tokens
            kwargs["reset_batch"] = model_input.reset_batch
            if model_input.output_token_counts is not None and getattr(
                runner.model, "needs_sampling_output_counts", False
            ):
                kwargs["output_token_counts"] = model_input.output_token_counts
        # Two consumers, gated differently: ``decode_forward`` moves per-slot GDN
        # state with it always, the seed manager reindexes only on device sampling.
        if model_input.slot_remap is not None:
            kwargs["slot_remap"] = model_input.slot_remap

        enc_dec_kwargs: dict[str, Any] = {}
        if runner.request_specific_rope:
            if any(
                req_id not in runner.previous_req_ids
                for req_id in runner.input_batch.req_ids
            ):
                enc_dec_kwargs = {
                    "rope_deltas_all_users": [
                        runner.requests[req_id].mrope_position_delta
                        for req_id in runner.input_batch.req_ids
                    ]
                }
            else:
                enc_dec_kwargs = {"rope_deltas_all_users": None}
            runner.previous_req_ids = set(runner.input_batch.req_ids)

        enable_trace = runner.trace_mode in ["all", "decode_only"]
        tt_out = runner.model.decode_forward(
            **kwargs,
            **enc_dec_kwargs,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
        )
        read_events = None
        if async_read:
            if hasattr(runner.model, "read_decode_output"):
                tt_out, read_events = cast(
                    tuple[Any, list[Any]],
                    runner.model.read_decode_output(tt_out, async_read=True),
                )
            else:
                is_host_tensor = isinstance(tt_out, torch.Tensor)
                is_host_tensor_tuple = isinstance(tt_out, tuple) and all(
                    tensor is None or isinstance(tensor, torch.Tensor)
                    for tensor in tt_out
                )
                if not (is_host_tensor or is_host_tensor_tuple):
                    raise AttributeError(
                        "TT model must implement read_decode_output() "
                        "unless decode_forward() already returns host tensors"
                    )
        return TTDecodeSubmission(
            tt_out=tt_out,
            read_events=read_events,
            batch_size_per_dp=batch_size_per_dp,
            sampling_params=sampling_params,
            perform_device_sampling=perform_device_sampling,
        )

    def finalize_decode(
        self,
        submission: TTDecodeSubmission,
    ) -> TTFinalizedDecode | None:
        runner = self.runner
        if submission.tt_out is None:
            return None

        if submission.read_events is not None:
            for read_event in submission.read_events:
                ttnn.event_synchronize(read_event)
            tt_out = submission.tt_out
        else:
            tt_out = submission.tt_out

        is_host_output = _is_host_decode_output(tt_out)
        if not is_host_output and hasattr(runner.model, "process_decode_output_host"):
            tt_out = runner.model.process_decode_output_host(
                tt_out,
                is_tokens=submission.perform_device_sampling,
            )
        elif not is_host_output:
            raise AttributeError(
                "TT model must implement process_decode_output_host() "
                "unless decode output is already a torch tensor"
            )

        tt_log_probs = None
        assert isinstance(submission.sampling_params.enable_log_probs, torch.Tensor)
        if (
            submission.perform_device_sampling
            and submission.sampling_params.enable_log_probs.any()
        ):
            assert isinstance(tt_out, tuple) and len(tt_out) == 2
            tt_out, tt_log_probs = tt_out
        elif isinstance(tt_out, tuple):
            tt_out, _ = tt_out

        return TTFinalizedDecode(tt_out=tt_out, tt_log_probs=tt_log_probs)

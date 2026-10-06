# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-process, multi-lane scheduling for TT data-parallel execution.

TT Llama3 70B Galaxy model is single-weights and single-execute, but
keeps 4 data-parallel (DP) KV caches. In this layout,
one engine process drives all replicas: each replica is a
"lane" with its own slice of requests, but every step the lanes must execute in
lockstep against a single gathered batch on device.

This module bridges vLLM's single-queue scheduler to that layout:

- ``TTLaneCoordinator`` owns one independent :class:`TTScheduler` per lane (each
  with its own KV cache manager and request queues) and stitches the per-lane
  results into one engine-facing ``SchedulerOutput``. Because each lane drives a
  physically separate DP submesh KV cache, the per-lane schedulers allocate
  block IDs independently: block IDs repeat across lanes, which is correct since
  the model runner routes each lane's block table to its own submesh.
- ``merge_lane_scheduler_outputs`` performs that stitching.
- ``TTStepPlan`` rides along on the merged output so the model runner receives
  a device-row plan instead of lane internals.

Because the device executes all lanes together, every lane in a step must agree
on a single scheduling mode (all-prefill or all-decode); the coordinator
negotiates that mode before scheduling any lane.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    SchedulerOutput,
)
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm_tt_plugin.config import (
    get_tt_data_parallel_size,
    get_tt_per_lane_max_num_seqs,
)
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

logger = init_tt_logger(__name__)


@dataclass(frozen=True)
class TTStepPlan:
    """Runner-facing physical layout for one single-process lane-DP step."""

    is_decode: bool
    capacity: int
    scheduled_req_ids: tuple[str, ...]
    scheduled_rows: tuple[int, ...]
    input_rows: tuple[int, ...]
    req_id_to_row: dict[str, int]
    batch_size_per_dp: tuple[int, ...]
    prefill_empty_slots: tuple[int, ...] | None


@dataclass(frozen=True)
class _LaneStepState:
    # The raw, unmerged output for each lane (empty for idle lanes), in lane
    # order. The coordinator's update paths read each lane's scheduled requests
    # back from it; the runner sees only ``plan``.
    lane_outputs: list[SchedulerOutput]
    plan: TTStepPlan


# The coordinator stashes per-step lane state on the engine-facing
# ``SchedulerOutput`` so the runner sees only ``TTStepPlan`` while the
# coordinator's own update paths can still recover each lane's unmerged output.
# This relies on ``SchedulerOutput`` being a plain attrs-mutable dataclass with
# no ``__slots__``; the attribute is invisible in that class's definition, so
# every access goes through the two helpers below to keep the name in one place.
_TT_STEP_STATE_ATTR = "_tt_step_state"


def _set_tt_step_state(
    scheduler_output: SchedulerOutput, state: _LaneStepState
) -> None:
    setattr(scheduler_output, _TT_STEP_STATE_ATTR, state)


def _get_tt_step_state(scheduler_output: SchedulerOutput) -> _LaneStepState | None:
    return getattr(scheduler_output, _TT_STEP_STATE_ATTR, None)


def get_tt_step_plan(scheduler_output: SchedulerOutput) -> TTStepPlan | None:
    state = _get_tt_step_state(scheduler_output)
    return None if state is None else state.plan


def merge_lane_scheduler_outputs(
    lane_outputs: list[SchedulerOutput],
) -> SchedulerOutput:
    """Merge per-lane scheduler outputs into one engine-facing output.

    Each field is combined according to its kind: list fields are concatenated
    in lane order, per-request dicts are merged (request IDs are globally
    unique, so keys never collide), set fields are unioned, and booleans are
    OR-ed. ``num_common_prefix_blocks`` is the elementwise max across lanes,
    since the gathered batch must reserve the largest common prefix any lane
    needs.

    Block IDs carried in ``scheduled_cached_reqs`` / ``scheduled_new_reqs`` are
    lane-local and may repeat across lanes; this is intentional — each lane
    indexes its own submesh KV cache — so they are concatenated verbatim
    without any global remapping.
    """
    if not lane_outputs:
        return SchedulerOutput.make_empty()

    # The general merge below already handles the "no lane scheduled any tokens"
    # case correctly: every list/dict/set field aggregates to empty while
    # ``finished_req_ids`` and ``free_encoder_mm_hashes`` (the bookkeeping the
    # runner still needs to release that state) are unioned/extended as usual.
    # So there is no separate fast path -- one path covers both.
    scheduled_new_reqs: list = []
    cached = CachedRequestData.make_empty()
    num_scheduled_tokens: dict[str, int] = {}
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    num_common_prefix_blocks: list[int] = []
    finished_req_ids: set[str] = set()
    free_encoder_mm_hashes: list[str] = []
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests = False
    pending_structured_output_tokens = False
    num_invalid_spec_tokens: dict[str, int] | None = None

    for out in lane_outputs:
        scheduled_new_reqs.extend(out.scheduled_new_reqs)
        # ``CachedRequestData`` is a struct-of-arrays (parallel ``req_ids`` /
        # ``new_token_ids`` / ``new_block_ids`` / ... lists plus
        # ``resumed_req_ids`` and ``all_token_ids``); concatenate each field so
        # the merged object stays internally consistent. An idle lane returns a
        # ``CachedRequestData.make_empty()`` struct (all-empty arrays), so the
        # ``num_reqs > 0`` guard just skips lanes that would contribute nothing
        # -- it is a clarity/efficiency skip, not a correctness guard
        # (``make_empty()`` builds a fresh object each call, so there is no
        # shared instance to protect).
        lane_cached = out.scheduled_cached_reqs
        if lane_cached.num_reqs > 0:
            cached.req_ids.extend(lane_cached.req_ids)
            cached.resumed_req_ids |= lane_cached.resumed_req_ids
            cached.new_token_ids.extend(lane_cached.new_token_ids)
            cached.all_token_ids.update(lane_cached.all_token_ids)
            cached.new_block_ids.extend(lane_cached.new_block_ids)
            cached.num_computed_tokens.extend(lane_cached.num_computed_tokens)
            cached.num_output_tokens.extend(lane_cached.num_output_tokens)
        num_scheduled_tokens.update(out.num_scheduled_tokens)
        scheduled_spec_decode_tokens.update(out.scheduled_spec_decode_tokens)
        scheduled_encoder_inputs.update(out.scheduled_encoder_inputs)
        # Take the elementwise max so the merged batch reserves enough common
        # prefix blocks for the most demanding lane. All lanes share one
        # ``kv_cache_config`` and so report one entry per KV cache group, making
        # the per-lane lists equal length; ``strict=True`` turns any mismatch
        # into an error instead of silently dropping the extra groups.
        if out.num_common_prefix_blocks:
            if not num_common_prefix_blocks:
                num_common_prefix_blocks = list(out.num_common_prefix_blocks)
            else:
                num_common_prefix_blocks = [
                    max(a, b)
                    for a, b in zip(
                        num_common_prefix_blocks,
                        out.num_common_prefix_blocks,
                        strict=True,
                    )
                ]
        finished_req_ids |= out.finished_req_ids
        free_encoder_mm_hashes.extend(out.free_encoder_mm_hashes)
        # A lane can preempt a running request under its own KV pressure. Union
        # the per-lane sets (request IDs are globally unique); stay None unless
        # some lane reported a preemption, matching the base output's "absent"
        # representation rather than an empty set.
        if out.preempted_req_ids:
            if preempted_req_ids is None:
                preempted_req_ids = set()
            preempted_req_ids |= out.preempted_req_ids
        has_structured_output_requests |= out.has_structured_output_requests
        pending_structured_output_tokens |= out.pending_structured_output_tokens
        # Stays None unless some lane reported invalid spec tokens, matching the
        # base output's "absent" representation rather than an empty dict.
        if out.num_invalid_spec_tokens:
            if num_invalid_spec_tokens is None:
                num_invalid_spec_tokens = {}
            num_invalid_spec_tokens.update(out.num_invalid_spec_tokens)

    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=free_encoder_mm_hashes,
        preempted_req_ids=preempted_req_ids,
        has_structured_output_requests=has_structured_output_requests,
        pending_structured_output_tokens=pending_structured_output_tokens,
        num_invalid_spec_tokens=num_invalid_spec_tokens,
    )


class TTLaneCoordinator(SchedulerInterface):
    """Single-process multi-lane scheduler for TT gathered-batch execution.

    Owns one fully independent :class:`TTScheduler` per lane. Each lane
    scheduler has its own waiting/running queues and its own KV cache manager,
    so the coordinator behaves like several co-located gathered-DP engines: a
    request belongs to exactly one lane and only that lane's scheduler ever
    sees it. Every lane's KV cache manager is sized identically (the same
    ``kv_cache_config``) because each lane drives a physically separate, equally
    sized DP submesh cache; block IDs are therefore lane-local and repeat across
    lanes.

    The coordinator implements :class:`SchedulerInterface` by routing requests
    to their lane, negotiating the single shared scheduling mode each step,
    running every lane, and merging the per-lane results into one engine-facing
    ``SchedulerOutput`` tagged with :class:`TTStepPlan` so the runner only sees
    device rows and slots.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.structured_output_manager = structured_output_manager
        self.log_stats = log_stats
        # Number of DP replicas (lanes) sharing this process.
        self.num_lanes = get_tt_data_parallel_size(vllm_config)
        # Max concurrent running requests a single lane may hold.
        self._per_lane_max = get_tt_per_lane_max_num_seqs(vllm_config)
        self._req_to_lane: dict[str, int] = {}
        self._req_to_row: dict[str, int] = {}
        self._free_slots_by_lane: list[list[int]] = [
            list(range(self._per_lane_max)) for _ in range(self.num_lanes)
        ]
        # No KV connector on TT; surfaced for engine-core attribute access.
        self.connector: KVConnectorBase_V1 | None = None

        # Each lane scheduler must cap its running set at the *per-lane*
        # capacity, not the global ``max_num_seqs`` the coordinator sees. The
        # base scheduler derives its hard running cap from
        # ``scheduler_config.max_num_seqs`` (``max_num_running_reqs``), so if
        # every lane saw the global value (``num_lanes * per_lane``) the lanes'
        # combined running set could reach ``num_lanes * max_num_seqs`` and
        # overflow the runner's merged persistent batch
        # (``req_index >= max_num_reqs``). Give each lane a config view whose
        # ``max_num_seqs`` is the per-lane cap so the running-cap derivation is
        # correct at construction, rather than mutating ``max_num_running_reqs``
        # after the fact. The coordinator keeps the original ``vllm_config`` so
        # the global capacity (used for KV/model sizing in the runner) is
        # untouched.
        per_lane_vllm_config = self._build_per_lane_vllm_config(vllm_config)

        # One independent scheduler per lane. Each gets the same kv_cache_config
        # (every submesh cache is the same size) and its own KV cache manager,
        # so lane block-ID spaces are independent. Per-lane stats are disabled;
        # the coordinator aggregates stats itself.
        self.lanes: list[TTScheduler] = [
            TTScheduler(
                per_lane_vllm_config,
                kv_cache_config,
                structured_output_manager,
                block_size,
                mm_registry,
                include_finished_set,
                log_stats=False,
            )
            for _ in range(self.num_lanes)
        ]

    def _build_per_lane_vllm_config(self, vllm_config: VllmConfig) -> VllmConfig:
        """Return a ``vllm_config`` view whose ``max_num_seqs`` is per-lane.

        Shallow-copies ``vllm_config`` and its ``scheduler_config`` so the
        per-lane ``max_num_seqs`` override does not leak back onto the shared
        config the coordinator and model runner read for global sizing. All
        lanes share this one view; the base scheduler only reads
        ``max_num_seqs`` from it (to set ``max_num_running_reqs``), never
        mutates it.
        """
        per_lane_vllm_config = copy.copy(vllm_config)
        per_lane_scheduler_config = copy.copy(vllm_config.scheduler_config)
        per_lane_scheduler_config.max_num_seqs = self._per_lane_max
        per_lane_vllm_config.scheduler_config = per_lane_scheduler_config
        return per_lane_vllm_config

    # ------------------------------------------------------------------
    # Lane selection / mode negotiation
    # ------------------------------------------------------------------

    def _pick_lane(self) -> int:
        """Choose the least-loaded lane for a newly arriving request.

        Scores each lane ``waiting * 4 + running`` and picks the lowest
        (ties resolve to the lowest lane index). This is intentionally the same
        load score vLLM uses to route requests across DP engines
        (``DPLBAsyncMPClient.get_core_engine_for_request`` in
        ``vllm/v1/engine/core_client.py``): queued requests are weighted more
        heavily than running ones because each will cost a future prefill. The
        lane coordinator stands in for vLLM's multi-process DP load balancer, so
        it keeps the same policy.

        Assignment is static: a request is bound to one lane at admission
        (recorded in the coordinator's maps) and never migrates,
        exactly as vLLM DP binds each request to one engine at intake with no
        later migration. So a lane can sit idle while another has queued work --
        no worse than the gathered-DP behavior this replaces. Cross-lane
        rebalancing (e.g. work-stealing) is a possible future follow-up, not
        part of this change.
        """
        best_lane = 0
        best_score: int | None = None
        for lane, sched in enumerate(self.lanes):
            score = len(sched.waiting) * 4 + len(sched.running)
            if best_score is None or score < best_score:
                best_score = score
                best_lane = lane
        return best_lane

    def _local_prefill_intent(self, sched: TTScheduler) -> int:
        """Whether this lane *wants* to prefill this step (1) or not (0).

        A running prefill continuation must always make progress, even when the
        lane is at its request capacity. New queued requests need either no
        running work or spare capacity to be admitted alongside running decodes.
        """
        has_waiting = bool(sched.waiting)
        has_running = bool(sched.running)
        has_partial_prefill = any(request.is_prefill_chunk for request in sched.running)
        has_capacity = len(sched.running) < self._per_lane_max
        return int(
            has_partial_prefill or (has_waiting and ((not has_running) or has_capacity))
        )

    def _negotiate_forced_mode(self) -> TTSchedulingMode:
        """Pick the single mode (prefill- or decode-only) all lanes will run.

        The device executes all lanes together and cannot mix prefill with
        decode, so the lanes must agree. If *any* lane wants to prefill, the
        whole step is prefill-only; otherwise it is decode-only. Lanes without
        work for the chosen mode simply contribute an empty batch.
        """
        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)

    def _schedule_all_lanes(
        self, forced_mode: TTSchedulingMode
    ) -> list[SchedulerOutput]:
        """Run every lane scheduler under ``forced_mode``.

        Idle lanes are scheduled too (rather than short-circuited to an empty
        output) so each lane drains its own pending ``finished_req_ids`` for the
        runner's cleanup; an empty schedule for an idle lane is cheap.

        No ``try/finally`` is needed to restore ``DEFAULT`` mode. Every lane's
        forced mode is set here immediately before its ``schedule()`` call and
        read only by that one call, so it never persists in a meaningful way
        across coordinator steps: the next ``_schedule_all_lanes`` re-sets it
        before scheduling again. A mid-loop ``schedule()`` exception fails the
        whole engine step, and any lane left in ``forced_mode`` is harmless
        because it is overwritten before it is next read. The reset after each
        call simply keeps lanes in a tidy ``DEFAULT`` state between steps.
        """
        lane_outputs: list[SchedulerOutput] = []
        for sched in self.lanes:
            sched.set_forced_mode(forced_mode)
            lane_outputs.append(sched.schedule())
            sched.set_forced_mode(TTSchedulingMode.DEFAULT)
        return lane_outputs

    # ------------------------------------------------------------------
    # SchedulerInterface: scheduling
    # ------------------------------------------------------------------

    def _lane_for_req(self, req_id: str) -> int:
        return self._req_to_lane[req_id]

    def _assign_slot(self, req_id: str, lane: int) -> int:
        existing = self._req_to_row.get(req_id)
        if existing is not None:
            return existing
        free_slots = self._free_slots_by_lane[lane]
        if not free_slots:
            raise ValueError(
                f"lane {lane} has no free slot (capacity {self._per_lane_max})"
            )
        local_slot = free_slots.pop(0)
        row = lane * self._per_lane_max + local_slot
        self._req_to_row[req_id] = row
        return row

    def _release_slot(self, req_id: str) -> None:
        row = self._req_to_row.pop(req_id, None)
        if row is None:
            return
        lane = row // self._per_lane_max
        local_slot = row % self._per_lane_max
        free_slots = self._free_slots_by_lane[lane]
        if local_slot not in free_slots:
            free_slots.append(local_slot)
            free_slots.sort()

    def _lane_of_scheduled_reqs(
        self, lane_outputs: list[SchedulerOutput]
    ) -> dict[str, int]:
        lane_of_req: dict[str, int] = {}
        for lane, lane_output in enumerate(lane_outputs):
            for req_id in lane_output.num_scheduled_tokens:
                lane_of_req[req_id] = lane
        return lane_of_req

    def _build_step_plan(
        self,
        lane_outputs: list[SchedulerOutput],
        merged: SchedulerOutput,
        is_decode: bool,
    ) -> TTStepPlan:
        lane_of_req = self._lane_of_scheduled_reqs(lane_outputs)
        for req_id in merged.finished_req_ids:
            self._release_slot(req_id)
            self._req_to_lane.pop(req_id, None)

        resumed_req_ids = set(merged.scheduled_cached_reqs.resumed_req_ids)
        for req_id in resumed_req_ids:
            self._release_slot(req_id)

        for req_id in merged.num_scheduled_tokens:
            lane = lane_of_req.get(req_id, self._req_to_lane.get(req_id))
            if lane is None:
                raise KeyError(f"no TT lane recorded for scheduled request {req_id!r}")
            self._req_to_lane[req_id] = lane
            self._assign_slot(req_id, lane)

        scheduled_pairs = [
            (req_id, self._req_to_row[req_id])
            for req_id in merged.num_scheduled_tokens
            if req_id in self._req_to_row
        ]
        scheduled_pairs.sort(key=lambda item: item[1])
        scheduled_req_ids = tuple(req_id for req_id, _ in scheduled_pairs)
        scheduled_rows = tuple(row for _, row in scheduled_pairs)
        capacity = self.num_lanes * self._per_lane_max
        input_rows = tuple(range(capacity)) if is_decode else scheduled_rows
        if is_decode:
            batch_size_per_dp = tuple([self._per_lane_max] * self.num_lanes)
            prefill_empty_slots = None
        else:
            per_lane_counts = [0] * self.num_lanes
            for row in scheduled_rows:
                per_lane_counts[row // self._per_lane_max] += 1
            batch_size_per_dp = tuple(per_lane_counts)
            prefill_empty_slots = scheduled_rows
        return TTStepPlan(
            is_decode=is_decode,
            capacity=capacity,
            scheduled_req_ids=scheduled_req_ids,
            scheduled_rows=scheduled_rows,
            input_rows=input_rows,
            req_id_to_row=dict(self._req_to_row),
            batch_size_per_dp=batch_size_per_dp,
            prefill_empty_slots=prefill_empty_slots,
        )

    def schedule(self) -> SchedulerOutput:
        forced_mode = self._negotiate_forced_mode()
        lane_outputs = self._schedule_all_lanes(forced_mode)
        merged = merge_lane_scheduler_outputs(lane_outputs)

        # Decode fallback: a forced prefill step can schedule zero tokens (no
        # chunked prefill + KV pressure means no full prefill fits). If any lane
        # has running decodes, falling back to a decode-only step keeps them
        # advancing — without this the step makes no global progress and the
        # engine livelocks. Mirrors the base scheduler's DEFAULT-mode fallback,
        # which the forced mode bypasses.
        if (
            forced_mode == TTSchedulingMode.PREFILL_ONLY
            and merged.total_num_scheduled_tokens == 0
            and any(
                any(not request.is_prefill_chunk for request in sched.running)
                for sched in self.lanes
            )
        ):
            # The discarded prefill pass already drained each lane's
            # finished/freed-encoder bookkeeping; carry it onto the decode pass
            # so the runner still releases that state.
            carried_finished = merged.finished_req_ids
            carried_free_encoder = merged.free_encoder_mm_hashes
            forced_mode = TTSchedulingMode.DECODE_ONLY
            lane_outputs = self._schedule_all_lanes(forced_mode)
            merged = merge_lane_scheduler_outputs(lane_outputs)
            merged.finished_req_ids |= carried_finished
            merged.free_encoder_mm_hashes = (
                carried_free_encoder + merged.free_encoder_mm_hashes
            )

        is_decode = forced_mode == TTSchedulingMode.DECODE_ONLY
        plan = self._build_step_plan(lane_outputs, merged, is_decode)
        _set_tt_step_state(merged, _LaneStepState(lane_outputs=lane_outputs, plan=plan))
        return merged

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Mirrors the base scheduler, but over the union of every lane's
        # requests (request IDs are globally unique). Row order within the
        # bitmask is irrelevant: the runner remaps rows back to batch positions
        # by request ID via reorder_grammar_bitmask_for_tt_batch.
        requests: dict[str, Request] = {}
        for sched in self.lanes:
            requests.update(sched.requests)
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None
        bitmask = self.structured_output_manager.grammar_bitmask(
            requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    # ------------------------------------------------------------------
    # SchedulerInterface: output handling
    # ------------------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        state = _get_tt_step_state(scheduler_output)
        if state is None:
            return {}

        # Each lane scheduler processes only its own SchedulerOutput. The merged
        # model_runner_output is passed through unchanged: its req_id_to_index
        # spans all lanes and the per-request dicts are keyed by (globally
        # unique) request ID, and the base update loop is driven by the lane's
        # num_scheduled_tokens, so a lane only ever touches its own requests.
        per_lane_outputs: list[dict[int, EngineCoreOutputs]] = [
            sched.update_from_output(lane_output, model_runner_output)
            for sched, lane_output in zip(self.lanes, state.lane_outputs, strict=True)
        ]
        merged = self._merge_engine_core_outputs(per_lane_outputs)

        # Lanes run with stats disabled; attach the coordinator's aggregate
        # stats here, mirroring the base scheduler's placement.
        stats = self.make_stats()
        if stats is not None:
            eco = next(iter(merged.values()), None)
            if eco is None:
                merged[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        return merged

    @staticmethod
    def _merge_engine_core_outputs(
        per_lane_outputs: list[dict[int, EngineCoreOutputs]],
    ) -> dict[int, EngineCoreOutputs]:
        """Merge per-lane ``{client_index: EngineCoreOutputs}`` dicts.

        Concatenates each client's request outputs and unions its
        finished-request set. Lists/sets are rebound to fresh objects rather
        than mutated in place to avoid touching msgspec defaults.
        """
        merged: dict[int, EngineCoreOutputs] = {}
        for lane_dict in per_lane_outputs:
            for client_index, eco in lane_dict.items():
                existing = merged.get(client_index)
                if existing is None:
                    merged[client_index] = eco
                    continue
                if eco.outputs:
                    existing.outputs = existing.outputs + eco.outputs
                if eco.finished_requests:
                    if existing.finished_requests is None:
                        existing.finished_requests = set(eco.finished_requests)
                    else:
                        existing.finished_requests = (
                            existing.finished_requests | eco.finished_requests
                        )
        return merged

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for sched in self.lanes:
            sched.update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        state = _get_tt_step_state(scheduler_output)
        if state is None:
            return
        for sched, lane_output in zip(self.lanes, state.lane_outputs, strict=True):
            sched.update_draft_token_ids_in_output(draft_token_ids, lane_output)

    # ------------------------------------------------------------------
    # SchedulerInterface: request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        # Bind the request before handing it to a lane scheduler. The binding is
        # coordinator-owned state; the shared vLLM Request object stays generic.
        lane = self._req_to_lane.get(request.request_id)
        if lane is None:
            lane = self._pick_lane()
            self._req_to_lane[request.request_id] = lane
        self.lanes[lane].add_request(request)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        # Broadcast to every lane; a lane no-ops for IDs it does not hold, so no
        # request->lane map is needed.
        for sched in self.lanes:
            sched.finish_requests(request_ids, finished_status)

    # ------------------------------------------------------------------
    # SchedulerInterface: queries / lifecycle
    # ------------------------------------------------------------------

    def get_num_unfinished_requests(self) -> int:
        return sum(sched.get_num_unfinished_requests() for sched in self.lanes)

    def has_finished_requests(self) -> bool:
        return any(sched.has_finished_requests() for sched in self.lanes)

    def get_request_counts(self) -> tuple[int, int]:
        num_running = 0
        num_waiting = 0
        for sched in self.lanes:
            running, waiting = sched.get_request_counts()
            num_running += running
            num_waiting += waiting
        return num_running, num_waiting

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        results = [
            sched.reset_prefix_cache(reset_running_requests, reset_connector)
            for sched in self.lanes
        ]
        return all(results)

    def reset_encoder_cache(self) -> None:
        for sched in self.lanes:
            sched.reset_encoder_cache()

    def make_stats(self) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        num_running, num_waiting = self.get_request_counts()
        kv_cache_usage = sum(
            sched.kv_cache_manager.usage for sched in self.lanes
        ) / len(self.lanes)
        return SchedulerStats(
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting,
            kv_cache_usage=kv_cache_usage,
        )

    def shutdown(self) -> None:
        for sched in self.lanes:
            sched.shutdown()

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return None

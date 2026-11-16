# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S0's multi-user resident adapter under asynchronous scheduling (S1a).

Plan: the lead's S1a ask (2026-11-12) on M5B_PLAN_v2.md §8; contract: vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md
§4c-4d. Results: bench/resident/s1a/S1A_RESULTS.md. S0's adapter (generator_resident_multi_vllm.py, accepted with
bench/resident/s0/s0_SHA256SUMS, unchanged) plus:

- supports_async_spec_decode. A speculating launch keeps async scheduling, so an ordinary step (every row declining)
  overlaps the next step's host work, as B32's decode does. A verify never overlaps: the plugin drains before it builds
  over one, and builds one only after the steps before it are applied (async_decode.must_drain_pending_async_steps).
- An ordinary step's commit and proposal run a step late, at the next build's drain
  (async_decode.apply_completed_decode_step), after the next step was launched. So a request's launched steps queue
  (QueuedSlotState.step is a FIFO here), and a proposal knows when it is stale: a newer step of its request is already
  on the device, so the anchor it was handed is behind the request.
- A stale proposal that decides to enter cannot draft from its anchor. It offers a one-token placeholder instead, which
  makes the plugin drain (async_decode._next_step_verifies) and hand over the next step's proposal fresh; the request
  enters there, one position later. The placeholder never reaches a verify: that fresh proposal replaces or erases it
  before the build reads it (_verify refuses a block that is not a retained proposal).
- Deferred tap upkeep reads taps_buffer(launch % 2), which launch + 2 overwrites. Upkeep a stale proposal deferred runs
  before the launch that would overwrite its buffer; the rest still runs inside the device window after the launch.
- A release resets only the slot's tables. resident_m5_3's release_slot also re-uploads every row's position from the
  host copy of the last write_rows, which undoes the device-fed advance of the other rows when no reload follows: under
  async scheduling a length-capped request leaves the batch a step before its finish arrives, and the steps between
  are steady (no reload). Found by the S1a host tests (bench/resident/s1a).
"""

from collections import deque

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt import resident_taps
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import (
    Gemma4ResidentMultiForCausalLM,
    SlotState,
    draft_cap,
)
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import (
    PLACEHOLDER_TOKEN_ID,
    Offer,
    ResidentSpecPolicy,
)
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM


class QueuedSlotState(SlotState):
    """SlotState whose `step` is the oldest of the request's launched, not yet proposed, steps: setting a step queues
    it, setting None takes the oldest off (S0's propose reads `step, state.step = state.step, None`)."""

    @property
    def step(self):
        steps = self.__dict__.setdefault("_steps", deque())
        return steps[0] if steps else None

    @step.setter
    def step(self, value):
        steps = self.__dict__.setdefault("_steps", deque())
        if value is not None:
            steps.append(value)
        elif steps:
            steps.popleft()

    @property
    def stale(self):
        """After the propose took its step: a newer step of this request is already launched."""
        return bool(self.__dict__.get("_steps"))


class Gemma4ResidentMultiAsyncForCausalLM(Gemma4ResidentMultiForCausalLM):
    model_capabilities = {**Gemma4ResidentMultiForCausalLM.model_capabilities, "supports_async_spec_decode": True}

    def _new_state(self, key, slot, tap_slot):
        book = resident_taps.TapBook() if tap_slot is not None and self.taps is not None else None
        state = QueuedSlotState(key, slot, tap_slot if book is not None else None, ResidentSpecPolicy(self.K), book)
        state.entry_pending = False
        return state

    def _launch(self, R, M=1):
        """Upkeep deferred from launch L reads taps_buffer(L % 2), which launch L + 2 overwrites: it runs first."""
        upcoming = self.engine.launches
        early = [entry for entry in self.deferred_taps if entry[1].launch <= upcoming - 2]
        if early:
            self.deferred_taps = [entry for entry in self.deferred_taps if entry[1].launch > upcoming - 2]
            for state, step, c in early:
                state.deferred -= 1
            for state, step, c in early:
                if self.slots.states.get(state.slot) is state:
                    self._upkeep_rows(state, step, c)
            self.counters["deferred_tap_steps_before_launch"] += len(early)
        super()._launch(R, M)

    def _upkeep_rows(self, state, step, c):
        """S0's _tap_upkeep without its drain-first: the caller runs the request's entries oldest first."""
        self.taps.stage(state.tap_slot, self.engine.taps_buffer(step.launch), step.row, c)
        self.taps.ring_insert(state.tap_slot, state.book.insert(step.position, c))
        if (step.position + c) // resident_taps.TILE > step.position // resident_taps.TILE:
            for page in state.book.complete_pages():
                state.book.check_write(page)
                self.taps.flush(state.tap_slot, page, state.book.entries(page))
                state.book.flushed(page)

    # ------------------------------------------------------------------ propose

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """S0's propose (generator_resident_multi_vllm.propose_draft_tokens), with placeholder offers for stale
        entries: (r, anchor) rows that offer one placeholder token and draft nothing."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        k = int(num_drafts)
        if k != self.K:
            raise ValueError(f"Expected {self.K} draft columns, got {k}")
        rows = committed_tokens.shape[0]
        ids = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        counts = accepted_counts.reshape(-1).tolist()
        tokens, positions = committed_tokens.tolist(), committed_positions.tolist()
        live = [r for r in range(min(rows, len(self.row_slots)))
                if positions[r][0] >= 0 and self.row_slots[r] is not None]
        cap = draft_cap(len(live))
        if cap and any(positions[r][counts[r] - 1] + self.BLOCK > self._spec_limit() for r in live):
            cap = 0  # S0's capacity-edge guard
            self.counters["capacity_edge_steps"] += 1
        jobs = []
        for r in live:
            state = self.slots.states.get(self.row_slots[r])
            if state is None:
                continue  # released between the step and its propose
            step, state.step = state.step, None
            if step is None:
                continue
            c = counts[r]
            position = positions[r][c - 1]
            if position != step.position + c:
                raise ValueError(f"Row {r}: anchor position {position} is not the step's {step.position} plus {c}")
            job = self._propose(state, step, c, tokens[r][c - 1], position, cap)
            if job == "placeholder":
                ids[r, 0] = tokens[r][c - 1]
                valid[r] = 1
                self.counters["placeholder_offers"] += 1
            elif job is not None:
                jobs.append((r, state, job))
        if jobs:
            drafted = self.drafter.draft([job for _, _, job in jobs])
            for r, state, job in jobs:
                drafts = [int(t) for t in drafted[state.tap_slot][:cap]]
                ids[r, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
                valid[r] = len(drafts)
                state.offer = Offer(job[1], job[2], drafts)
                state.regime = "hold"
                state.counters["drafts"] += 1
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    def _propose(self, state, step, c, anchor, position, cap):
        """S0's _propose, plus staleness: an entry decided on a stale proposal is carried to the next, fresh one."""
        stale = state.stale
        if not stale and not state.entry_pending:
            return super()._propose(state, step, c, anchor, position, cap)
        if state.regime == "hold":
            raise RuntimeError(f"Engine slot {state.slot}: a stale proposal in hold (a verify is never overlapped)")
        if (step.host or step.dropped or cap == 0 or state.book is None or self.drafter is None or state.suppressed
                or state.exhausted or position + self.BLOCK > self._spec_limit()):
            if state.entry_pending:
                state.counters["spec_entries_cancelled"] += 1
            state.entry_pending = False
            return super()._propose(state, step, c, anchor, position, cap)  # S0's declines, before any decide
        state.counters["committed_tokens"] += c
        if state.entry_pending and not stale:
            state.entry_pending = False
            self._tap_upkeep(state, step, c)
            self._enter(state, position)
            return (state.tap_slot, anchor, position, position, 0)
        if not state.entry_pending and not state.policy.decide(position, entering=True):
            return self._decline(state, step, c, asked=True)
        state.entry_pending = True
        state.counters["spec_entries_deferred"] += 1
        state.deferred += 1
        self.deferred_taps.append((state, step, c))
        return "placeholder"

    # ------------------------------------------------------------------ lifecycle

    def release_request(self, slot):
        """S0's release (generator_resident_multi_vllm.release_request), with the engine's tables reset for the slot
        and nothing else (module docstring). A slot still in the device feed marks the feed stale: the plugin reloads
        after the layout change a finish makes, and a step that does not would be refused."""
        Gemma4ForCausalLM.release_request(self, slot)
        state = self.slots.release(slot)
        if state is None:
            self.counters["releases_unknown"] += 1
            return
        self.deferred_taps = [entry for entry in self.deferred_taps if entry[0] is not state]
        if self.engine is not None:
            scratch = torch.tensor(self._group_scratch(), dtype=torch.int32)[:, state.slot]
            self.engine.write_tables(scratch[None, :, None].expand(1, -1, self.slot_raw.shape[2]), slots=[state.slot])
        if state.slot in self.row_slots:
            self.feed_stale = True
        self.slot_raw[:, state.slot] = -1
        self.seen = None
        self.positions = [-1 if s == state.slot else p for s, p in zip(self.row_slots, self.positions)]
        self.row_slots = [None if s == state.slot else s for s in self.row_slots]
        if state.counters:
            logger.debug("Resident request done (slot {}): {}", state.slot, dict(state.counters))
        self.counters.update({f"done_{k}": v for k, v in state.counters.items()})
        self.counters["releases"] += 1

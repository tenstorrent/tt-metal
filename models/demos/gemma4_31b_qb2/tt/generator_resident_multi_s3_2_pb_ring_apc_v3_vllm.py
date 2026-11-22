# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""S1 `batch` class v3, for M7 (bench/resident/m7/M7_S1_INTERACTION.md + ADDENDUM_1; LEAD_DECISIONS_M7 M5):
generator_resident_multi_s3_2_pb_ring_apc_v2_vllm.Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM (update 5, unchanged)
with two differences:

- decode: v1 refuses `spec_mode` ("S1's batch class serves no speculation"). v3 serves it. It runs v1's decode body
  itself, the rows' request keys for S0's `_layout`, which S0's `_verify` also maps its rows through, and then the
  classes after v1 in the MRO. So a verify launch finds each row's engine slot by request id, as an ordinary launch
  does; under prefix caching full block 0 is shared, so a block-derived key would be ambiguous.
- release: once v1 (the request key) and S0 (the engine slot and its tap slot) have released a request,
  `released(state)` is called with its SlotState, for M7's ring hand-off (M7_REBASE_v2_ADDENDUM_2 §1).

What v3 relies on and leaves as it is (M7_S1_INTERACTION C1-C8):
- C1: per-request speculation state is S0's SlotState, keyed by engine slot. S1's rebinds move only the plugin slot ->
  engine slot map, and S1's same-step row order hands S0 each row's own key;
- C2: an admission past 0 (S1's `_admit`) goes through SlotTable.admit with `self._new_state`, as S0's start-0
  admission does, with the same tap and drafter resets;
- C4: S0's `_enter` seeds the drafter at the request's first SPEC step from resident_taps.window(position). The plugin
  does not mark a request's last prefill chunk; the first decode position follows it for a cold request, a hit and a
  preempted resume alike. For a hit, n - c >= MIN_SUFFIX (2,080) puts that window inside the recomputed suffix [c, n);
- C8: a restored prefill history is the tail pool slot's own tensors (s1_tail_pool.TailPool.restore). v3 writes none of
  it, and nothing M7 adds on the prefill path may.
Update 5's S1 modules stay byte-frozen (C7): `single-user` imports prefix_resume and tail_slots from v1's module."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_v2_vllm import (
    Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM,
)
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_vllm import (
    Gemma4ResidentMultiS32PBRingAPCForCausalLM,
)


class Gemma4ResidentMultiS32PBRingAPCV3ForCausalLM(Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM):
    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, req_ids=None, **kwargs):
        """v1's decode_forward without its refusal of spec_mode: the rows' request keys for `_layout` while the launch
        runs, then the MRO past v1 (v1's body is exactly these lines plus the refusal)."""
        self._decode_keys = None if req_ids is None else [self._req_keys.get(r) for r in req_ids]
        try:
            return super(Gemma4ResidentMultiS32PBRingAPCForCausalLM, self).decode_forward(
                tokens, start_pos, page_table, kv_cache, **kwargs
            )
        finally:
            self._decode_keys = None

    def release_request(self, slot):
        engine = self.slots.plugin.get(int(slot))
        state = self.slots.states.get(engine) if engine is not None else None
        super().release_request(slot)
        if state is not None:
            self.released(state)

    def released(self, state):
        """A request has fully released: its request key, engine slot and tap slot are free again. M7's ring hand-off
        goes here; v3 itself does nothing."""

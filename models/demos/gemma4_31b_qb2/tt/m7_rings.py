# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7's drafter-ring rule on S0's SlotTable (bench/resident/m7/M7_REBASE_v2_ADDENDUM_2.md §1; M7_BUILD_PREREG_v1 C2, C6;
prereg v2 addendum 1 C9'). A ring is a tap slot of S0's table (one drafter ring slot each, `rings` of them, 16):

- admission (S0's at position 0, or S1's past 0: both go through SlotTable.admit): a ring only while the live count
  after the admission is at most `ring_users` (16), lowest free ring first;
- every assignment calls on_assign(ring, state, why) before the ring is used: the class zeroes the ring there
  (DrafterRing.zero_slot, C9'), whether it then seeds it or not;
- release frees the engine slot and the ring (S0's release). The hand-off is separate (hand_off()), so the class runs it
  after S1 has dropped the request key and S0 has freed the slots (C6): while the live count is at most `ring_users`,
  each free ring goes to the earliest-admitted live user without one, cold.
S0's admit drops a tap slot that make_state declines (it pops one, the state says None); here it goes back to the free
list."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import SlotTable


class M7SlotTable(SlotTable):
    def __init__(self, users, rings, *, on_assign, ring_users=16):
        super().__init__(users, rings)
        self.ring_users, self.on_assign = ring_users, on_assign
        self.order, self.seq = {}, 0  # key -> admission sequence number

    def admit(self, key, plugin_slot, make_state):
        if key in self.by_key:
            raise ValueError(f"A prefill from position 0 names block {key}, which engine slot {self.by_key[key]} holds")
        if not self.free:
            raise RuntimeError(f"No free engine slot for block {key}: {self.users} slots, owners {self.by_key}")
        slot = self.free.pop(0)
        ring = self.free_taps.pop(0) if self.free_taps and len(self.by_key) + 1 <= self.ring_users else None
        state = make_state(key, slot, ring)
        if ring is not None and state.tap_slot is None:
            self.free_taps = sorted(self.free_taps + [ring])
        self.by_key[key], self.states[slot] = slot, state
        self.bind(plugin_slot, slot)
        self.order[key], self.seq = self.seq, self.seq + 1
        if state.tap_slot is not None:
            self.on_assign(state.tap_slot, state, "admit")
        return state

    def release(self, plugin_slot):
        state = super().release(plugin_slot)
        if state is not None:
            self.order.pop(state.key, None)
        return state

    def hand_off(self):
        """free rings to ring-less live users, earliest admission first, while at most ring_users are live"""
        given = []
        if len(self.by_key) > self.ring_users:
            return given
        waiting = sorted((s for s in self.states.values() if s.tap_slot is None), key=lambda s: self.order[s.key])
        for state in waiting:
            if not self.free_taps:
                break
            state.tap_slot = self.free_taps.pop(0)
            self.on_assign(state.tap_slot, state, "hand_off")
            given.append((state.key, state.tap_slot))
        return given

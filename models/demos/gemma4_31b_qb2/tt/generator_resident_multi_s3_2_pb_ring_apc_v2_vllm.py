# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""S1 `batch` class v2 (update 5; bench/resident/s1prefix/S1_B4_ROOT_CAUSE.md): generator_resident_multi_s3_2_pb_ring_apc_
vllm.Gemma4ResidentMultiS32PBRingAPCForCausalLM (unchanged) with its prefill's binding order fixed.

The plugin assigns prefill state slots per step, preferring each request's row (vllm_tt_plugin model_runner.
_alloc_prefill_state_slots), so a continuing chunk can arrive with another plugin slot than its last one, and a new
request in the same step can get the slot the continuation held. S1 (generator_resident_multi_s1_vllm.
_rebind_continuations) rebinds every continuation first, before S0's loop admits the step's new requests. v1 admits a
request starting past 0 (a restore) in its own loop BEFORE that rebind, so a restore given the plugin slot a
continuation just left found it still bound to the continuation's live engine slot and raised (P1-B4, CHURN, 18:57:21:
"Plugin slot 9 is bound to engine slot 2, not 3"). v2 rebinds the step's continuations (rows whose request already holds
an engine slot) before v1's prefill runs, so the restore admissions see the step's own binding, as start-0 admissions
do under S1. The rebinding is S1's rule unchanged: one permutation of the continuations' plugin slots, refusing a slot
another live engine slot holds."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_vllm import (
    Gemma4ResidentMultiS32PBRingAPCForCausalLM,
)


class Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM(Gemma4ResidentMultiS32PBRingAPCForCausalLM):
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, req_ids=None, start_pos=None,
                        empty_slots=None, **kwargs):
        if req_ids is not None and empty_slots is not None:
            n = len(prompt_lens)
            self._rebind_rows([self._req_keys.get(r) for r in list(req_ids)[:n]],
                              [int(s) for s in list(empty_slots)[:n]])
        return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, req_ids=req_ids,
                                       start_pos=start_pos, empty_slots=empty_slots, **kwargs)

    def _rebind_rows(self, keys, plugin_slots):
        """S1's continuation rebind (v1 _rebind_continuations) over the rows whose request already holds an engine
        slot; rows of requests new to the adapter (no key yet, or no engine slot) are left to the admissions."""
        moving = {}
        for key, plugin_slot in zip(keys, plugin_slots):
            slot = self.slots.by_key.get(key) if key is not None else None
            if slot is not None:
                moving[plugin_slot] = slot
        if not moving or all(self.slots.plugin.get(p) == s for p, s in moving.items()):
            return
        engine = set(moving.values())
        self.slots.plugin = {p: s for p, s in self.slots.plugin.items() if s not in engine}
        for plugin_slot, slot in moving.items():
            other = self.slots.plugin.get(plugin_slot)
            if other is not None and other in self.slots.states:
                raise RuntimeError(f"Plugin slot {plugin_slot} is bound to engine slot {other}, not {slot}: "
                                   f"{self.slots.plugin}")
            self.slots.plugin[plugin_slot] = slot
        self.counters["prefill_rebinds"] += 1

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`single-user` PB, review D13 (DRAM_PREFILL_FIX_REVIEW.md e9d3ff16; bench/packaging/DRAM_PREFILL_FIX_PLAN_v2_ADDENDUM_4.md):
Gemma4ResidentSpecPBForCausalLM (generator_resident_spec_vllm_pb.py, frozen for P3 hold 1) whose PB guard fire also
runs the L1 guard.

With GEMMA4_PB_GUARD=retire (the production default) a guard fire means a prefill built a program under a live trace
on a key the adapter did not treat as novel, so its own `if novel: self._l1_guard(...)` after the generator call is
skipped. A persistent ttnn L1 buffer that program allocated could then sit inside the resident engine's CB range
unchecked. The fire handler therefore also runs the L1 guard (which rebuilds the engine below a lowered floor and
releases the sampling trace, as after a novel prefill). B32 has no L1 guard and is unchanged.
"""

from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb import Gemma4ResidentSpecPBForCausalLM


class Gemma4ResidentSpecPB2ForCausalLM(Gemma4ResidentSpecPBForCausalLM):
    def _pb_release_traces(self):
        self._release_sampling_trace()
        self._release_draft_trace()
        self._l1_guard("pb guard fire")

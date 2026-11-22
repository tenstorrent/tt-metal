# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7 batch class v5.6, the branch-(B) class revision (LEAD_DECISIONS_M7 M270, M273, M275;
bench/resident/m7/hsplit_gates/HSPLIT_GATES_PREREG.md §2, §5): class v5.5 (generator_resident_multi_m7_v5_5_vllm.py,
package ac09673a, critic GO 1a646b3d / M258) with its drafter built from tt/m7_device_v6.py (v5 with D v10: the heads'
O and sem_o delivered as NoC1 unicasts, R4; v10 = v8 bitwise, gate G1) instead of tt/m7_device_v5.py. A subclass of
v5.5 overriding build_drafter only (v5.5's body, the import changed). v5.5's sync sites are unchanged and are v5.5's
code (M273: they close real cross-chip hazards; T2' showed they do not cure (B)): m7_device_v6.py:183 (v5's l.178,
set_launch), generator_resident_multi_m7_v5_5_vllm.py:75 / 78 / 91 (marks l.97 / l.102). Ingest v5 is unchanged.
Every other attribute, method and module-level rule is v5.5's and v5.3's, read from their modules."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_m7_v5_5_vllm import Gemma4ResidentMultiM7V55ForCausalLM


class Gemma4ResidentMultiM7V56ForCausalLM(Gemma4ResidentMultiM7V55ForCausalLM):
    def build_drafter(self):
        """M7Device (tt/m7_device_v6.py: v5 with D v10) after the engine, before any trace (I3): ring, ingest v5 (staging
        max(1, P), never written while the seed is held), D v10 per speculating band's block with RoPE aliased to the
        engine's sliding pair. No band speculates: nothing is built. (v5.5's body, the import changed.)"""
        if not self.draft_blocks:
            return None
        from models.demos.gemma4_31b_qb2.tt.m7_device_v6 import M7Device

        setup = self.generator.model.shared_setup
        rope = (setup[("rope", "sliding_attention")]["decode_cos"], setup[("rope", "sliding_attention")]["decode_sin"])
        device = M7Device.build(self.mesh, self.engine, blocks=self.draft_blocks,
                                staging_slots=max(1, self.serving["seed"]["P"]), rings=self.rings, rope=rope)
        return self._drafter_view(device)

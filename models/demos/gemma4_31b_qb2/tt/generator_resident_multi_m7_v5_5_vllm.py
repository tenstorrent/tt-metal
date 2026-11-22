# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7 batch class v5.5 (LEAD_DECISIONS_M7 M251, M254, M255; prereg v3 addendum 15.2 F-ALL; m5-kernels' placement review
m71/M7_DEVICE_V4_REVIEW.md b7e3d225): class v5.3 (generator_resident_multi_m7_v5_3_vllm.py 5d9990cf, frozen, critic +
verifier GO) with a mesh-wide finish before every resident fabric launch whose predecessor differs, ttnn ops included.
A subclass of v5.3 (class v5.4, 871c63a1, is retired: it lacked R1 / R2) overriding six methods and adding one class
attribute and one module-level helper (synced_d), nothing else:
  - build_drafter: v5.3's body with tt/m7_device_v5.py instead of tt/m7_device_v3.py (R1 + the G-T v3 hang's
    transition: m7_device_v5's set_launch syncs the mesh unconditionally before every D launch, so ingest -> D and
    verify -> D in a step with no ingest are both synced);
  - _launch: before v5.3's, self._sync() when the predecessor of this engine launch is
      (a) an ingest still in flight: propose_draft_tokens flushes the queue (v5.3 l.932) after every verify, and on a
          step that drafts nothing (no job: a penalized co-tenant's forced ordinary steps, placeholders, declines) no D
          launch's sync follows it; S0's drain (_drain_taps -> v5.3 l.770-772, after an ordinary launch) likewise
          (prereg 15.2 N1; found by this class's F-ALL spy: 58 unsynced ingest -> engine transitions on v5.3);
          counted in counters["m7_engine_after_ingest_syncs"];
      (b) a ttnn launch with no mesh sync after it (prereg 15.2 N2 and the prefill row): the sampling trace (S0's
          _replay_sampling after a sampled step's launch; TTSampling all-gathers over the fabric) or a prefill (its
          ttnn ops, its first-token sample included, all run inside prefill_forward), when no ingest or D launch
          followed it (the spy: a sampled user while the band is PLAIN, taps off: 28 unsynced sampling -> engine on
          v5.3); counted in counters["m7_engine_after_sampling_syncs"] / ["m7_engine_after_prefill_syncs"] by the mark's
          source. A prefill whose ring zero (zero_slot, synced) followed it is synced once more here (one finish per
          prefill): only ingest rows or a D launch clear the mark;
    a step that drafts finds both already finished by the D launch's sync, so it adds nothing there;
  - _flush_ingest: v5.3's, then (R2) when the call is the early upkeep (`upto` given: v5.3 _launch l.612-613) and it
    launched rows, self._sync() before the engine launch that follows (v5.3 _launch l.619); counted in
    counters["m7_early_upkeep_syncs"]. In v5.3's flow the early upkeep finds nothing to flush (_propose, l.967, is the
    queue's only appender and propose_draft_tokens flushes the whole queue, l.932, before it returns): R2 holds the
    rule for a host that defers that flush. An ingest's set_rows ends with a mesh sync (d_ingest_v5.py:567), so a
    flush that launched rows clears (b)'s mark;
  - _replay_sampling, prefill_forward: v5.3's, then (b)'s mark: m7_ttnn_mark = ("sampling" | "prefill", synced_d), the
    drafter's synced-D count at that point (0 with no drafter); a D launch since (m7_device_v5 syncs first) makes it
    stale;
  - released: v5.3's, then, when the last live request is released (the mesh idle: a wave or block ended), one INFO
    line "M7 drafter syncs at idle: {...}" with m7_device_v5's sync_counts and the four class sync counts above (M255;
    the critic's da7886d5 N3: the per-block D totals the soak / T2' reports need, which the per-1,000 line cannot
    give; the sampling count is where a device leg shows N2 occurring).
Every other attribute, method and module-level rule (the K1 loader, the bands, the switch, the seed hold, the release
line) is v5.3's, read from v5.3's module, so a later v5.3 revision at the same path (the X6 release revision: only
l.251's constant set) is the base this class runs on.
"""

from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_m7_v5_3_vllm import Gemma4ResidentMultiM7V53ForCausalLM


def synced_d(model):
    """the drafter's count of D launches after a mesh sync (0 with no drafter)"""
    return model.drafter.sync_counts["d_after_mesh_sync"] if model.drafter is not None else 0


class Gemma4ResidentMultiM7V55ForCausalLM(Gemma4ResidentMultiM7V53ForCausalLM):
    m7_ttnn_mark = None  # (b): ("sampling" | "prefill", synced_d) of the last ttnn launch no mesh sync has followed

    def build_drafter(self):
        """M7Device (tt/m7_device_v5.py: v3 + an unconditional mesh sync before every D launch) after the engine,
        before any trace (I3): ring, ingest v5 (staging max(1, P), never written while the seed is held), D v8 per
        speculating band's block with RoPE aliased to the engine's sliding pair. No band speculates: nothing is built.
        (v5.3's body, the import changed.)"""
        if not self.draft_blocks:
            return None
        from models.demos.gemma4_31b_qb2.tt.m7_device_v5 import M7Device

        setup = self.generator.model.shared_setup
        rope = (setup[("rope", "sliding_attention")]["decode_cos"], setup[("rope", "sliding_attention")]["decode_sin"])
        device = M7Device.build(self.mesh, self.engine, blocks=self.draft_blocks,
                                staging_slots=max(1, self.serving["seed"]["P"]), rings=self.rings, rope=rope)
        return self._drafter_view(device)

    def _launch(self, R, M=1):
        """v5.3's launch; first, the mesh finished when the predecessor is (a) an ingest in flight or (b) a ttnn launch
        no sync followed (module docstring)"""
        if self.drafter is not None and "ingest" in self.drafter.in_flight:
            self._sync()
            self.counters["m7_engine_after_ingest_syncs"] += 1
        elif self.m7_ttnn_mark is not None and self.m7_ttnn_mark[1] == synced_d(self):
            self._sync()
            self.counters[f"m7_engine_after_{self.m7_ttnn_mark[0]}_syncs"] += 1
        self.m7_ttnn_mark = None
        super()._launch(R, M)

    def _flush_ingest(self, upto=None):
        """v5.3's flush; R2: the early upkeep's flush (upto given) that launched rows syncs the mesh before the engine
        launch that follows it (no engine launch behind an in-flight ingest); launched rows clear (b)'s mark"""
        rows = self.counters["m7_ingest_rows"]
        super()._flush_ingest(upto=upto)
        if self.counters["m7_ingest_rows"] != rows:
            self.m7_ttnn_mark = None
            if upto is not None:
                self._sync()
                self.counters["m7_early_upkeep_syncs"] += 1

    def _replay_sampling(self):
        """S0's replay, then (b)'s mark"""
        super()._replay_sampling()
        self.m7_ttnn_mark = ("sampling", synced_d(self))

    def prefill_forward(self, *args, **kwargs):
        """v5.3's prefill, then (b)'s mark"""
        out = super().prefill_forward(*args, **kwargs)
        self.m7_ttnn_mark = ("prefill", synced_d(self))
        return out

    def released(self, state):
        """v5.3's; then, the mesh idle (no live request left), the drafter's sync counts (M255)"""
        super().released(state)
        if not self.slots.states and self.drafter is not None:
            syncs = ("early_upkeep_syncs", "engine_after_ingest_syncs", "engine_after_sampling_syncs",
                     "engine_after_prefill_syncs")
            logger.info("M7 drafter syncs at idle: {}",
                        {**self.drafter.sync_counts, **{k: self.counters[f"m7_{k}"] for k in syncs}})

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`single-user` PB3 (bench/packaging/dram_fix/PB3_DESIGN.md; m5-verifier's PB3_DESIGN_REVIEW.md: GO with five build
items; LEAD_DECISIONS_PKG P26.1): Gemma4ResidentSpecPB2ForCausalLM (generator_resident_spec_vllm_pb2.py, unchanged)
plus the drafter seed path made trace-safe. The SU growth probe (SU_GROWTH_PROBE_RESULTS.md) found that PB2's whole
program-cache growth outside the PB window is the seed (DFlashDrafter.seed through _enter: dflash.py:640 _ring_order,
keyed by k0 = start % RING, and :562 _valid_rows, keyed by (valid // 32, valid % 32 > 0)) and the first tap commit,
built while the sampling or draft trace is live. PB3:
- warmup_model_prefill: after PB2's warmups and before any trace capture, every seed program class once, on the serving
  seed's own input (taps.read_window(0): the persistent window buffers _enter passes): seed_warm_calls() = 72 ring
  classes (start 32 k, valid WINDOW_ROWS) and 129 mask classes (start 0, valid 32 full + has_r < WINDOW_ROWS), then the
  tap commit; a device synchronize, then the drafter ring and request state reset (slot_pos -1: the warm ring data
  stays masked, as after the shipped warmup_tap_seed). Bound: 213 ring + 257 mask + 6 commit = 476 entries (review
  item 1).
- a seed-path guard around _enter and the tap commit: program-cache growth while the sampling or draft trace (or a
  generator trace) is live logs "PB guard (seed): ..." at ERROR and releases the sampling and draft traces (review
  item 2: no L1 guard here, since an engine rebuild mid-decode would leave the feed stale; seed builds allocate no
  persistent L1), or raises PBGuardError with GEMMA4_PB_GUARD=raise. _draft re-captures the draft trace and the next
  sampled launch the sampling trace.
PB2's prefill (bucket-key novelty, the PB guard, the L1 guard on its fire) is unchanged; no exact-signature release.
"""

import time

from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt import resident_taps
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb2 import Gemma4ResidentSpecPB2ForCausalLM
from models.demos.gemma4_31b_qb2.tt.prefill_paged import PBGuardError


def seed_warm_calls(ring, window_rows, tile=resident_taps.TILE):
    """The (start, valid) seed calls that build every _ring_order and _valid_rows program class once: in serving,
    start = 32 * first (resident_taps.window), so k0 = start % ring takes ring // tile values, and valid takes
    1..window_rows; the two op groups are independent, so each is warmed on its own."""
    ring_calls = [(tile * k, window_rows) for k in range(ring // tile)]
    mask_calls = [(0, tile * full + r) for full in range(window_rows // tile) for r in (0, 1)
                  if 1 <= tile * full + r < window_rows]
    return ring_calls, mask_calls


def seed_class(start, valid, ring, window_rows, tile=resident_taps.TILE):
    """The program class a serving seed (start, valid) uses: (k0, full, has_r); full / has_r are None when valid
    covers the window (no mask)."""
    if valid >= window_rows:
        return start % ring, None, None
    return start % ring, valid // tile, valid % tile > 0


class Gemma4ResidentSpecPB3ForCausalLM(Gemma4ResidentSpecPB2ForCausalLM):
    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None, tap_layers=()):
        super().__init__(generator, max_batch_size, max_seq_len, vllm_config=vllm_config, tap_layers=tap_layers)
        self.pb3_seed_warm = None

    # ------------------------------------------------------------------ the seed path, guarded

    def _pb3_live(self):
        return self.sampling_trace is not None or self.draft_trace is not None or bool(self.generator.traces)

    def _pb3_guarded(self, where, fn, *args, **kwargs):
        live = self._pb3_live()
        entries = self.mesh.num_program_cache_entries()
        out = fn(*args, **kwargs)
        grew = self.mesh.num_program_cache_entries() - entries
        if grew > 0 and live:
            message = (f"program cache grew by {grew} during {where} with a live trace (sampling "
                       f"{self.sampling_trace is not None}, draft {self.draft_trace is not None}); a seed-path "
                       f"program escaped the PB3 pre-warm")
            self.generator.pb_taints.append(message)
            if self.generator.pb_guard_mode == "raise":
                raise PBGuardError(message)
            logger.error("PB guard (seed): {}; releasing the sampling and draft traces", message)
            self._release_sampling_trace()
            self._release_draft_trace()
        return out

    def _pb3_seed_enter(self, position):
        return super()._enter(position)

    def _enter(self, position):
        return self._pb3_guarded("the drafter seed", self._pb3_seed_enter, position)

    def _pb3_wrap_commit(self):
        commit = self.taps.commit

        def guarded_commit(drafter_taps, rows):
            return self._pb3_guarded("the tap commit", commit, drafter_taps, rows)

        self.taps.commit = guarded_commit

    # ------------------------------------------------------------------ warmup

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        first = self.pb3_seed_warm is None
        super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        if first and self.drafter is not None and self.taps is not None:
            self._pb3_wrap_commit()
            self._pb3_warm_seed()

    def _pb3_warm_seed(self):
        if self._pb3_live():
            raise RuntimeError("The PB3 seed warmup must run before any trace capture")
        start_s = time.perf_counter()
        before = self.mesh.num_program_cache_entries()
        windows = self.taps.read_window(0)
        taps = {layer: windows[i] for i, layer in enumerate(self.tap_layers)}
        ring_calls, mask_calls = seed_warm_calls(self.drafter.RING, resident_taps.WINDOW_ROWS)
        for start, valid in ring_calls + mask_calls:
            self.drafter.seed(taps, start, valid)
        self.taps.commit(self.drafter.taps, 1)
        ttnn.synchronize_device(self.mesh)
        self.drafter.reset()
        self._reset_request()
        after = self.mesh.num_program_cache_entries()
        self.pb3_seed_warm = {"ring": len(ring_calls), "mask": len(mask_calls), "before": before, "after": after,
                              "s": round(time.perf_counter() - start_s, 1)}
        logger.info("PB3 seed warmup: {} ring classes, {} mask classes, commit; program cache {} -> {} (+{}); {:.1f} s",
                    len(ring_calls), len(mask_calls), before, after, after - before, self.pb3_seed_warm["s"])

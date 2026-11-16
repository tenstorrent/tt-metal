# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: option S over R in {1, 8, 32} on resident_m5_7 (S3). The S2b adapter (generator_resident_multi_s2b_vllm,
unchanged: R by live rows, hysteresis on the way down, the XCHIP sync + feed reload on every switch, bf8_partials) plus:

- ROWS = (1, 8, 32): one live user runs the R = 1 program, two to eight the multi-user R = 8 program (resident_m5_7's
  D16: K2a at R = 8 with one-row user blocks, ResidentEngine(..., k2a_r8=True)), nine or more the R = 32 one.
  bench/resident/s1/BATCH_PROFILE_SWAP_PLAN.md, LEAD_DECISIONS_S1 S17.2.
- The decode warmup refuses an engine whose R = 8 program serves fewer than 8 users (a package without D16, or k2a_r8
  not taken), instead of serving c = 2..8 at R = 32 without saying so.
- Capacity: the smallest capacity_for over ROWS (R = 8 and R = 32 are both K2a programs: 10,240 until K2b).
- The live L1 floor (bench/resident/m5_3/LEAD_DECISIONS.md D35): after the decode warmup, the lowest ttnn L1 buffer must
  sit at or above GEMMA4_RESIDENT_L1_FLOOR_MIN (default 1,483,712: ttnn's prefill linear CB region ends at 1,479,616,
  plus 4 KB), or the server refuses to start. A later guard below it logs a warning and counts l1_floor_below_min.
- Every program switch logs its direction at INFO and counts program_switch_<from>_to_<to>.
- A DRAM line every GEMMA4_RESIDENT_DRAM_LOG_EVERY engine launches (default 16,384; 0: never), so a long run shows
  DRAM free per chip over time, not only after novel prefill shapes (a soak's leak check reads it).
- GEMMA4_RESIDENT_R_IDLE_RESET=1 drops the hysteresis state when the last live request is released, so a request that
  starts on an idle server runs at the R its own load picks from its first step. The default (0) keeps S2b's
  behaviour: the R of the last busy period holds for up to GEMMA4_RESIDENT_R_DOWN_STEPS decodes.
"""

import os

from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s2b_vllm import Gemma4ResidentMultiS2bForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import DEFAULT_L1_TOP, RAW_WEIGHT_CACHE

MULTI_R8 = 8  # the multi-user R = 8 program (D16): R // M users at M = 1
PREFILL_CB_END = 1_479_616  # ttnn's prefill linear static CB region end in the served process (D35)
L1_FLOOR_MIN = PREFILL_CB_END + 4096


class Gemma4ResidentMultiS3ForCausalLM(Gemma4ResidentMultiS2bForCausalLM):
    ROWS = (1, MULTI_R8, MAX_USERS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idle_reset = os.environ.get("GEMMA4_RESIDENT_R_IDLE_RESET", "0") == "1"
        self.l1_floor_min = int(os.environ.get("GEMMA4_RESIDENT_L1_FLOOR_MIN", str(L1_FLOOR_MIN)))
        self.l1_floor_checked = False  # the startup floor check has passed
        self.dram_every = int(os.environ.get("GEMMA4_RESIDENT_DRAM_LOG_EVERY", "16384"))

    # ------------------------------------------------------------------ engine

    def _build_engine(self):
        """S2b's engine (generator_resident_multi_s2b_vllm._build_engine) from resident_m5_7, with k2a_r8."""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m5_7")
        bf8 = os.environ.get("GEMMA4_RESIDENT_BF8_PARTIALS", "1") == "1"
        resident = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.engine")
        setup = self.generator.model.shared_setup
        cfg = resident.EngineConfig(
            weight_cache=os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE),
            rows=self.ROWS,
            max_users=self.max_batch_size,
            capacity=self.max_seq_len,
            tap_layers=self.tap_layers,
            l1_top=int(os.environ.get("GRES_L1_TOP", DEFAULT_L1_TOP)),
            logits=True,
            embedding="resident",
            block=self._engine_block_map(),
        )
        pools = resident.KVPools(
            kv=list(self.cache.kv),
            layer_group=list(self.groups),
            scratch=self._group_scratch(),
            rope={
                kind: (setup[("rope", kind)]["decode_cos"], setup[("rope", kind)]["decode_sin"])
                for kind in ("sliding_attention", "full_attention")
            },
        )
        logger.info("Building the option S engine from {}, rows {}, blocks {}, users {}, bf8_partials {}, k2a_r8 True",
                    package, cfg.rows, cfg.block, cfg.max_users, bf8)
        return resident.ResidentEngine(self.mesh, cfg, pools, bf8_partials=bf8, k2a_r8=True)

    def _capacity(self):
        return min(int(self.engine.capacity_for(R)) for R in self.ROWS)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        """The R = 8 program must serve 8 users; then S1's warmup (its capacity refusal reads _capacity); then the live
        L1 floor check (D35). prefill_tokens is M1's warmed marker (None until the first decode warmup)."""
        first = self.prefill_tokens is None
        if first:
            served = int(self.engine.users_for(MULTI_R8))
            if served < MULTI_R8:
                raise ValueError(f"The engine's R = {MULTI_R8} program serves {served} user(s), not {MULTI_R8}: S3 "
                                 "needs resident_m5_7's multi-user R = 8 (D16, k2a_r8)")
        super().warmup_model_decode(kv_cache, enable_trace, **kwargs)
        if first:
            self._check_l1_floor()

    def _check_l1_floor(self):
        floors = [f for _, f, _ in self.l1_checks if f is not None]
        if not floors:
            logger.info("L1 floor at startup: no ttnn L1 buffer allocated (minimum {})", self.l1_floor_min)
        else:
            low = min(floors)
            logger.info("L1 floor at startup: lowest ttnn L1 buffer {} over {} guards, minimum {}, margin {} B", low,
                        len(floors), self.l1_floor_min, low - self.l1_floor_min)
            if low < self.l1_floor_min:
                raise RuntimeError(f"The lowest ttnn L1 buffer {low} is below GEMMA4_RESIDENT_L1_FLOOR_MIN "
                                   f"{self.l1_floor_min} (ttnn's prefill CB region ends at {PREFILL_CB_END}; "
                                   "LEAD_DECISIONS m5_3 D35)")
        self.l1_floor_checked = True

    def _l1_guard(self, reason, before_rebuild=None):
        super()._l1_guard(reason, before_rebuild=before_rebuild)
        floor = self.l1_checks[-1][1] if self.l1_checks else None
        if self.l1_floor_checked and floor is not None and floor < self.l1_floor_min:
            self.counters["l1_floor_below_min"] += 1
            logger.warning("L1 floor {} after {} is below GEMMA4_RESIDENT_L1_FLOOR_MIN {}", floor, reason,
                           self.l1_floor_min)

    # ------------------------------------------------------------------ R by live rows

    def _launch(self, R, M=1):
        super()._launch(R, M)
        if self.dram_every > 0 and self.engine.launches % self.dram_every == 0:
            self._log_dram(f"{self.engine.launches} launches")

    def _switch_program(self, R, reload=True):
        old = None if self.last_program is None else self.last_program[0]
        super()._switch_program(R, reload=reload)
        self.counters[f"program_switch_{old}_to_{R}"] += 1
        logger.info("Program switch R {} -> {} (switch {})", old, R, self.counters["program_switch_syncs"])

    def release_request(self, slot):
        super().release_request(slot)
        if self.idle_reset and not self.slots.states and self.current_rows is not None:
            self.current_rows, self.down_steps = None, 0
            self.counters["r_idle_resets"] += 1

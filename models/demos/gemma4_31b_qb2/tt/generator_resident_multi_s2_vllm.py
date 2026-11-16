# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: option S at c = 1, R by live rows over {1, 32} (S2). The S1 adapter (generator_resident_multi_s1_vllm,
CP2's, unchanged) plus:

- ROWS = (1, 32) and the S0 launch policy `by_users`: one live user runs the R = 1 program (M1's single-user decode),
  two or more the R = 32 one (K2a). bench/resident/s1/OPTION_S_GAPS.md G1, LEAD_DECISIONS_S1 S10.2.
- Hysteresis (G4): moving up is immediate (R = 1 cannot serve a second user); moving down waits until the live users
  have fit the smaller R for GEMMA4_RESIDENT_R_DOWN_STEPS (default 32) consecutive ordinary decodes, which still run at
  the larger R; the next one runs at the smaller.
- The XCHIP rule (bench/resident/XCHIP_ALTERNATION_AUDIT.md, G3): async decode has no per-launch blocking read, the
  only mesh-wide finish between two engine programs. Every change of (R, M) between launches first synchronizes the
  mesh, then reloads the device feed (G5): the live rows' tokens, positions and slots are read back, checked against the
  host's rows (each live row keeps its slot and sits below the new R) and written again, so the engine's host mirror is
  current and rows at or past the new R are inert.
- The engine: R = 1 builds only M = 1; R = 32 builds GEMMA4_RESIDENT_ENGINE_BLOCKS as in S1.
Load-dependent text (R = 1 and R = 32 differ at near-ties) is accepted for option S (LEAD_DECISIONS_S1 S10.3).
"""

import os

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s1_vllm import Gemma4ResidentMultiS1ForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import DEFAULT_L1_TOP, RAW_WEIGHT_CACHE


class Gemma4ResidentMultiS2ForCausalLM(Gemma4ResidentMultiS1ForCausalLM):
    ROWS = (1, MAX_USERS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.launch_mode = "by_users"
        self.down_after = int(os.environ.get("GEMMA4_RESIDENT_R_DOWN_STEPS", "32"))
        self.current_rows = None  # R of the last ordinary decode
        self.down_steps = 0  # consecutive ordinary decodes whose users fit a smaller R
        self.last_program = None  # (R, M) of the last launch

    # ------------------------------------------------------------------ engine

    def _engine_block_map(self):
        blocks = self._engine_blocks()
        return {R: blocks if R == MAX_USERS else (1,) for R in self.ROWS}

    def _build_engine(self):
        """S1's engine with a block map per R: the single-user programs build M = 1 only."""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m5_4")
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
        logger.info("Building the option S engine from {}, rows {}, blocks {}, users {}", package, cfg.rows, cfg.block,
                    cfg.max_users)
        return resident.ResidentEngine(self.mesh, cfg, pools)

    # ------------------------------------------------------------------ R by live rows

    def _launch_rows(self, users):
        """S0's by_users choice, with hysteresis on the way down."""
        want = super()._launch_rows(users)
        if self.current_rows is None or want >= self.current_rows:
            self.down_steps = 0
            self.current_rows = want
            return want
        self.down_steps += 1
        if self.down_steps > self.down_after:
            self.down_steps = 0
            self.current_rows = want
        return self.current_rows

    def _launch(self, R, M=1):
        if self.last_program is not None and (R, M) != self.last_program:
            self._switch_program(R, reload=M == 1 and self.last_program[1] == 1)
        self.last_program = (R, M)
        super()._launch(R, M)

    def _switch_program(self, R, reload=True):
        """Another engine program next: a mesh-wide finish first (the XCHIP rule), then, between two M = 1 programs
        (whose rows are the decode rows of the front-packed batch), the device feed read back, checked and written
        again for the new R. A verify launch (M > 1) writes its own rows just before it and only needs the finish."""
        import ttnn

        ttnn.synchronize_device(self.mesh)
        self.counters["program_switch_syncs"] += 1
        if not reload:
            return
        read = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)[:MAX_USERS].to(torch.int64)
        tokens = read(self.engine.tokens) & 0xFFFFFFFF
        positions = read(self.engine.positions)
        positions = torch.where(positions >= 1 << 31, positions - (1 << 32), positions)
        slots = read(self.engine.slots)
        live = [r for r, s in enumerate(self.row_slots) if s is not None and self.positions[r] >= 0]
        for r in live:
            if r >= R or int(slots[r]) != self.row_slots[r] or int(positions[r]) < 0:
                raise RuntimeError(f"Program switch to R = {R}: live row {r} (slot {self.row_slots[r]}) is not in the "
                                   f"device feed (slot {int(slots[r])}, position {int(positions[r])})")
        keep = [r in live for r in range(MAX_USERS)]
        self.engine.write_rows([int(s) if k else 0 for s, k in zip(slots, keep)],
                               [int(p) if k else -1 for p, k in zip(positions, keep)],
                               [int(t) if k else 0 for t, k in zip(tokens, keep)])
        self.counters["program_switch_reloads"] += 1
        logger.debug("Program switch to R = {}: live rows {} reloaded", R, live)

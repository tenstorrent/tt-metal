# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: option S on an engine with D6 (S2b). The S2 adapter (generator_resident_multi_s2_vllm, unchanged: R by
live rows over {1, 32}, hysteresis, the XCHIP sync + feed reload on every switch) with the engine built with
`bf8_partials=GEMMA4_RESIDENT_BF8_PARTIALS` (default "1"), as S1b does for S1: resident_m5_6 or later only (D6 acts at
R = 32; the R = 1 program is unaffected). LEAD_DECISIONS_S1 S13: the option S c-sweep runs on S2b.
"""

import os

from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s2_vllm import Gemma4ResidentMultiS2ForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import DEFAULT_L1_TOP, RAW_WEIGHT_CACHE


class Gemma4ResidentMultiS2bForCausalLM(Gemma4ResidentMultiS2ForCausalLM):
    def _build_engine(self):
        """S2's engine (generator_resident_multi_s2_vllm._build_engine), with bf8_partials."""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m5_6")
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
        logger.info("Building the option S engine from {}, rows {}, blocks {}, users {}, bf8_partials {}", package,
                    cfg.rows, cfg.block, cfg.max_users, bf8)
        return resident.ResidentEngine(self.mesh, cfg, pools, bf8_partials=bf8)

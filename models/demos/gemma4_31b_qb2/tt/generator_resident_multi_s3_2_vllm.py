# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S3.1 on resident_m5_9 (K2b: K2a programs to 262,144 positions) for a 262K `batch` (S3.2).
bench/resident/s1/S3X_262K_PLAN.md. The S3.1 adapter (generator_resident_multi_s3_1_vllm, unchanged: rows 1 / 8 / 32,
k2a_r8, BF8 partials, the plain launch only, the D35 floor gate, the idle reset knob) plus:

- The engine package defaults to resident_m5_9 (GEMMA4_RESIDENT_ENGINE_PACKAGE still overrides it). S3.1's capacity
  is the smallest capacity_for over ROWS, so at max_model_len 262,144 every program serves the whole context.
- F_SCHED (m5_9, 0x10 in status word 1: a schedule overflow made the launch run inert, ids 0, no row moved) is named in
  the error. The step already fails on any nonzero word 1 (S0's _check_status, on every decode readback).
- Construction hooks for the bounded-prefill (PB) variant (bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md §1.4, "S3-PB on
  the final S3.x"): initialize_vllm_model builds its ttnn model and prefill generator through the classmethods
  build_model and build_generator, so a PB subclass swaps in its model and generator without copying the construction.
  The bodies are S0's (generator_resident_multi_vllm.initialize_vllm_model), unchanged otherwise.
"""

import os

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_1_vllm import Gemma4ResidentMultiS31ForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import Gemma4ResidentSpecForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import DEFAULT_L1_TOP, RAW_WEIGHT_CACHE, _PrefillGenerator
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model

F_SCHED = 0x10  # resident_m5_9/engine.py:47 (status word 1 only)


class Gemma4ResidentMultiS32ForCausalLM(Gemma4ResidentMultiS31ForCausalLM):
    DEFAULT_PACKAGE = "resident_m5_9"

    # ------------------------------------------------------------------ construction (PB hooks)

    @classmethod
    def build_model(cls, mesh_device, tile_pad):
        """the prefill-only ttnn model (S0's): a PB subclass returns its paged-prefill model here"""
        return Gemma4Model(mesh_device, decode_weights=False, prefill_tile_pad=tile_pad)

    @classmethod
    def build_generator(cls, model, max_seq_len):
        """the prefill generator (S0's): a PB subclass returns its generator here"""
        return _PrefillGenerator(model, max_seq_len=max_seq_len)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        """S0's construction (generator_resident_multi_vllm.initialize_vllm_model) through build_model / build_generator"""
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= MAX_USERS:
            raise ValueError(f"Resident Gemma4 QB2 (multi-user) serves TP4, DP1, 1..{MAX_USERS} requests")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        tile_pad = os.environ.get("GEMMA4_RESIDENT_TILE_PAD", "1") == "1"
        model = cls.build_model(mesh_device, tile_pad)
        return cls(cls.build_generator(model, max_seq_len), max_batch_size, max_seq_len,
                   tap_layers=Gemma4ResidentSpecForCausalLM.drafter_tap_layers())

    def _build_engine(self):
        """S3's engine (generator_resident_multi_s3_vllm._build_engine) with DEFAULT_PACKAGE"""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", self.DEFAULT_PACKAGE)
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
        logger.info("Building the option S engine from {}, rows {}, blocks {}, users {}, capacity {}, bf8_partials {}, "
                    "k2a_r8 True", package, cfg.rows, cfg.block, cfg.max_users, cfg.capacity, bf8)
        return resident.ResidentEngine(self.mesh, cfg, pools, bf8_partials=bf8, k2a_r8=True)

    # ------------------------------------------------------------------ status

    def _check_status(self, status):
        """S0's check (raises on any flag in word 1), with an F_SCHED launch named first"""
        import ttnn

        word1 = int(ttnn.to_torch(status).reshape(-1)[1].item()) & 0xFFFFFFFF
        if word1 & F_SCHED:
            self.counters["f_sched"] += 1
            logger.error("Resident engine F_SCHED (0x{:x}): a K2a schedule overflow made the launch run inert (ids 0, "
                         "no row moved); the step fails", word1)
        super()._check_status(status)

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill ISL sweep (criteria C1/C6): demonstrate that forward_prefill_chunked RUNS at long
context (4K/8K/16K) without the L1 circular-buffer clash that caps single-shot prefill (~4K). 2-layer
model (the L1 clash is per-layer, so 2 layers exercises the same per-layer attention path that overflows
at full depth); logs wall-clock prefill time per ISL + the largest ISL that runs. B=1."""
import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = int(os.environ.get("M4_CHUNK_LAYERS", "2"))  # set 36 for the full-depth TTFT numbers
B = 1
CHUNK = 2048  # query window per iteration (fits L1 like the 4K single-shot); paging block stays 128
BLOCK = 128
ISLS = [int(s) for s in os.environ.get("M4_CHUNK_ISLS", "4096,8192,16384").split(",")]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_chunked_isl(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    rope = cfg.qk_rope_head_dim
    ref, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    ref = ref.float()
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )

    largest = 0
    for S in ISLS:
        torch.manual_seed(0)
        h = torch.randn(B, S, cfg.hidden_size) * 0.02
        cos1, sin1 = ref.model.rotary_emb(h, torch.arange(S)[None])
        hd = _repl(h, mesh_device)
        cosd = _repl(cos1[0].float().reshape(1, 1, S, rope), mesh_device)
        sind = _repl(sin1[0].float().reshape(1, 1, S, rope), mesh_device)
        caches = tt.init_paged_kv_caches(B, S, block_size=BLOCK)
        t0 = time.perf_counter()
        o = tt.forward_prefill_chunked(hd, cosd, sind, caches, chunk=CHUNK, block_size=BLOCK)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        logger.info(
            f"A6 chunked-prefill ISL {S}: RAN in {dt*1000:.0f} ms ({N_LAYERS}-layer, B={B}) -> {dt*1000/N_LAYERS:.0f} ms/layer"
        )
        largest = S
        ttnn.deallocate(o)
        for paged_kv, pt in caches:
            ttnn.deallocate(paged_kv[0])
            ttnn.deallocate(paged_kv[1])
    logger.info(f"A6 chunked-prefill largest ISL that RAN: {largest} (single-shot prefill L1-caps ~4K)")
    assert largest >= max(ISLS), f"chunked prefill did not reach {max(ISLS)}, only {largest}"

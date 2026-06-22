# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Empirical prefill ISL cap + latency probe.

Runs TtMistral4TextModel.forward_prefill at increasing sequence lengths to find where the
single-shot prefill structurally caps (L1 circular-buffer clash / OOM) and to record per-ISL
prefill wall-time. Determines whether chunked prefill is required and at what threshold (criteria
C6). 2-layer model: the L1/structural cap is per-layer, independent of depth.
"""
import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
ISLS = [128, 512, 1024, 2048, 4096]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_prefill_isl(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    rope, hidden = cfg.qk_rope_head_dim, cfg.hidden_size
    B = 1
    results = {}
    for S in ISLS:
        try:
            x = _repl(torch.randn(B, S, hidden) * 0.1, mesh_device)
            cos = _repl(torch.randn(B, 1, S, rope), mesh_device)
            sin = _repl(torch.randn(B, 1, S, rope), mesh_device)
            kv = tt.init_kv_caches(B, max_seq=S + 64)
            t0 = time.time()
            out = tt.forward_prefill(x, cos, sin, kv)
            ttnn.synchronize_device(mesh_device)
            dt = time.time() - t0
            ok = torch.isfinite(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B]).all()
            results[S] = (True, dt, bool(ok))
            logger.info(f"prefill S={S}: OK {dt*1000:.0f}ms ({S/dt:.0f} tok/s) finite={ok}")
        except Exception as e:
            results[S] = (False, None, str(e)[:120])
            logger.warning(f"prefill S={S}: FAILED {str(e)[:120]}")
            break
    ok_isls = [s for s, r in results.items() if r[0]]
    logger.info(f"single-shot prefill max ISL = {max(ok_isls) if ok_isls else 'NONE'}; results={results}")
    assert ok_isls, "prefill failed even at S=128"

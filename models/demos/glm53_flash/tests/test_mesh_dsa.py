# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 bisect: the DSA attention alone on a (2, 4) block (heads over the columns, one user
group per row, replicated latent cache / index), decode step by step vs the HF attention's prefill
output rows (same computation token by token), then HF-cache decode steps.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_dsa.py -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_dsa import LAYER, _hf_attention, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache, decode_position_tensors
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

ROWS, COLS = 2, 4
B = 2
S = 40
N_DECODE = 2


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_mesh_dsa_decode_vs_hf(mesh_device, weights):
    from transformers import DynamicCache

    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_attention(text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    mesh = GLMMesh.from_device(mesh_device)
    tt = TtDSA(mesh_device, cfg, attn, layer_idx=LAYER, mesh=mesh)
    tt_cache = TtLatentCache(mesh_device, cfg, max_batch=B, max_seq_len=128, block_size=32)
    U = ROWS * B
    torch.manual_seed(0)
    x = (torch.randn(U, S + N_DECODE, cfg.hidden_size) * 0.8).to(torch.bfloat16)  # one sequence per user
    caches, y_hf, y_hf_dec = [], [], []
    for u in range(U):
        cache = DynamicCache(config=text)
        with torch.no_grad():
            y, _, _ = hf(x[u : u + 1, :S], attention_mask=torch.ones(1, S, dtype=torch.bool), past_key_values=cache)
            yd = []
            for t in range(N_DECODE):
                ytd, _, _ = hf(
                    x[u : u + 1, S + t : S + t + 1],
                    attention_mask=torch.ones(1, 1, dtype=torch.bool),
                    past_key_values=cache,
                )
                yd.append(ytd)
        y_hf.append(y)
        y_hf_dec.append(torch.cat(yd, dim=1))
        caches.append(cache)

    def step(t: int):
        xin = x[:, t : t + 1].reshape(1, 1, U, -1)  # [1,1,U,4096], row r = users [rB, (r+1)B)
        x_dev = mesh.upload(xin, ttnn.bfloat16, rows=2)
        pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [t] * U, cfg.index_kpool, mesh=mesh)
        logger.info(f"  step {t}: decode_forward ...")
        y = tt.decode_forward(x_dev, tt_cache, pos, pool_ids, ring_slots, [t] * U)
        ttnn.synchronize_device(mesh_device)
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]  # [1,1,B,4096] each
        for r in range(ROWS):
            for c in range(1, COLS):
                assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (t, r, c)
        return torch.cat([shards[r * COLS][0, 0] for r in range(ROWS)], dim=0)  # [U, 4096]

    outs = [[] for _ in range(U)]
    for t in range(S):
        out = step(t)
        for u in range(U):
            outs[u].append(out[u : u + 1])
    per_user = [pcc(y_hf[u][0], torch.cat(outs[u], dim=0)) for u in range(U)]
    logger.info(
        f"DSA layer {LAYER} on {ROWS}x{COLS}: {U} users x {S} tokens through decode: per-user pcc {['%.6f' % p for p in per_user]}"
    )
    assert min(per_user) > 0.995
    for t in range(N_DECODE):
        out = step(S + t)
        ps = [pcc(y_hf_dec[u][0, t : t + 1], out[u : u + 1]) for u in range(U)]
        logger.info(f"  HF-cache decode at position {S + t}: per-user pcc {['%.6f' % p for p in ps]}")
        assert min(ps) > 0.995

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eight user slots: per-user prefill (different lengths, one with a page-table permutation),
then batched decode of all 8 users for 3 steps, per-user PCC vs HF (batch-1 HF caches).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_batch8.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import build_layer, load_config
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

DUMP = Path("bringup/references/glm53_flash/R4097")
B = 8
LENS = [100, 37, 64, 1, 200, 33, 96, 129]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [3, 4])
def test_eight_users_decode_vs_hf(mesh_device, layer_idx):
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf, _ = build_layer(snapshot, text, layer_idx, headers)
    pre = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")
    base = pre["x_in"]  # [1, 4097, 4, 4096] random-token streams
    offs = [0, 300, 600, 900, 1200, 1500, 1800, 2100]
    prompts = [base[:, o : o + L] for o, L in zip(offs, LENS)]
    n_decode = 3
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=B, max_seq_len=256
    )
    if dec.cache is not None:
        bps = dec.cache.blocks_per_seq
        perm = torch.randperm(B * bps, generator=torch.Generator().manual_seed(9)).to(torch.int32).reshape(B, bps)
        dec.cache.set_page_table(perm)

    caches = []
    for u, (L, x) in enumerate(zip(LENS, prompts)):
        cache = DynamicCache(config=text)
        with torch.no_grad():
            y_hf, _ = hf(x, attention_mask=torch.ones(1, L, dtype=torch.bool), past_key_values=cache, use_cache=True)
        y = dec.prefill_forward(host_streams_to_device(x, mesh_device), user=u, seq_len=L)
        p = pcc(y_hf, device_streams_to_host(y, 4, L))
        logger.info(f"layer {layer_idx} user {u} prefill S={L}: pcc {p:.6f}")
        assert p > 0.995
        caches.append(cache)

    for step in range(n_decode):
        toks = [base[:, 3000 + 10 * u + step : 3001 + 10 * u + step] for u in range(B)]
        y_hf = []
        for u in range(B):
            with torch.no_grad():
                yu, _ = hf(
                    toks[u],
                    attention_mask=torch.ones(1, 1, dtype=torch.bool),
                    past_key_values=caches[u],
                    use_cache=True,
                )
            y_hf.append(yu)
        xin = torch.cat(toks, dim=1)  # [1, B, 4, 4096]
        dp = dec.make_decode_positions([L + step for L in LENS])
        yd = dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=B), positions=dp)
        out = device_streams_to_host(yd, 4, B)
        worst = 1.0
        for u in range(B):
            p_d = pcc(y_hf[u], out[:, u : u + 1])
            worst = min(worst, p_d)
            assert p_d > 0.995, f"user {u} step {step}: pcc {p_d}"
        logger.info(f"  decode step {step}: worst per-user pcc over {B} users {worst:.6f}")

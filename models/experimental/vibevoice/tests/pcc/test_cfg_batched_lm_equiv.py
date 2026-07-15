# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CFG-batched LM: one row-stacked batch-2 decode forward == two separate batch-1 decodes.

Runs N identical decode steps on (a) two single-stream caches via the traced decode path and
(b) a combined [2,n_kv,maxS,hd] cache via forward_decode_batched2, then compares the final
step's hidden (both rows) + logits (positive row).  Numerically exact -> PCC ~1.0."""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import build_tt_lm


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_cfg_batched_lm_equiv(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder
    H = cfg.hidden_size
    lm = build_tt_lm(lm_state, mesh_device, cfg)

    pos_cache = lm.alloc_kv_cache(64)
    neg_cache = lm.alloc_kv_cache(64)
    comb = lm.alloc_kv_cache_batched2(64)

    def to4(t):
        return ttnn.from_torch(
            t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def pos(vals):
        return ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), device=mesh_device, dtype=ttnn.int32)

    STEPS = 6
    hp = hn = lp = h2 = lb = None
    for step in range(STEPS):
        ep = torch.randn(1, 1, 1, H)
        en = torch.randn(1, 1, 1, H)
        p_pos, p_neg = step, step  # same positions (different-position case validated separately)
        # separate single-stream decodes
        lp, hp = lm.forward_decode_traced_embeds_dev_rope(
            to4(ep), pos([p_pos]), pos_cache, return_last_hidden=True, need_logits=True
        )
        _, hn = lm.forward_decode_traced_embeds_dev_rope(
            to4(en), pos([p_neg]), neg_cache, return_last_hidden=True, need_logits=False
        )
        # batched: row0 = pos embed, row1 = neg embed
        e2 = ttnn.concat([to4(ep), to4(en)], dim=2)  # [1,1,2,H]
        cos_rows, sin_rows = lm._rope_rows_from_pos_int2(p_pos, p_neg)
        lb, h2 = lm.forward_decode_batched2(e2, cos_rows, sin_rows, pos([p_pos, p_neg]), comb)

    hp_t = ttnn.to_torch(hp).float().reshape(1, 1, 1, H)
    hn_t = ttnn.to_torch(hn).float().reshape(1, 1, 1, H)
    h2_t = ttnn.to_torch(h2).float().reshape(1, 1, 2, H)
    lp_t = ttnn.to_torch(lp).float().reshape(1, 1, 1, -1)[..., -1, :]
    lb_t = ttnn.to_torch(lb).float().reshape(1, 1, 1, -1)[..., -1, :]

    ok_p, pcc_p = comp_pcc(hp_t, h2_t[:, :, 0:1, :], pcc=0.99)
    ok_n, pcc_n = comp_pcc(hn_t, h2_t[:, :, 1:2, :], pcc=0.99)
    ok_l, pcc_l = comp_pcc(lp_t, lb_t, pcc=0.99)
    print(f"batched-vs-separate  pos_hidden={pcc_p}  neg_hidden={pcc_n}  pos_logits={pcc_l}")
    tm = (lp_t.argmax(-1) == lb_t.argmax(-1)).float().mean().item()
    print(f"argmax token match (pos): {tm}")
    assert ok_p and ok_n and ok_l, f"CFG-batched decode mismatch: {pcc_p}, {pcc_n}, {pcc_l}"

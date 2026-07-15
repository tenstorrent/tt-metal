# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate that prefill + batched decode on a COMBINED [2,n_kv,maxS,hd] cache (row 0 = pos)
matches a normal single-stream [1,...] prefill + decode.  This is the last integration risk
for wiring the CFG-batched LM into generate() (prefill must populate combined row 0)."""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import build_tt_lm


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_cfg_batched_prefill_equiv(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder
    H = cfg.hidden_size
    lm = build_tt_lm(lm_state, mesh_device, cfg)
    S = 32  # prefill length

    def emb(t):
        return ttnn.from_torch(
            t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def pos(vals):
        return ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), device=mesh_device, dtype=ttnn.int32)

    pre = torch.randn(1, 1, S, H)

    # reference: single-stream [1,...] cache prefill + one decode step
    ref_cache = lm.alloc_kv_cache(S + 16)
    lm.prefill_embeds(emb(pre), kv_cache=ref_cache, return_last_hidden=True)
    d0 = torch.randn(1, 1, 1, H)
    lp_ref, hp_ref = lm.forward_decode_traced_embeds_dev_rope(
        emb(d0), pos([S]), ref_cache, return_last_hidden=True, need_logits=True
    )

    # combined [2,...] cache: prefill writes row 0, then batched decode (row0=pos, row1=dummy scratch)
    comb = lm.alloc_kv_cache_batched2(S + 16)
    lm.prefill_embeds(emb(pre), kv_cache=comb, return_last_hidden=True)
    dummy = torch.randn(1, 1, 1, H)
    e2 = ttnn.concat([emb(d0), emb(dummy)], dim=2)
    cos_rows, sin_rows = lm._rope_rows_from_pos_int2(S, comb.max_seq - 32)  # neg row = scratch pos
    lp_b, h2_b = lm.forward_decode_batched2(e2, cos_rows, sin_rows, pos([S, comb.max_seq - 32]), comb)

    hp_ref_t = ttnn.to_torch(hp_ref).float().reshape(1, 1, 1, H)
    h2_t = ttnn.to_torch(h2_b).float().reshape(1, 1, 2, H)
    lp_ref_t = ttnn.to_torch(lp_ref).float().reshape(1, 1, 1, -1)[..., -1, :]
    lp_b_t = ttnn.to_torch(lp_b).float().reshape(1, 1, 1, -1)[..., -1, :]

    ok_h, pcc_h = comp_pcc(hp_ref_t, h2_t[:, :, 0:1, :], pcc=0.99)
    ok_l, pcc_l = comp_pcc(lp_ref_t, lp_b_t, pcc=0.99)
    tm = (lp_ref_t.argmax(-1) == lp_b_t.argmax(-1)).float().mean().item()
    print(f"prefill-into-combined-row0: pos_hidden={pcc_h}  pos_logits={pcc_l}  token_match={tm}")
    assert ok_h and ok_l, f"combined-cache prefill mismatch: hidden={pcc_h} logits={pcc_l}"

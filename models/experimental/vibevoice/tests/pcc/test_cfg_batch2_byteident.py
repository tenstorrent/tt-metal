# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CFG batch-2 fused LM decode — REAL-WEIGHTS byte-identity gate.

Proves the assembled 28-layer B=2 forward (forward_decode_traced_embeds_b2) is BYTE-IDENTICAL,
per row, to two separate B=1 forwards (forward_decode_traced_embeds):
    hidden_b2[0] == pos-LM B=1 last_hidden   (row0: input+pos+kv0)
    hidden_b2[1] == neg-LM B=1 last_hidden   (row1: input+pos+kv1)
    logits0      == pos-LM B=1 constrained lm_head logits

maxabsdiff == 0 on all three ⇒ the CFG batch-2 fusion is math-preserving (Tier-0), i.e.
long-form-safe by construction (no full 100-min render needed to accept it).  This closes the
gap the isolated op probes (cfg_batch2_byteident_probe.py / cfg_batch2_sdpa_byteident_probe.py)
left: that the WIRING of the assembled forward routes each row's math correctly.

Run: pytest models/experimental/vibevoice/tests/pcc/test_cfg_batch2_byteident.py -s
"""
import numpy as np
import pytest
import torch
import ttnn

from models.experimental.vibevoice.tests.pcc.lm_pcc_common import build_tt_lm
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import KVCache


def _clone_kv(kv: KVCache) -> KVCache:
    return KVCache(
        keys=[ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in kv.keys],
        values=[ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in kv.values],
        max_seq=kv.max_seq,
    )


def _maxdiff(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.abs(a - b).max())


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_cfg_batch2_forward_byteident(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder
    hd = cfg.head_dim
    H = cfg.hidden_size

    lm = build_tt_lm(lm_state, mesh_device, cfg)

    # Populate two independent caches (pos + neg) with real KV via a short prefill each, so the
    # decode step reads a realistic (non-zero) prefix.  Different lengths mirror the real frame
    # (pos carries the text prefill; neg is a short bootstrap).
    P_POS, P_NEG = 40, 8
    ids_pos = torch.randint(0, cfg.vocab_size, (1, P_POS), dtype=torch.long)
    ids_neg = torch.randint(0, cfg.vocab_size, (1, P_NEG), dtype=torch.long)
    kv_pos = lm.alloc_kv_cache(P_POS + 16)
    kv_neg = lm.alloc_kv_cache(P_NEG + 16)
    lm.prefill(ids_pos, kv_cache=kv_pos)
    lm.prefill(ids_neg, kv_cache=kv_neg)

    # Independent byte-copies for the two runners (paged_update_cache mutates the cache).
    kv_pos_ref, kv_neg_ref = _clone_kv(kv_pos), _clone_kv(kv_neg)
    kv_pos_b2, kv_neg_b2 = _clone_kv(kv_pos), _clone_kv(kv_neg)

    # Decode inputs: row0 = pos input @ pos position, row1 = neg input @ neg position.
    emb_pos = lm._embed(torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long))  # [1,1,1,H]
    emb_neg = lm._embed(torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long))
    pos_pos, pos_neg = P_POS, P_NEG

    def _pos_tt(p):
        return ttnn.as_tensor(
            torch.tensor([p], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rope_row():
        r = torch.randn(1, 1, 1, hd, dtype=torch.float32)
        return ttnn.as_tensor(
            r,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Fixed rope rows per row (values are arbitrary but IDENTICAL between B=1 and B=2 — the test
    # is op-graph byte-identity, so only the routing matters, not RoPE correctness).
    cos0, sin0, cos1, sin1 = _rope_row(), _rope_row(), _rope_row(), _rope_row()
    cur0, cur1 = _pos_tt(pos_pos), _pos_tt(pos_neg)

    # Constrained lm_head subset (as the deployed pos-LM uses).
    valid = sorted([151643, 151644, 151645, 151646, 151647][: min(5, cfg.vocab_size)])
    lm_head_valid = lm.build_lm_head_subset(valid)

    # ── B=1 references ──
    logits_pos_b1, hidden_pos_b1 = lm.forward_decode_traced_embeds(
        emb_pos,
        cos0,
        sin0,
        _pos_tt(pos_pos),
        kv_pos_ref,
        return_last_hidden=True,
        lm_head_w=lm_head_valid,
    )
    _, hidden_neg_b1 = lm.forward_decode_traced_embeds(
        emb_neg,
        cos1,
        sin1,
        _pos_tt(pos_neg),
        kv_neg_ref,
        return_last_hidden=True,
        need_logits=False,
    )

    # ── B=2 fused ──
    emb_b2 = ttnn.concat([emb_pos, emb_neg], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [2,1,1,H]
    logits0_b2, hidden_b2 = lm.forward_decode_traced_embeds_b2(
        emb_b2,
        [(cos0, sin0), (cos1, sin1)],
        [cur0, cur1],
        [kv_pos_b2, kv_neg_b2],
        lm_head_w=lm_head_valid,
    )

    h_b1_pos = ttnn.to_torch(hidden_pos_b1).float()[0, 0, 0]
    h_b1_neg = ttnn.to_torch(hidden_neg_b1).float()[0, 0, 0]
    h_b2 = ttnn.to_torch(hidden_b2).float()
    lg_b1 = ttnn.to_torch(logits_pos_b1).float().reshape(-1)
    lg_b2 = ttnn.to_torch(logits0_b2).float().reshape(-1)

    d_hpos = _maxdiff(h_b1_pos, h_b2[0, 0, 0])
    d_hneg = _maxdiff(h_b1_neg, h_b2[1, 0, 0])
    d_logits = _maxdiff(lg_b1, lg_b2)

    print(f"\n[cfg_batch2_byteident] pos hidden  row0 maxabsdiff={d_hpos:.6e}")
    print(f"[cfg_batch2_byteident] neg hidden  row1 maxabsdiff={d_hneg:.6e}")
    print(f"[cfg_batch2_byteident] pos logits  maxabsdiff={d_logits:.6e}")
    print(f"[cfg_batch2_byteident] argmax B1={int(lg_b1.argmax())} B2={int(lg_b2.argmax())}")

    assert d_hpos == 0.0, f"pos hidden not byte-identical: {d_hpos}"
    assert d_hneg == 0.0, f"neg hidden not byte-identical: {d_hneg}"
    assert d_logits == 0.0, f"pos logits not byte-identical: {d_logits}"

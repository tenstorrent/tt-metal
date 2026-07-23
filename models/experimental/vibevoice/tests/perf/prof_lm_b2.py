# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated device profile of the deployed CFG batch-2 LM decode forward.

Warms the B=2 28-layer forward, then runs ONE signpost-bounded steady call so
tt-perf-report --start-signpost start --end-signpost stop isolates a single
deploy-shaped LM decode.  Run under tracy:

  VV_TRACE_SEGMENT=0 python -m tracy -v -r -p --op-support-count 100000 -m pytest \
    models/experimental/vibevoice/tests/perf/prof_lm_b2.py -s
"""
import torch
import ttnn
import pytest

from models.experimental.vibevoice.tests.pcc.pcc_helpers import build_tt_lm


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_prof_lm_b2(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder
    hd = cfg.head_dim
    lm = build_tt_lm(lm_state, mesh_device, cfg)

    P_POS, P_NEG = 2048, 8  # realistic deploy prefix on the pos cache
    ids_pos = torch.randint(0, cfg.vocab_size, (1, P_POS), dtype=torch.long)
    ids_neg = torch.randint(0, cfg.vocab_size, (1, P_NEG), dtype=torch.long)
    kv_pos = lm.alloc_kv_cache(P_POS + 64)
    kv_neg = lm.alloc_kv_cache(P_NEG + 64)
    lm.prefill(ids_pos, kv_cache=kv_pos)
    lm.prefill(ids_neg, kv_cache=kv_neg)

    emb_pos = lm._embed(torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long))
    emb_neg = lm._embed(torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long))
    emb_b2 = ttnn.concat([emb_pos, emb_neg], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
            r, device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    cos0, sin0, cos1, sin1 = _rope_row(), _rope_row(), _rope_row(), _rope_row()
    cur0, cur1 = _pos_tt(P_POS), _pos_tt(P_NEG)
    valid = sorted([151643, 151644, 151645, 151646, 151647][: min(5, cfg.vocab_size)])
    lm_head_valid = lm.build_lm_head_subset(valid)

    def _fwd():
        return lm.forward_decode_traced_embeds_b2(
            emb_b2,
            [(cos0, sin0), (cos1, sin1)],
            [cur0, cur1],
            [kv_pos, kv_neg],
            lm_head_w=lm_head_valid,
        )

    for _ in range(3):  # warm (JIT + program cache)
        _fwd()
    ttnn.synchronize_device(mesh_device)

    import tracy

    tracy.signpost("start")
    logits0, hidden_b2 = _fwd()
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("stop")
    print(f"[prof_lm_b2] done; logits {logits0.shape} hidden {hidden_b2.shape}")

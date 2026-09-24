# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gates P2.11-P2.14: full 28-layer chunked prefill on device vs the CPU golden.

    s4096      2k->2k   (2 x 2048)                                   P2.11
    s16384     8k->8k   (2 x 8192)                                   P2.12
    lastchunk  (b) golden KV prefix [0, 51200) loaded, device runs 51200->56320   P2.13
    s56320     (a) all 11 x 5120 chunks on device                    P2.14

Measured on the LAST chunk: per-layer output PCC trail (pcc_layer_Lxx; device activations propagate,
no teacher forcing), final hidden PCC, top-1/top-5 agreement with golden logits on sampled rows.
Over the whole sequence: per-layer K/V cache PCC (pcc_kv_min = worst).
"""

import time

import pytest
import torch

from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.reference.ernie_ref import pcc
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import Golden
from models.demos.ernie45_d_p.tt.model import TtErnieModel

TASK = "P2.11"
CASES = {
    "s4096": dict(seq=4096, chunk=2048, prefix_from_golden=False),
    "s16384": dict(seq=16384, chunk=8192, prefix_from_golden=False),
    "lastchunk": dict(seq=56320, chunk=5120, prefix_from_golden=True),
    "s56320": dict(seq=56320, chunk=5120, prefix_from_golden=False),
}


@mesh_1x4
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_model_chunked(mesh_device, cfg, loader, record, case):
    c = CASES[case]
    G = Golden(c["seq"], c["chunk"])
    task = record.task
    tokens = G.tokens()
    n_chunks = c["seq"] // c["chunk"]
    last = n_chunks - 1

    model = TtErnieModel(mesh_device, loader, cfg)
    metrics.record(task, "model_load_s", round(model.load_seconds, 1))
    cache = model.new_cache(c["seq"])

    first = 0
    if c["prefix_from_golden"]:
        for i in range(cfg.num_hidden_layers):
            k, v = G.kv(i)
            cache.load_prefix(i, k[:, : last * c["chunk"]], v[:, : last * c["chunk"]])
        first = last

    layer_out = {}
    for ci in range(first, n_chunks):
        s = ci * c["chunk"]
        grab = ci == last

        def on_layer(i, h, grab=grab):
            if grab:
                import ttnn

                layer_out[i] = ttnn.to_torch(ttnn.get_device_tensors(h)[0])[0, 0].float()

        t0 = time.time()
        hidden = model.prefill_chunk(tokens[s : s + c["chunk"]], s, cache, on_layer=on_layer if grab else None)
        import ttnn

        ttnn.synchronize_device(mesh_device)
        dt = time.time() - t0
        metrics.record(task, f"chunk_seconds_c{ci:02d}", round(dt, 3))
        print(f"chunk {ci} [{s},{s + c['chunk']}) {dt:.2f}s")
        if ci != last:
            ttnn.deallocate(hidden)

    # Per-layer trail on the last chunk (layers that were dumped)
    for i in range(cfg.num_hidden_layers):
        gl = G.layer(last, i)["out"]
        p = pcc(layer_out[i], gl)
        metrics.record(task, f"pcc_layer_L{i:02d}", p)
    gm = G.model(last)
    final = ttnn.to_torch(ttnn.get_device_tensors(hidden)[0])[0, 0].float()
    record("pcc_final_hidden", final, gm["final_norm"], 0.97)

    # Logits on sampled rows of the last chunk: every 16th row + the final 32 rows
    rows = torch.unique(torch.cat([torch.arange(0, c["chunk"], 16), torch.arange(c["chunk"] - 32, c["chunk"])]))
    lg = model.logits(hidden, rows)
    tt_top5 = lg.topk(5, dim=-1).indices
    g_top = gm["top32_ids"][rows].long()
    top1 = (tt_top5[:, 0] == g_top[:, 0]).float().mean().item()
    top5 = (tt_top5 == g_top[:, :1]).any(-1).float().mean().item()
    metrics.record(task, "top1_match", top1)
    metrics.record(task, "top5_overlap", top5)
    tail = gm["logits_tail"]
    metrics.record(task, "pcc_logits_tail", pcc(lg[-32:], tail))
    print(f"top1 agreement {top1:.4f}  golden-top1 in TT top5 {top5:.4f}  logits tail pcc {pcc(lg[-32:], tail):.5f}")

    # KV cache over the whole sequence
    worst = 1.0
    for i in range(cfg.num_hidden_layers):
        k, v = cache.to_torch(i, c["seq"])
        gk, gv = G.kv(i)
        pk, pv = pcc(k, gk), pcc(v, gv)
        metrics.record(task, f"pcc_kv_k_L{i:02d}", pk)
        metrics.record(task, f"pcc_kv_v_L{i:02d}", pv)
        worst = min(worst, pk, pv)
    metrics.record(task, "pcc_kv_min", worst)
    print(
        f"worst KV pcc {worst:.6f}; worst layer pcc {min(pcc(layer_out[i], G.layer(last, i)['out']) for i in range(cfg.num_hidden_layers)):.6f}"
    )
    assert worst >= 0.97 and top5 >= 0.9
    record.check()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (no device): where the gain of a decode-appended row's RoPE part comes from in a chained run (PLAN.md
section 22). kv_a's rows from the checkpoint, exact, in float64, applied to the DEVICE's own attn_in of every layer
(the dump a chained run of fused/model leaves) and to the golden's attn_in: the gain of the raw k_rope (64 rope rows;
the rotation does not change a gain), of the raw k_pass and of k_pass after the kv norm. No arithmetic of the device's
pool unit or append is in it: what it shows is what the layer's INPUT does to the row.

    python models/experimental/hy4_preview/fused/layer_attn/tests/host_rope_gain.py <golden .pt> <chain dump .pt>
(HY4_HF_MODEL set; TT_METAL_MOCK_CLUSTER_DESC_PATH set, as for anything that imports the test helpers)
"""
import sys, torch
from models.experimental.hy4_preview.tests.common import hy4_tensors

gold_path, dump_path = sys.argv[1], sys.argv[2]
g = torch.load(gold_path, weights_only=False)
d = torch.load(dump_path, weights_only=False)
pcc = lambda a, b: float(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1])
gain = lambda ref, dev: float((ref.double() * dev.double()).sum() / (ref.double() ** 2).sum())
step = d["steps"][0]
J = step["J"]
n = len(g["layers"])
print(f"golden {gold_path.split('/')[-1]}; dump {dump_path.split('/')[-1]}; rows J={J}")
print(
    "layer | attn_in dev vs golden pcc / gain | raw k_rope (kv_a rope rows, exact weights) on the device's input vs on the golden's: pcc / gain | raw k_pass: pcc / gain | k_pass after the kv norm: gain | share of |x|^2 the rope rows see"
)
for l in range(n):
    (w, wn) = hy4_tensors(
        [f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight", f"model.layers.{l}.self_attn.kv_a_layernorm.weight"]
    )
    w = w.double()
    xg = g["layers"][l]["attn_in"].double()[J]
    xd = step["out"][(l, "attn_in")].double()
    rg, rd = xg @ w[512:].t(), xd @ w[512:].t()
    pg, pd = xg @ w[:512].t(), xd @ w[:512].t()
    norm = lambda v: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
    # how much of x the rope rows read: |P x|^2 / |x|^2 with P the projector on the rope rows' span
    q, _ = torch.linalg.qr(w[512:].t())
    share = float(((xg @ q) ** 2).sum() / (xg**2).sum())
    err = xd - xg
    share_err = float(((err @ q) ** 2).sum() / (err**2).sum())
    print(
        f"{l} | {pcc(xg, xd):.6f} / {gain(xg, xd):.4f} | {pcc(rg, rd):.6f} / {gain(rg, rd):.4f} | {pcc(pg, pd):.6f} / {gain(pg, pd):.4f} | {gain(norm(pg), norm(pd)):.4f} | x {share:.4f}, error {share_err:.4f}"
    )

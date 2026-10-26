# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU bf16 control for the per-row accuracy question (D1 item 1): the device decomposition in torch with
bf16 weights and bf16 rounding at every op boundary but fp32 accumulation, next to the plain fp32 dataflow, on
the paper golden's named rows (458, 483, 806, 226), its inactive (flat) rows and a random sample of active rows.
CPU only (python_env), no device:

    python models/experimental/timesfm3_forecast/tests/golden/bf16_control.py <out.json> [--rows N]

Writes the per-row spread ratio, PCC and error std of both controls against the golden head, plus the gate
metrics on the selected rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.experimental.timesfm3_forecast.common import host  # noqa: E402
from models.experimental.timesfm3_forecast.tests.golden import dataflow as df  # noqa: E402
from models.experimental.timesfm3_forecast.tests.golden import make_goldens as mg  # noqa: E402
from models.experimental.timesfm3_forecast.tests.golden_io import active_rows, head_metrics, load_golden  # noqa: E402

NAMED = [458, 483, 806, 226]


def R(t: torch.Tensor) -> torch.Tensor:
    return t.bfloat16().float()


def lin(x, w):
    """bf16 operands, fp32 accumulation, bf16 result."""
    return R(R(x) @ R(w).T)


def norm(x, gain=None, eps=None):
    return R(df.rms_norm(x, None if gain is None else R(gain), eps))


def seq_attention(x, aw, masks, cos, sin):
    q, k, v = (df.split_heads(t) for t in lin(norm(x), aw.wqkv).split(df.NUM_HEADS * df.HEAD_DIM_PAD, -1))
    cos, sin = R(cos), R(sin)
    q, k = R(q * cos + df.rotate_pairs(q) * sin), R(k * cos + df.rotate_pairs(k) * sin)
    eps = torch.finfo(torch.float32).eps * df.QK_EPS_PAD
    q, k = norm(q, aw.q_gain, eps), norm(k, aw.k_gain, eps)
    o = R(df.attend(q, k, v, masks.seq_allowed, df.SDPA_SCALE))
    return R(x + norm(lin(df.merge_heads(o), aw.wo), aw.post_gain))


def layer_bf16(x, lw, masks, cos, sin):
    h1 = seq_attention(x, lw.seq, masks, cos, sin)
    h2 = R(h1 + norm(lin(norm(h1), lw.var.w_ov), lw.var.post_gain))  # V = 1
    ff = lin(R(torch.relu(lin(norm(h2), lw.ff0))), lw.ff1)
    return R(h2 + norm(ff, lw.post_ff_gain))


def forward_bf16(hi, w):
    n = hi.num_patches
    masks = df.Masks.build(hi.num_front_masked, 1, n)
    cos, sin = df.rope_tables(n)
    t = R(hi.tokens.float())
    x = R(lin(R(torch.relu(lin(t, w.tok_hidden))), w.tok_out) + lin(t, w.tok_res))
    for lw in w.layers:
        x = layer_bf16(x, lw, masks, cos, sin)
    return lin(x[:, list(hi.readout_tokens)], w.head_w) + R(w.head_b)


def per_row(ref, got):
    e = ref.double().reshape(ref.shape[0], -1, 9)
    d = got.double().reshape(got.shape[0], -1, 9)
    es, ds = e - e.mean(-1, keepdim=True), d - d.mean(-1, keepdim=True)
    ef, dfl = e.reshape(e.shape[0], -1), d.reshape(d.shape[0], -1)
    ec, dc = ef - ef.mean(-1, keepdim=True), dfl - dfl.mean(-1, keepdim=True)
    pccs = (ec * dc).sum(-1) / (ec.norm(dim=-1) * dc.norm(dim=-1)).clamp_min(1e-300)
    return {
        "spread_ratio": (ds.reshape(d.shape[0], -1).norm(dim=-1) / es.reshape(e.shape[0], -1).norm(dim=-1)).tolist(),
        "pcc": pccs.tolist(),
        "err_std": (dfl - ef).std(-1).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, default=60, help="random active rows besides the named and flat ones")
    ap.add_argument("--all", action="store_true", help="every row of the paper golden")
    args = ap.parse_args()
    torch.set_num_threads(8)
    from models.experimental.timesfm3_forecast.tt.model import load_checkpoint_weights

    g = load_golden("bench_paper")
    active = active_rows(g["head_readout"])
    rng = np.random.default_rng(0)
    pool = [r for r in np.flatnonzero(active.numpy()).tolist() if r not in NAMED]
    rows = (
        NAMED + np.flatnonzero(~active.numpy()).tolist() + sorted(rng.choice(pool, args.rows, replace=False).tolist())
    )
    if args.all:
        rows = list(range(len(active)))
    kinds = np.random.default_rng(20).choice(list(mg.REALISTIC_MIX), size=1024, p=list(mg.REALISTIC_MIX.values()))
    w = load_checkpoint_weights()
    ref = g["head_readout"][rows]
    outs = {"fp32 dataflow": [], "bf16 ops, fp32 accumulation": []}
    with torch.no_grad():
        for i in range(0, len(rows), 128):  # chunks bound the host memory
            hi = host.prepare([g["contexts"][r].numpy() for r in rows[i : i + 128]], 64)
            outs["fp32 dataflow"].append(df.forward(hi.tokens, hi.num_front_masked, 1, w, hi.readout_tokens))
            outs["bf16 ops, fp32 accumulation"].append(forward_bf16(hi, w))
    outs = {k: torch.cat(v) for k, v in outs.items()}
    report = {
        "rows": rows,
        "kinds": [str(kinds[r]) for r in rows],
        "active": [bool(active[r]) for r in rows],
        "controls": {},
    }
    for label, got in outs.items():
        report["controls"][label] = {
            "metrics_on_selected_rows": head_metrics(ref, got, active[rows]),
            "per_row": per_row(ref, got),
        }
        pr = report["controls"][label]["per_row"]
        named = {r: (round(pr["spread_ratio"][i], 3), round(pr["pcc"][i], 4)) for i, r in enumerate(rows) if r in NAMED}
        print(
            f"[CONTROL] {label}: named rows (spread ratio, pcc) {named}; "
            + ", ".join(f"{k} {v:.5f}" for k, v in report["controls"][label]["metrics_on_selected_rows"].items())
        )
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

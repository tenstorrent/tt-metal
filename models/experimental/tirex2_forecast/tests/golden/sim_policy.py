# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU only: the device's exact ``accuracy`` dtype policy, with exact (fp32-accumulated) matmuls.

    python models/experimental/tirex2_forecast/tests/golden/sim_policy.py [--rows 0:64,618] [--y32] [--out f.json]

Mirrors ``tt/layers.py`` rounding point by rounding point on the realistic paper golden:
- weights folded as on device (norm gains, the q scale, the gate soft-cap 1/15, the multi-head-norm weight) and
  then rounded to bf16, except the fp32 mLSTM gate weights and the fp32 biases of ``W_in``;
- bf16 tokens, norm outputs, linear outputs (SiLU applied on the bf16 linear output, as the trailing unary does),
  SwiGLU product, residual stream and head output;
- fp32 outputs where the device keeps them: ``W_in`` + bias, the mLSTM gates, the chunkwise core, its per-head
  LayerNorm and ``nlp_concat_heads``, ``y @ R`` and the whole sLSTM step, the sLSTM per-head norm;
- the bf16 cast of ``y`` before ``y @ R`` at every step (``--y32`` keeps it fp32, the device's ``y32`` variant).
What it does not model: matmul fidelity (HiFi2 / tf32-like fp32 operands), the device's transcendental and norm
kernels. The difference to the device's all-HiFi4 run is therefore the device-kernel error, and the difference
between the device's HiFi4 and HiFi2 runs is fidelity.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden
from models.experimental.tirex2_forecast.tests.pcc import metrics as M


def bf(t):
    return t.bfloat16().float()


def rows_of(w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return g[:, None] * w


def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def mm(x, w, out_bf16=True):
    y = x.double() @ w.double()
    return bf(y.float()) if out_bf16 else y.float()


def residual(x, w: df.ResidualBlockW, gain=None):
    wh, wr = (w.w_hidden, w.w_res) if gain is None else (rows_of(w.w_hidden, gain), rows_of(w.w_res, gain))
    h = bf(F.silu(bf(mm(x, bf(wh), False) + bf(w.b_hidden))))
    out = bf(mm(h, bf(w.w_out), False) + bf(w.b_out + w.b_res))
    res = mm(x, bf(wr))
    return bf(out + res)


def ffn(x, b: df.BlockW):
    xn = bf(rms(x, df.NORM_EPS))
    gate = bf(F.silu(mm(xn, bf(rows_of(b.ffn.w_gate, b.w_norm_ffn)))))
    up = mm(xn, bf(rows_of(b.ffn.w_up, b.w_norm_ffn)))
    h = bf(gate * up)
    return bf(x + mm(h, bf(b.ffn.w_down)))


def mlstm(x, b: df.BlockW):
    c: df.MLSTMW = b.cell
    B, S, _ = x.shape
    xn = bf(rms(x, df.NORM_EPS))
    w = rows_of(c.w_qkvoif, b.w_norm_lstm)
    q, k, v, o, g = torch.split(w, [256, 256, 512, 512, 8], dim=1)
    qq, kk, vv, oo = (mm(xn, bf(t)) for t in (q * df.QK_DIM**-0.5, k, v, o))
    gates = df.GATE_CAP * torch.tanh(mm(xn, g / df.GATE_CAP, False) + c.b_if / df.GATE_CAP)  # fp32 weights
    th = lambda t: t.reshape(B, S, 4, -1).transpose(1, 2)
    i, f = gates[..., :4].transpose(1, 2), gates[..., 4:].transpose(1, 2)
    h = df.mlstm_chunkwise(th(qq) * df.QK_DIM**0.5, th(kk), th(vv), i, f)  # undo: the dataflow applies 1/8
    hn = F.layer_norm(h, (h.shape[-1],), eps=df.NORM_EPS).transpose(1, 2).reshape(B, S, -1)  # fp32
    gated = bf(hn * bf(torch.sigmoid(oo)))
    y = mm(gated, bf(rows_of(c.w_out, c.w_mhnorm)))
    return bf(x + y)


def slstm(x, b: df.BlockW, y32=False):
    c: df.SLSTMW = b.cell
    xn = bf(rms(x, df.NORM_EPS))
    Wx = mm(xn, bf(rows_of(c.w_in, b.w_norm_lstm)), False) + c.bias  # fp32 out, fp32 bias
    r = bf(c.r)
    B, S, _ = Wx.shape
    D = df.D_MODEL
    y = cc = n = m = torch.zeros(B, D)
    ys = []
    for t in range(S):
        if t == 0:
            pre = Wx[:, 0]
        else:
            pre = Wx[:, t] + mm(y if y32 else bf(y), r, False)
        pi, pf, pz, po = pre.split(D, dim=-1)
        if t == 0:
            m, cc, n = pi, torch.tanh(pz), torch.ones_like(pi)
            y = torch.sigmoid(po) * cc
        else:
            lfm = F.logsigmoid(pf.clamp_min(-80)) + m
            mn = torch.maximum(pi, lfm)
            ig, fg = torch.exp(pi - mn), torch.exp(lfm - mn)
            cc = fg * cc + ig * torch.tanh(pz)
            n = torch.clamp_min(fg * n + ig, 1.0)
            y = torch.sigmoid(po) * cc / n
            m = mn
        ys.append(y)
    Y = torch.stack(ys, 1)
    yn = bf(df.mh_layer_norm(Y, torch.ones(D)) * c.w_gnorm)
    return bf(x + yn)


def forward(W: df.TiRex2W, tokens, y32=False):
    x = residual(bf(tokens), W.embed)
    for b in W.blocks:
        x = mlstm(x, b) if df.is_mlstm(b.index) else slstm(x, b, y32)
        x = ffn(x, b)
    xn = bf(rms(x, df.OUT_NORM_EPS))
    return residual(xn, W.head, gain=W.w_out_norm)


def parse_rows(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a or 0), int(b or n)))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0:64")
    ap.add_argument("--y32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(16)
    g = golden("bench_paper")
    rows = parse_rows(args.rows, g["context"].shape[0])
    W = df.build_weights(df.load_state_dict(df.default_checkpoint()))
    prep = df.prepare(g["context"][rows], g["horizon"])
    start = time.perf_counter()
    head = forward(W, prep.tokens, y32=args.y32)[:, df.readout_tokens(g["horizon"])]
    q = df.postprocess(head, prep)
    gh, gq, sc = g["head_out"][rows], g["q_one"][rows], g["scale"][rows]
    m = M.gate_metrics(head, gh, q, gq, sc)
    med = ((q[:, 4] - gq[:, 4]).abs().amax(-1) / sc.reshape(-1)).double()
    kinds = g["kinds"][rows]
    per_row = [
        {"row": r, "kind": g["kind_names"][int(kinds[i])], "median_err": float(med[i])} for i, r in enumerate(rows)
    ]
    per_row.sort(key=lambda d: -d["median_err"])
    print(f"policy sim (y32={args.y32}) on paper rows {args.rows} ({len(rows)} rows, {time.perf_counter() - start:.0f} s)")
    print(M.fmt(m))
    print("worst rows by median error:", per_row[:8])
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": args.rows, "y32": args.y32, "metrics": m, "per_row": per_row}, f, indent=2)


if __name__ == "__main__":
    main()

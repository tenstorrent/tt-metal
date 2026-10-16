# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Grades the dumps of zz_scratch_expert_precision_models.py (several worktrees / ops) against one set of references:
  ref_q     bfp8 x, bfp4 weights (bf16-valued, so every upload path rounds alike), fp32 math: the op's arithmetic
  ref_q_h8  ref_q with h (the activation output, gate/up -> down) rounded to bfp8: what an exact op with a bfp8 h gets
  ref_f     fp32 x and weights: end to end
Usage: python zz_expert_precision_report.py DUMP_PREFIX [DUMP_PREFIX ...] (each holds DUMP_PREFIX_<model>.pt)."""

import sys

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.perf.zz_scratch_expert_precision_models import (
    MODELS,
    X_CASES,
    inputs,
    layout,
    weights,
)


def q(t, dt):
    return ttnn.to_torch(ttnn.from_torch(t.contiguous().float(), dtype=dt, layout=ttnn.TILE_LAYOUT)).float()


def act_fn(name):
    if name == "silu":
        return lambda g, u: torch.nn.functional.silu(g) * u
    return lambda g, u: 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))


def metrics(ref, got):
    ref, got = ref.double(), got.double()
    if not torch.isfinite(got).all():
        return dict(rel=float("nan"), norm=float("nan"), row=float("nan"), pcc=float("nan"))
    rel = float((got - ref).norm() / ref.norm())
    row = float(((got - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)).max())
    a, b = ref.flatten() - ref.mean(), got.flatten() - got.mean()
    return dict(rel=rel, norm=float(got.norm() / ref.norm()), row=row, pcc=float((a @ b) / (a.norm() * b.norm())))


def main():
    prefixes = sys.argv[1:]
    offs, _, sel = layout()
    for model, (H, I, act, _) in MODELS.items():
        dumps = {}
        for p in prefixes:
            lbl, _, p = p.rpartition("=")  # (LABEL=PREFIX: the ops of that dump are named op@LABEL)
            try:
                for (o, c), v in torch.load(f"{p}_{model}.pt").items():
                    dumps[(f"{o}@{lbl}" if lbl else o, c)] = v
            except FileNotFoundError:
                pass
        if not dumps:
            continue
        f = act_fn(act)
        w = weights(model)
        qw = [tuple(q(d[k].T.contiguous(), ttnn.bfloat4_b) for k in ("gate_proj", "up_proj", "down_proj")) for d in w]
        fw = [tuple(d[k].T.contiguous() for k in ("gate_proj", "up_proj", "down_proj")) for d in w]
        ops = sorted({k[0] for k in dumps})
        print(f"\n=== {model}: {H} x {I} {act}, ops {ops} ===")
        print(f"  {'case':18s} {'op':16s} {'arith rel':>10s} {'norm':>7s} {'worst row':>10s} {'vs h8 ref':>10s} {'end-to-end':>10s}")
        agg = {o: [] for o in ops}
        for case in X_CASES:
            x = inputs(model, case)
            xq = q(x, ttnn.bfloat8_b)
            idx = lambda e, s: offs[e] + torch.tensor(s)  # noqa: E731
            ref_q, ref_h8, ref_f = [], [], []
            for e, s in sel.items():
                h = f(xq[idx(e, s)] @ qw[e][0], xq[idx(e, s)] @ qw[e][1])
                ref_q.append(h @ qw[e][2])
                ref_h8.append(q(h, ttnn.bfloat8_b) @ qw[e][2])
                ref_f.append(f(x[idx(e, s)] @ fw[e][0], x[idx(e, s)] @ fw[e][1]) @ fw[e][2])
            ref_q, ref_h8, ref_f = torch.cat(ref_q), torch.cat(ref_h8), torch.cat(ref_f)
            h8_floor = metrics(ref_q, ref_h8)["rel"]
            for o in ops:
                got = dumps.get((o, case))
                if got is None:
                    continue
                mq, mh, mf = metrics(ref_q, got), metrics(ref_h8, got), metrics(ref_f, got)
                if case != "student_t2":  # (heavy tails: every op agrees within ~4%, all ~145% off the host bfp8 x)
                    agg[o].append(mq["rel"])
                print(f"  {case:18s} {o:16s} {mq['rel']:10.4f} {mq['norm']:7.4f} {mq['row']:10.4f} {mh['rel']:10.4f} {mf['rel']:10.4f}")
            print(f"  {case:18s} {'(bfp8 h alone)':16s} {h8_floor:10.4f}")
        for o, v in agg.items():
            if v:
                print(f"  {model} {o:16s} arithmetic rel error (w/o student_t2): median {sorted(v)[len(v) // 2]:.4f}, max {max(v):.4f}")


if __name__ == "__main__":
    main()

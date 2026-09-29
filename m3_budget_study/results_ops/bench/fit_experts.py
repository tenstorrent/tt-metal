# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fit the routed-expert timings from bench_experts.py to two cost models, per path.

  additive  t = a*W + b*T          (weight read and token compute serialise)
  max       t = max(a*W, b*T)      (they overlap; the larger one bounds)

W = weight_bytes_read (MB), T = total_tokens on the chip (tokens_per_expert * active_experts), t = worst_ms
(the slowest chip, which is what the layer waits for; --col mean_ms to fit the chip mean instead).
Both models also get a "+ c" variant (fixed launch cost). Reports R^2, RMSE, max |residual|, mean |%| and
the per-point residuals, and which of the two requested models fits better -> experts_fit.txt.

  python3 fit_experts.py [experts.csv] [--out experts_fit.txt] [--col worst_ms]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path, col):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status", "OK") != "OK" or not r.get(col):
                continue
            rows.append(
                dict(
                    path=r["path"],
                    tpe=int(r["tokens_per_expert"]),
                    act=int(r["active_experts"]),
                    W=float(r["weight_bytes_read"]) / 1e6,
                    T=float(r["tokens_per_expert"]) * float(r["active_experts"]),
                    t=float(r[col]),
                )
            )
    return rows


def stats(t, pred):
    res = t - pred
    sst = float(((t - t.mean()) ** 2).sum())
    sse = float((res**2).sum())
    return dict(
        r2=1 - sse / sst if sst > 0 else float("nan"),
        rmse=float(np.sqrt(sse / len(t))),
        max_abs=float(np.abs(res).max()),
        mean_pct=float((np.abs(res) / t).mean() * 100),
        res=res,
    )


def fit_additive(W, T, t, intercept):
    A = np.stack([W, T] + ([np.ones_like(W)] if intercept else []), axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef, A @ coef


def fit_max(W, T, t, intercept):
    """min SSE of max(a*W, b*T) [+ c]: alternate regime assignment + LS refit, from a log-grid start."""

    def pred(a, b, c):
        return np.maximum(a * W, b * T) + c

    def sse(a, b, c):
        return float(((t - pred(a, b, c)) ** 2).sum())

    a0, b0 = np.median(t / W), np.median(t / T)
    best = None
    for fa in np.logspace(-1.5, 0.5, 41):
        for fb in np.logspace(-1.5, 0.5, 41):
            a, b = a0 * fa, b0 * fb
            c = float(np.mean(t - pred(a, b, 0))) if intercept else 0.0
            s = sse(a, b, c)
            if best is None or s < best[0]:
                best = (s, a, b, c)
    _, a, b, c = best
    for _ in range(100):
        wreg = a * W >= b * T
        na, nb, nc = a, b, c
        y = t - c
        if wreg.any():
            na = float((W[wreg] * y[wreg]).sum() / (W[wreg] ** 2).sum())
        if (~wreg).any():
            nb = float((T[~wreg] * y[~wreg]).sum() / (T[~wreg] ** 2).sum())
        if intercept:
            nc = float(np.mean(t - pred(na, nb, 0)))
        if sse(na, nb, nc) >= sse(a, b, c) - 1e-15:
            break
        a, b, c = na, nb, nc
    return np.array([a, b] + ([c] if intercept else [])), pred(a, b, c)


def fmt_coef(name, coef):
    a, b = coef[0], coef[1]
    s = f"a={a:.5g} ms/MB (-> {1 / a if a > 0 else float('inf'):.3g} GB/s eff. weight BW)  b={b * 1e3:.5g} us/token"
    if len(coef) > 2:
        s += f"  c={coef[2] * 1e3:.4g} us"
    return f"{name:14s} {s}"


def report(rows, label, lines):
    W = np.array([r["W"] for r in rows])
    T = np.array([r["T"] for r in rows])
    t = np.array([r["t"] for r in rows])
    lines.append(f"\n=== {label}: {len(rows)} points ===")
    if len(rows) < 3:
        lines.append("  too few points")
        return
    results = {}
    for name, fn, icpt in (
        ("additive", fit_additive, False),
        ("max", fit_max, False),
        ("additive+c", fit_additive, True),
        ("max+c", fit_max, True),
    ):
        coef, pred = fn(W, T, t, icpt)
        st = stats(t, pred)
        results[name] = (coef, pred, st)
        lines.append(
            fmt_coef(name, coef)
            + f"\n{'':14s} R2={st['r2']:.4f} RMSE={st['rmse'] * 1e3:.1f} us max|res|={st['max_abs'] * 1e3:.1f} us "
            f"mean|res|={st['mean_pct']:.1f}%"
        )
    a_st, m_st = results["additive"][2], results["max"][2]
    better = "max" if m_st["rmse"] < a_st["rmse"] else "additive"
    lines.append(
        f"VERDICT ({label}): {better} fits better "
        f"(R2 additive {a_st['r2']:.4f} vs max {m_st['r2']:.4f}; RMSE {a_st['rmse'] * 1e3:.1f} vs {m_st['rmse'] * 1e3:.1f} us)"
    )
    lines.append(f"{'tok/exp':>7} {'active':>6} {'W MB':>8} {'T':>6} {'t ms':>9} {'res_add us':>11} {'res_max us':>11}")
    for i, r in enumerate(rows):
        lines.append(
            f"{r['tpe']:7d} {r['act']:6d} {r['W']:8.1f} {int(r['T']):6d} {r['t']:9.4f} "
            f"{a_st['res'][i] * 1e3:11.1f} {m_st['res'][i] * 1e3:11.1f}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default=str(HERE / "experts.csv"))
    p.add_argument("--out", default=str(HERE / "experts_fit.txt"))
    p.add_argument("--col", default="worst_ms")
    args = p.parse_args()
    rows = load(args.csv, args.col)
    lines = [
        f"routed-expert cost-model fit: {args.csv} column {args.col} (W = weight MB read per chip, T = tokens per chip)"
    ]
    for path in sorted({r["path"] for r in rows}):
        report([r for r in rows if r["path"] == path], f"path={path}", lines)
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

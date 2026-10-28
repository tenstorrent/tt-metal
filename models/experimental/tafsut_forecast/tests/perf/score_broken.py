# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs against the gate floors, in both precisions, on every bench and suite case (CPU only).

    python models/experimental/tafsut_forecast/tests/perf/score_broken.py [--gates gates.json] [--out out.json]

Adapted from TiRex-2 ``651e8c4af93`` ``tests/perf/broken_outputs.py`` (PLAN v1.9, v1.20). Each broken output is
made from forward 1's golden head output and scored by the bench's metric code: ``collapsed`` (every quantile ->
the quantile mean), ``inactive_collapsed`` (only inactive rows collapsed), ``active1pct_0.3x`` (the band of ceil(1 %)
of the active rows, the widest ones, scaled 0.3x about the quantile mean), ``all_0.88x`` and ``all_1.12x`` (every
band scaled) and ``sign_flip`` (the head negated: the forecast mirrored about the series location, quantile order
reversed; Tafsut has no sign-flip path of its own, so this is the output-level analog of TiRex-2's D7 case),
``level_shift`` (+0.05 on every value in normalized asinh space; also reported in real units against the row std),
``shift10pct`` (+0.05 on ceil(10 %) of the active rows only, D3 item 3), ``level_shift_rows#p`` (+0.05 on 1 active row
at B <= 16, 2 at B <= 32, 4 at B <= 64, ceil(1 %) of B above; D4 item 1), ``band1.3x_row#p`` (one active row's band
scaled 1.3x; D4 item 2), each at PLACEMENTS random placements, and ``lag1`` (every quantile track delayed one step: step t takes step t-1's value, step 0 kept; PLAN v1.22).
Every broken output is also made from forward 2's golden head (teacher-forced) wherever the case has a forward 2,
and scored as the ``f2_*`` family (D3 item 1). A broken record is the fixed round-0 record of its case and
precision with the broken forward's family replaced, so the other forward passes as measured. Each must fail at
least one floor or drift check (against the fixed round-0 records) per case, forward and precision, **beyond the
checks the golden itself fails** when scored the same way (``golden``, scored first; D4 item 5). Exit status 1 otherwise.

v1.27: the golden is scored with its own row vectors (bias 0, spread ratio 1 on every row) and must pass every
drift; ``level_shift_rows`` and ``shift10pct`` (+/-0.05) and the one-row ``band1.3x`` / ``band0.7x`` probes come in both signs, and the
sign that moves a probed row away from the reference must fail (the other is reported). The golden's own failures
are subtracted by (metric, row).

Row vectors (D4 items 1-2): the per-row signed bias and spread ratio of a broken record are the fixed record's device
rows with the broken output's own row change composed onto them, ``e0_i + e_i`` and ``r0_i x r_i``. This is exact for
the shift and band-scaling probes (the bias is linear in the output; a band scaled about its quantile mean scales its
ratio), so a probe is scored as it would land on the device output, not on the golden.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.pcc import metrics as M
import torch

from models.experimental.tafsut_forecast.tests.perf.check_gates import (
    DEFAULT_BASELINES,
    DEFAULT_GATES,
    check,
    load_records,
    session_stamp,
)

BENCH = ("paper", "long_context", "long_horizon", "gift_ragged", "gift_ragged_odd")
SUITE = ("single", "single_long", "ragged4", "model16", "fallback", "horizon4")


def scale_band(head, rows, factor):
    h = head.reshape(*head.shape[:-1], M.Q, M.P).clone()
    mean = h.mean(-2, keepdim=True)
    h[rows] = mean[rows] + factor * (h[rows] - mean[rows])
    return h.reshape(head.shape)


LEVEL_SHIFT = 0.05  # normalized (asinh) units


def lag1(head):
    h = head.reshape(*head.shape[:-1], M.Q, M.P).clone()
    h[..., 1:] = h[..., :-1].clone()
    return h.reshape(head.shape)


def shift_rows(head, rows, sign=1.0, size=LEVEL_SHIFT):
    h = head.clone()
    h[rows] += sign * size
    return h


# D6 / PLAN v1.28: named rows whose population floor binds before the row drift get forced probes in both signs: the
# level shift, and a smaller shift that lands between the floor and the drift (horizon4/performance row 1: round 0
# -0.0561, bias_row_max 0.08 fails at -0.081, the drift at -0.086), which only the binding floor can catch.
FORCED_ROWS = {"horizon4": {1: 0.027}}


PLACEMENTS = 8


def n_shift_rows(rows: int) -> int:
    """PLAN v1.25 sizes: 1 row at B <= 16, 2 at B <= 32, 4 at B <= 64, ceil(1 %) above."""
    return 1 if rows <= 16 else 2 if rows <= 32 else 4 if rows <= 64 else math.ceil(0.01 * rows)


def broken(g_head, shape=None):
    act = M.active_rows(g_head)
    idx = act.nonzero().flatten()
    k = math.ceil(0.01 * len(idx))
    width = M.spread(g_head).double().reshape(g_head.shape[0], -1).norm(dim=1)[idx]
    wide = idx[width.argsort(descending=True)[:k]]
    inact = M.collapsed(g_head).clone()
    inact[act] = g_head[act]
    out = {
        "collapsed": (M.collapsed(g_head), "all rows"),
        "inactive_collapsed": (inact, f"{int((~act).sum())} inactive rows"),
        "active1pct_0.3x": (scale_band(g_head, wide, 0.3), f"{k} of {len(idx)} active rows"),
        "all_0.88x": (scale_band(g_head, slice(None), 0.88), "all rows"),
        "all_1.12x": (scale_band(g_head, slice(None), 1.12), "all rows"),
        "sign_flip": (-g_head, "all rows"),
        "level_shift": (g_head + LEVEL_SHIFT, "all rows"),
        "shift10pct": (
            shift_rows(g_head, idx[: math.ceil(0.1 * len(idx))]),
            f"{math.ceil(0.1 * len(idx))} of {len(idx)} active rows",
        ),
        "shift10pct-": (
            shift_rows(g_head, idx[: math.ceil(0.1 * len(idx))], -1.0),
            f"{math.ceil(0.1 * len(idx))} of {len(idx)} active rows",
        ),
        "lag1": (lag1(g_head), "all rows"),
    }
    gen = torch.Generator().manual_seed(0)
    n = min(n_shift_rows(g_head.shape[0]), len(idx))
    for p in range(PLACEMENTS):
        rows = idx[torch.randperm(len(idx), generator=gen)[:n]]
        what = f"{n} of {len(idx)} active rows {rows.tolist()[:4]}"
        out[f"level_shift_rows#{p}"] = (shift_rows(g_head, rows), what)
        out[f"level_shift_rows-#{p}"] = (shift_rows(g_head, rows, -1.0), what)  # v1.27: both signs
    for p in range(PLACEMENTS):
        row = idx[torch.randint(len(idx), (1,), generator=gen)]
        out[f"band1.3x_row#{p}"] = (scale_band(g_head, row, 1.3), f"active row {int(row)}")
        out[f"band0.7x_row#{p}"] = (scale_band(g_head, row, 0.7), f"active row {int(row)}")
    for row, size in FORCED_ROWS.get(shape, {}).items():
        out[f"level_shift_row{row}"] = (shift_rows(g_head, [row]), f"forced row {row}")
        out[f"level_shift_row{row}-"] = (shift_rows(g_head, [row], -1.0), f"forced row {row}")
        out[f"bound_row{row}"] = (shift_rows(g_head, [row], size=size), f"forced row {row}, {size}")
        out[f"bound_row{row}-"] = (shift_rows(g_head, [row], -1.0, size), f"forced row {row}, -{size}")
    if not bool((~act).any()):  # no inactive rows: the "broken" output is the golden itself
        del out["inactive_collapsed"]
    return out


def case_inputs(name, k=0):
    """Forward ``k``'s golden head at its read-out tokens, its prep, its horizon slice and the golden quantiles there."""
    g = load_golden(name)
    take = min(g["horizon"] - 1024 * k, 1024)
    gh = g["forwards"][k]["head"][:, : -(-take // df.PATCH)]
    x = g["x"] if k == 0 else torch.cat([g["x"], g["quantiles"][:, :1024, df.MEDIAN_IDX]], -1)[:, -df.MAX_CTX :]
    return g, gh, df.prepare_inputs(x), take


def check_id(failure: str) -> str:
    """``<metric>[ drift][ row <i>]`` of a check_gates failure line, its values stripped (v1.27: by metric and row)."""
    rest = failure.split(") ", 1)[-1].split(" ")
    row = f" row {rest[rest.index('row') + 1]}" if "row" in rest[1:-1] else ""
    return rest[0] + (" drift" if len(rest) > 1 and rest[1] == "drift" else "") + row


BIAS_DRIFT, RATIO_DRIFT = 0.03, 0.15  # the v1.27 row drift bounds (gates.json drift)


def away(name: str, e0: list, r0: list, e: list, r: list) -> bool:
    """PLAN v1.28: whether a row probe is required to fail. It may pass ("toward") only if every probed row moves toward
    the reference and its composed value lands inside the v1.27 interval [min(ref, x0) - d, max(ref, x0) + d] (ref 0 / d 0.03 for bias, 1 / 0.15 for
    the spread ratio); a probe that crosses the reference and overshoots is required to fail. Every other broken
    output is required to fail."""
    toward = lambda x, x0, ref, d: abs(x - ref) < abs(x0 - ref) and min(ref, x0) - d <= x <= max(ref, x0) + d
    if name.startswith(("level_shift_row", "shift10pct", "bound_row")):
        return any(abs(d) > 1e-9 and not toward(x + d, x, 0.0, BIAS_DRIFT) for x, d in zip(e0, e))
    if name.startswith("band"):
        return any(abs(q - 1) > 1e-9 and not toward(x * q, x, 1.0, RATIO_DRIFT) for x, q in zip(r0, r))
    return True


def compose_population(m: dict, sbias: torch.Tensor, act: torch.Tensor, e0: list, rv: list) -> dict:
    """PLAN v1.28: the population metrics recomputed from the composed rows, so score_broken scores those floors too.
    ``sbias`` is the probe's signed bias on every row; the fixed record's device bias is added on the active rows
    (its inactive rows have no recorded bias and count as 0). ``rv`` is the composed spread ratio of the active rows."""
    b = sbias.clone()
    b[act] += torch.tensor(e0, dtype=b.dtype)
    a = b.abs()
    out = {"bias_row_p50": float(a.median()), "bias_row_max": float(a.max())}
    if m.get("bias_row_p95") is not None:
        out["bias_row_p95"] = float(torch.quantile(a, 0.95))
    r = torch.tensor(rv, dtype=torch.float64)
    if len(r):
        out["spread_ratio_min"], out["spread_ratio_max"] = float(r.min()), float(r.max())
    for kk, qq in (("spread_ratio_p01", 0.01), ("spread_ratio_p99", 0.99), ("spread_ratio_p50", 0.5)):
        if m.get(kk) is not None:
            out[kk] = float(torch.quantile(r, qq))
    return out


def score(gates, base, quiet=True):
    """Every (case, forward, precision, broken output); ``fails`` lists the floor and drift checks it fails."""
    import contextlib
    import io

    base_by = {(r["name"], r["precision"]): r for r in base}
    stamp = session_stamp()
    results = []
    for shape in BENCH + SUITE:
        g = load_golden(shape)
        for k in range(len(g["forwards"])):
            g, gh, prep, take = case_inputs(shape, k)
            gq = g["quantiles"].transpose(1, 2)  # [rows, 9, H]
            fam = "f2_" if k else ""
            golden_fails = {}
            for name, (head, what) in [("golden", (gh, "none"))] + list(broken(gh, shape).items()):
                qk = df.postprocess(head.float(), prep, take).transpose(1, 2)
                if k == 0:  # forward 1: its quantiles are the forecast's first `take` steps
                    m = M.gate_metrics(head, gh, qk, gq[:, :, :take], prep.scale)
                else:  # forward 2 as the bench scores it: the full forecast, golden forward 1 then broken forward 2
                    q = torch.cat([gq[:, :, :1024], qk], -1)
                    m = {
                        kk: v
                        for kk, v in M.gate_metrics(head, gh, q, gq, prep.scale).items()
                        if not kk.startswith("pcc_final")
                    }
                if name == "level_shift":  # the shift in real units, against the per-row std of the golden forecast
                    gs = gq[:, :, 1024 * k : 1024 * k + take]
                    d = (qk - gs).abs().double().mean((1, 2)) / gs.double().std((1, 2)).clamp_min(1e-12)
                    m["real_shift_over_row_std_p50"] = float(d.median())
                for precision in ("accuracy", "performance"):
                    b = base_by.get((shape, precision), {})
                    mp = dict(m)
                    required = True
                    if fam + "bias_row_vec" in b and name != "golden":  # composed onto the fixed record's device rows
                        required = away(
                            name,
                            b[fam + "bias_row_vec"],
                            b[fam + "spread_ratio_row_vec"],
                            m["bias_row_vec"],
                            m["spread_ratio_row_vec"],
                        )
                        mp["bias_row_vec"] = [x + y for x, y in zip(b[fam + "bias_row_vec"], m["bias_row_vec"])]
                        mp["spread_ratio_row_vec"] = [
                            x * y for x, y in zip(b[fam + "spread_ratio_row_vec"], m["spread_ratio_row_vec"])
                        ]
                        sbias = (head.double() - gh.double()).reshape(gh.shape[0], -1).mean(1)
                        mp.update(
                            compose_population(
                                m, sbias, M.active_rows(gh), b[fam + "bias_row_vec"], mp["spread_ratio_row_vec"]
                            )
                        )
                    keep = {
                        kk: v for kk, v in b.items() if (kk.startswith("f2_") if not k else not kk.startswith("f2_"))
                    }
                    rec = dict(keep, **{fam + kk: v for kk, v in mp.items()}, name=shape, precision=precision, **stamp)
                    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                        fails = check([rec], base, gates, allow_unstamped=True)  # stamped once, in the output
                    if name == "golden":
                        golden_fails[precision] = {check_id(f) for f in fails}
                    beyond = [f for f in fails if check_id(f) not in golden_fails[precision]]
                    results.append(
                        {
                            "shape": shape,
                            "forward": k + 1,
                            "broken": name,
                            "rows": what,
                            "precision": precision,
                            "fails": beyond,
                            "required": required,
                            "golden_also_fails": len(fails) - len(beyond),
                            "metrics": {kk: v for kk, v in m.items() if not isinstance(v, list)},
                        }
                    )
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--baseline", nargs="*", type=Path, default=list(DEFAULT_BASELINES))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    st = session_stamp()
    print(
        f"score_broken at {st['git_head']} dirty={st['git_dirty']!r} started_at {st['started_at']}, gates {args.gates}"
    )
    results = score(gates, load_records(args.baseline))
    bad, golden_bad = [], []
    for r in results:
        if r["broken"] == "golden":
            if r["golden_also_fails"]:
                golden_bad.append(f"{r['shape']}/f{r['forward']}/{r['precision']}: {r['golden_also_fails']}")
            continue
        short = sorted({f.split(" ", 2)[-1].split(" ")[0] + (" drift" if " drift" in f else "") for f in r["fails"]})
        extra = (
            f" real shift/row std p50 {r['metrics']['real_shift_over_row_std_p50']:.3f}"
            if r["broken"] == "level_shift"
            else ""
        )
        print(
            f"{r['shape']:15s} f{r['forward']} {r['precision']:11s} {r['broken']:19s} ({r['rows']}): fails {len(r['fails'])} {short}{extra}"
            + ("" if r["required"] else " (toward the reference: may pass)")
        )
        if not r["fails"] and r["required"]:
            bad.append(f"{r['shape']}/f{r['forward']}/{r['precision']}/{r['broken']}")
    if args.out:
        args.out.write_text(json.dumps({"stamp": st, "results": results}, indent=1) + "\n")
    print(
        f"golden scored as an output fails {len(golden_bad)} case(s)"
        + (": " + ", ".join(golden_bad) if golden_bad else "")
    )
    print(f"{len(bad)} broken output(s) pass every floor and drift check" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

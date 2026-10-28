# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Where the named rows' error arises (D4 item 3): stage-by-stage error of single rows, teacher-forced.

For each named row (the worst rows behind the forward-2 and row-bias case floors) and a few control rows (random
active rows of the same case), the device runs the case's forward ``k`` input on just those rows (forward 2 is
teacher-forced: the golden forward-1 median, as the bench) and taps the residual stream after the embedding and every
block. The fp32 reference (``dataflow.forward``) runs the same rows with the same taps. Each row prints:

- its head metrics against the golden: signed bias, spread ratio and row PCC (also of the reference, which must
  reproduce the golden), and the fixed record's value of that row where it has one (the sub-batch must reproduce the
  full batch);
- the relative L2 error ||dev - ref|| / ||ref|| of the future (read-out) rows and of the valid context rows after the
  embedding and after each block: steady growth through the stack is precision, a jump at one stage is a defect.

``test_row_conditioning`` (CPU only) measures how ill-conditioned the rows are: the change of each row's bias and
spread ratio when every weight is perturbed by bf16-sized relative noise (2^-9 x N(0, 1)), named rows against the
control rows' distribution.
"""

from __future__ import annotations

import json

import pytest
import torch

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.pcc import metrics as M
from models.experimental.tafsut_forecast.tests.perf.check_gates import DEFAULT_BASELINES, load_records
from models.experimental.tafsut_forecast.tests.perf.test_forward_perf import forward_inputs

# (case, forward index, named rows): D4 item 3; 121, 394, 250 and 973 are the case band floors' rows (D5 R1-0a)
NAMED = [
    ("long_horizon", 1, [16, 311, 393, 250]),
    ("long_horizon", 0, [283, 121, 394]),
    ("paper", 0, [536, 973]),
    ("horizon4", 0, [1]),
]
N_CONTROL = 5


def rows_of(case, k, named):
    g = load_golden(case)
    take = min(g["horizon"] - 1024 * k, 1024)
    gh = g["forwards"][k]["head"][:, : -(-take // df.PATCH)]
    act = M.active_rows(gh).nonzero().flatten()
    pool = [int(i) for i in act if int(i) not in named]
    gen = torch.Generator().manual_seed(0)
    ctl = sorted(pool[int(i)] for i in torch.randperm(len(pool), generator=gen)[: min(N_CONTROL, len(pool))])
    return g, gh, forward_inputs(g)[k], named + ctl, len(named)


def row_head_metrics(head, gh):
    """Signed bias, spread ratio and PCC of each row against the golden head ``gh`` (same rows)."""
    bias = (head.double() - gh.double()).reshape(gh.shape[0], -1).mean(1)
    return bias, M.spread_ratios(gh, head), M.row_pccs(gh, head)


def record_rows(case, precision, k):
    fam = "f2_" if k else ""
    for r in load_records(DEFAULT_BASELINES):
        if (r["name"], r["precision"]) == (case, precision) and fam + "row_active_idx" in r:
            idx = r[fam + "row_active_idx"]
            return {i: (r[fam + "bias_row_vec"][j], r[fam + "spread_ratio_row_vec"][j]) for j, i in enumerate(idx)}
    return {}


def rel(d, r):
    return float((d - r).double().norm() / r.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("case,k,named", NAMED, ids=[f"{c}-f{k + 1}" for c, k, _ in NAMED])
def test_row_stages(device, case, k, named, precision):
    from models.experimental.tafsut_forecast.tt.common import Precision
    from models.experimental.tafsut_forecast.tt.model import TtTafsut

    g, gh, x, rows, n_named = rows_of(case, k, named)
    n = gh.shape[1]
    prep = df.prepare_inputs(x[rows])
    tt = TtTafsut(device, precision=Precision.by_name(precision))
    dtaps = []
    dev = tt.forward(prep, taps=dtaps)[:, :n]
    rtaps = []
    ref = df.forward(df.Weights(), prep, n_read=df.NFUT, taps=rtaps)[:, :n]
    P = prep.tokens.shape[1]
    rec = record_rows(case, precision, k)
    b, r, p = row_head_metrics(dev, gh[rows])
    rb, rr, rp = row_head_metrics(ref, gh[rows])
    print(f"\n[rows] {case} f{k + 1} {precision}: named {rows[:n_named]}, controls {rows[n_named:]}")
    for j, row in enumerate(rows):
        valid = prep.key_valid[j]
        fut = [rel(dtaps[0][j, P : P + df.NFUT], rtaps[0][j, P : P + df.NFUT])]
        ctx = [rel(dtaps[0][j, :P][valid], rtaps[0][j, :P][valid])]
        for i in range(1, len(rtaps)):
            if i < len(rtaps) - 1:
                fut.append(rel(dtaps[i][j, P : P + df.NFUT], rtaps[i][j, P : P + df.NFUT]))
                ctx.append(rel(dtaps[i][j, :P][valid], rtaps[i][j, :P][valid]))
            else:  # read-out cut: the last block holds the future rows only
                fut.append(rel(dtaps[i][j, : df.NFUT], rtaps[i][j, : df.NFUT]))
        fut.append(rel(dev[j], ref[j]))
        hw = float(M.spread(gh[row : row + 1]).double().abs().mean())
        full = rec.get(row)
        print(
            f"[rows] {'NAMED' if j < n_named else 'ctl  '} row {row:4d}: dev bias {float(b[j]):+.4f} ratio "
            f"{float(r[j]):.4f} pcc {float(p[j]):.5f} | ref bias {float(rb[j]):+.1e} ratio {float(rr[j]):.5f} "
            f"pcc {float(rp[j]):.6f} | full batch "
            + (f"bias {full[0]:+.4f} ratio {full[1]:.4f}" if full else "n/a")
            + f" | |spread| mean {hw:.3f}"
            + f"\n[rows]     fut rel err (embed, blocks 0-{df.NL - 1}, head): "
            + " ".join(f"{e:.4f}" for e in fut)
            + f"\n[rows]     ctx rel err (embed, blocks 0-{df.NL - 2}):       "
            + " ".join(f"{e:.4f}" for e in ctx)
        )
    assert float(rp.min()) > 0.9999, "the reference does not reproduce the golden rows"


# D5 R1-0c: the rows whose error jumps at the last block (283) or at the head (536); D6 R1-0e(b): horizon4 row 1
TAIL = [("long_horizon", 0, [283]), ("paper", 0, [536]), ("horizon4", 0, [1])]


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("case,k,named", TAIL, ids=[f"{c}-f{k + 1}" for c, k, _ in TAIL])
def test_tail_isolation(device, case, k, named, precision):
    """The last block and the head alone, fed the reference's residual stream: block 13 from the reference's block-12
    output, the head from the reference's block-13 output. Error made there on the named rows against the controls."""
    import ttnn

    from models.experimental.tafsut_forecast.tt.common import Precision
    from models.experimental.tafsut_forecast.tt.model import TtTafsut

    g, gh, x, rows, n_named = rows_of(case, k, named)
    n = gh.shape[1]
    prep = df.prepare_inputs(x[rows])
    tt = TtTafsut(device, precision=Precision.by_name(precision))
    rtaps = []
    ref = df.forward(df.Weights(), prep, n_read=df.NFUT, taps=rtaps)
    P = prep.tokens.shape[1]
    c = tt.constants(len(rows), prep.ctx_len)
    inp = tt.upload_inputs(tt.host_inputs(prep))
    up = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h13 = tt.layer(up(rtaps[df.NL - 1]), tt.layers[df.NL - 1], inp, c, last=True)
    b13 = ttnn.to_torch(h13).float()
    o13 = tt.head(h13)
    head13 = ttnn.to_torch(o13).float()
    oh = tt.head(up(rtaps[df.NL]))
    headh = ttnn.to_torch(oh).float()
    for t in (h13, o13, oh):
        ttnn.deallocate(t)
    tt.deallocate_inputs(inp)
    fut = slice(0, df.NFUT)
    print(f"\n[tail] {case} f{k + 1} {precision}: named {rows[:n_named]}, controls {rows[n_named:]}")
    print("[tail] rel err: input (ref bf16) | block 13 out | head via device block 13 | head from ref block 13 | bias")
    for j, row in enumerate(rows):
        e_in = rel(rtaps[df.NL - 1][j, P : P + df.NFUT].bfloat16().float(), rtaps[df.NL - 1][j, P : P + df.NFUT])
        e13 = rel(b13[j, fut], rtaps[df.NL][j, fut])
        eh13 = rel(head13[j, :n], ref[j, :n])
        ehh = rel(headh[j, :n], ref[j, :n])
        bias = float((head13[j, :n].double() - ref[j, :n].double()).mean())
        print(
            f"[tail] {'NAMED' if j < n_named else 'ctl  '} row {row:4d}: {e_in:.4f} | {e13:.4f} | {eh13:.4f} | "
            f"{ehh:.4f} | {bias:+.4f}"
        )


EPS = 2.0**-9
SEEDS = 3


@pytest.mark.parametrize("case,k,named", NAMED, ids=[f"{c}-f{k + 1}" for c, k, _ in NAMED])
def test_row_conditioning(case, k, named):
    """CPU: per-row bias / spread-ratio change under bf16-sized weight noise (rows only, fp32 reference)."""
    g, gh, x, rows, n_named = rows_of(case, k, named)
    rows = rows[:n_named] + rows_of_controls(gh, named)
    n = gh.shape[1]
    prep = df.prepare_inputs(x[rows])
    sd = df.load_state_dict()
    base = df.forward(df.Weights(sd), prep)[:, :n]
    db, dr = [], []
    for s in range(SEEDS):
        gen = torch.Generator().manual_seed(s)
        noisy = {
            key: (v * (1 + EPS * torch.randn(v.shape, generator=gen)) if v.is_floating_point() else v)
            for key, v in sd.items()
        }
        h = df.forward(df.Weights(noisy), prep)[:, :n]
        b, r, _ = row_head_metrics(h, base)
        db.append(b.abs())
        dr.append((r - 1).abs())
    db, dr = torch.stack(db).amax(0), torch.stack(dr).amax(0)
    cb, cr = db[n_named:], dr[n_named:]
    out = {"case": case, "forward": k + 1, "controls": len(cb)}
    print(
        f"\n[cond] {case} f{k + 1}: {len(cb)} control rows: |d bias| p50 {float(cb.median()):.2e} max "
        f"{float(cb.max()):.2e}; |d ratio| p50 {float(cr.median()):.2e} max {float(cr.max()):.2e}"
    )
    for j in range(n_named):
        pb = float((cb < db[j]).double().mean())
        pr = float((cr < dr[j]).double().mean())
        print(
            f"[cond] NAMED row {rows[j]}: |d bias| {float(db[j]):.2e} (above {pb:.0%} of controls), "
            f"|d ratio| {float(dr[j]):.2e} (above {pr:.0%})"
        )
        out[rows[j]] = {"d_bias": float(db[j]), "d_ratio": float(dr[j]), "pct_bias": pb, "pct_ratio": pr}
    print("[cond] json " + json.dumps(out))


N_COND_CONTROL = 48


def rows_of_controls(gh, named):
    act = [int(i) for i in M.active_rows(gh).nonzero().flatten() if int(i) not in named]
    gen = torch.Generator().manual_seed(1)
    take = min(N_COND_CONTROL, len(act))
    return sorted(act[int(i)] for i in torch.randperm(len(act), generator=gen)[:take]) if take else []


N_WIDE = 24
WIDE_SEED = 50


def test_row_conditioning_wide():
    """D6 R1-0e(b), CPU: horizon4 row 1 under bf16-sized weight noise against its 3 case rows plus 24 extra 2048-step
    realistic-mix series (seed 50, horizon 1100, reference only). Ranks row 1's |d bias| and its reference-to-noisy
    head gain among all controls."""
    from models.experimental.tafsut_forecast.tests.golden.make_goldens import make_series_mix

    g = load_golden("horizon4")
    gh = g["forwards"][0]["head"]
    n = min(-(-min(g["horizon"], 1024) // df.PATCH), gh.shape[1])
    xw, _ = make_series_mix(N_WIDE, g["x"].shape[1], seed=WIDE_SEED)
    x = torch.cat([g["x"][[1, 0, 2, 3]], xw.to(g["x"].dtype)])
    prep = df.prepare_inputs(x)
    sd = df.load_state_dict()
    base = df.forward(df.Weights(sd), prep)[:, :n]
    db = []
    for s in range(SEEDS):
        gen = torch.Generator().manual_seed(s)
        noisy = {
            key: (v * (1 + EPS * torch.randn(v.shape, generator=gen)) if v.is_floating_point() else v)
            for key, v in sd.items()
        }
        b, _, _ = row_head_metrics(df.forward(df.Weights(noisy), prep)[:, :n], base)
        db.append(b.abs())
    db = torch.stack(db).amax(0)
    act = M.active_rows(base)
    act[0] = True
    cb = db[1:][act[1:]]
    pb = float((cb < db[0]).double().mean())
    rank_med = "above" if db[0] > cb.median() else "at or below"
    print(
        f"\n[wide] horizon4 row 1 against {len(cb)} active controls (3 case rows + {N_WIDE} extra, seed {WIDE_SEED}): "
        f"|d bias| {float(db[0]):.2e}, controls p50 {float(cb.median()):.2e} max {float(cb.max()):.2e}; "
        f"row 1 above {pb:.0%} of controls ({rank_med} the median)"
    )
    print("[wide] json " + json.dumps({"d_bias": float(db[0]), "pct_bias": pb, "controls": len(cb),
                                       "ctl_p50": float(cb.median()), "ctl_max": float(cb.max())}))

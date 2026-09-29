# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 3: block type kda_moe (layer 4) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 4 (kda_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm

Reviewed (S.kda_moe.03.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attn_norm is input_layernorm, w * x * rsqrt(mean(x^2) + 1e-5), on attn_in
[S, H] (w nearly constant, 0.162..0.220, so a wrong or missing weight is mostly a scale). The structure is the dsa_moe
swap 03 test (layer 3) on top of the kda_moe swap 02 limits (layer 4). Unlike layer 3 (post <= 0.024, q/kv norms after
it), layer 4's KDA passes the norm scale into v and post is not saturated, so the norm share of block out is large and
its limits are much tighter than layer 3's. Measured on the CPU (golden s4096 chunk 1, CPU attn_hc and collapse,
attn_norm output perturbed, rest CPU; host script /tmp/kmoe03/sens.py, not kept). Columns: block-out PCC vs golden,
norm share (vs the all-CPU block: flips / same-routing rel L2 / ratio / flipped-row ratio), then attn_norm vs the fp32
CPU norm of the same input (rel L2 / per-token ratio / worst row):
  fp32 CPU: PCC 0.999991. Noise: bf16 output share 6 / 0.00015 / [0.9997, 1.0004], norm 0.0017 / 0.0017; bf16
  input, weight and output 11 / 0.00021 / [0.9995, 1.0007], 0.0023 / 0.0025; 0.3% element noise 12 / 0.00027 /
  [0.9993, 1.0006] / [0.989, 1.011], 0.0030 / 0.0032; squares accumulated in bf16 41 / 0.00093, 0.0043 /
  [0.9868, 1.0196] / 0.0196; rsqrt 5e-3 row noise 43 / 0.00093, 0.0050 / [0.984, 1.019].
  Bugs: x1.003 24 / 0.00049 / 1.0038, norm ratio 1.003; x1.005 33 / 0.00082 / 1.0064; x1.01 79 / 0.0016; x1.02 159 /
  0.0032; x0.98 190 / 0.0033; eps 1.1e-5 183 / 0.0021, norm 0.017 / [0.963, 0.9995] / 0.037; eps 9e-6 174 / 0.0023,
  0.019 / 1.042; eps 0 / 1e-6 >= 1426 flips; mean subtraction 50 / 0.0015, 0.0135 / worst row 0.041; a norm over 4 TP
  shards 60 / 0.0014, 0.0145 / 0.047; w reversed 272 / 0.0042, 0.045; no weight, 1 + w, ffn_norm's weight >= 1887
  flips (PCC 0.9765..0.9953: the gate catches the first two only); row 100 x1.05 1 / 0.00019 / 1.0066, worst row
  0.050; last row zeroed 1 / 0.0 / flipped-row ratio 1.297, norm ratio 0; last 32 rows zeroed 32 / 0.0 / flipped
  0.417; last 32 columns zeroed 449 / 0.0075, 0.089. Every bug here passes the 0.98 gate except no weight and 1 + w.
Extra asserted checks (informational metrics, not in the runner's list; every limit is written `not x <= lim`, so a
NaN metric fails):
  - everything kda_moe swap 02 asserts, unchanged: block out vs golden (rel L2 <= 0.01; per-row ratio [0.985, 1.015]
    on rows with the golden's top-8, [0.9, 1.1] on flipped rows); block out vs the all-CPU block of the same golden
    `in` (now all three device steps' share: flips <= 64, same-routing rel L2 <= 0.004, ratio [0.985, 1.015],
    flipped rows [0.9, 1.1]); the attn_hc part checks; attn_collapse vs golden attn_in and vs the fp32 CPU collapse of
    its own inputs;
  - the collapse share now runs the device attn_norm on the CPU collapse, so only the collapse differs (limits as
    swap 02: flips <= 24, same-routing rel L2 <= 0.0015, ratio [0.995, 1.005], flipped rows [0.9, 1.1]);
  - attn_norm vs golden attn_norm (its input is the device attn_in): the component test's limits, rel L2 <= 0.01,
    per-token ratio [0.99, 1.01], worst row <= 0.015, coefficient <got, want> / <want, want> in [0.996, 1.004];
  - attn_norm vs the fp32 CPU norm of the attn_in it got: rel L2 <= 0.0045, per-token ratio [0.996, 1.004], worst row
    <= 0.008, coefficient [0.999, 1.001] (fails x1.003, eps off by 10%, mean subtraction, TP-shard norm, w reversed,
    bf16 square accumulation, rsqrt row noise, any zeroed or rescaled row or column; bf16 rounding and 0.3% element
    noise pass);
  - the attn_norm share of block out: block out vs the CPU block with the device attn_hc and attn_in and the CPU norm:
    flips <= 20, same-routing rel L2 <= 0.0004, ratio [0.998, 1.002], flipped rows [0.9, 1.1] (fails every bug above,
    including x1.003, row 100 x1.05 and a zeroed last row; bf16 rounding and 0.3% element noise pass).
Device (first run): PCC 0.999994, rel 0.0035; norm same-input 0.0017 / [0.9991, 1.0006] / 0.0019 / coefficient
0.99993; vs golden 0.0039 / [0.9945, 1.0004] / 0.0070 / 0.99872 (low on every row: the device attn_hc's per-column
bias, see swap 02, survives the norm's eps); norm share 8 flips / 0.00016 / [0.9995, 1.0004] / flipped [0.9982,
1.0269]; collapse share 0 flips / 0.0 (the device norm rounds its input to bf16, so the device and CPU collapse feed it
the same values); block out vs CPU 37 flips / 0.0012 / [0.9930, 1.0034].
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import run_block
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
BLOCK_TYPE = "kda_moe"
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.985, 1.015)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 64  # rows whose top-8 differs from the CPU block of the same `in`
CPU_MAX_REL_L2 = 0.004  # block out vs CPU block, same-routing rows
CPU_ROW_RATIO = (0.985, 1.015)
CPU_ROW_RATIO_FLIPPED = (0.9, 1.1)
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.02, "post": 6e-3, "comb": 0.02}
MAX_COL_REL_L2 = 0.05
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
COL_GOLD = {"rel": 0.01, "ratio": (0.985, 1.015), "row": 0.02}  # attn_collapse vs golden attn_in (device attn_hc in)
COL_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.0065}  # vs the fp32 CPU collapse of the same inputs
SHARE_MAX_FLIPS = 24  # collapse share: vs the CPU block with device attn_hc, CPU collapse, device attn_norm
SHARE_MAX_REL_L2 = 0.0015  # collapse share, same-routing rows
SHARE_RATIO = (0.995, 1.005)
SHARE_RATIO_FLIPPED = (0.9, 1.1)
NORM_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015, "coef": (0.996, 1.004)}  # vs golden (device attn_in in)
NORM_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.008, "coef": (0.999, 1.001)}  # vs fp32 CPU norm, same in
NORM_SHARE_MAX_FLIPS = 20  # norm share: vs the CPU block with device attn_hc + attn_in and the CPU norm
NORM_SHARE_MAX_REL_L2 = 0.0004
NORM_SHARE_RATIO = (0.998, 1.002)
NORM_SHARE_RATIO_FLIPPED = (0.9, 1.1)


def _f32(t):
    return t.float() if torch.is_tensor(t) and t.is_floating_point() else t


def _flipped(router, want_router):
    """Per token: top-8 selection (nonzero routing weights) differs."""
    return ((router.float() != 0) != (want_router.float() != 0)).any(dim=-1)


def _row_checks(tag, out, want, flipped_tok, same_lim, flipped_lim, max_rel=None):
    fails = []
    flipped = flipped_tok.repeat_interleave(N)
    ratio = out.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)
    same = ~flipped
    metrics.record(f"flipped_rows_swap_out_{tag}", int(flipped_tok.sum()))
    if same.any():
        r = ratio[same]
        rmin, rmax = r.min().item(), r.max().item()
        rel = ((out[same] - want[same]).norm() / want[same].norm().clamp_min(1e-12)).item()
        metrics.record(f"rel_l2_same_routing_swap_out_{tag}", rel)
        metrics.record(f"row_norm_ratio_min_swap_out_{tag}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_out_{tag}", rmax)
        print(
            f"block out vs {tag}: {int(flipped_tok.sum())} tokens with another top-8; same-routing rel_l2={rel:.5f}"
            f" (<= {max_rel}) ratio=[{rmin:.4f}, {rmax:.4f}] (limit {same_lim})"
        )
        if not (same_lim[0] <= rmin and rmax <= same_lim[1]):
            fails.append(f"block out vs {tag}: same-routing per-row ratio [{rmin:.4f}, {rmax:.4f}] outside {same_lim}")
        if max_rel is not None and not rel <= max_rel:
            fails.append(f"block out vs {tag}: same-routing rel L2 {rel:.5f} > {max_rel}")
    else:
        fails.append(f"block out vs {tag}: every token has another top-8")
    if flipped.any() and flipped_lim is not None:
        r = ratio[flipped]
        rmin, rmax = r.min().item(), r.max().item()
        print(f"block out vs {tag}: flipped rows ratio=[{rmin:.4f}, {rmax:.4f}] (limit {flipped_lim})")
        if not (flipped_lim[0] <= rmin and rmax <= flipped_lim[1]):
            fails.append(f"block out vs {tag}: flipped-row ratio [{rmin:.4f}, {rmax:.4f}] outside {flipped_lim}")
    return fails


def _out_checks(step, tag, got, want, lim):
    if got.numel() != want.numel():
        return [f"{step} {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"{step} {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_{step}_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_{step}_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_{step}_{tag}", hi)
    metrics.record(f"row_rel_l2_max_swap_{step}_{tag}", row)
    msg = (
        f"{step} {tag}: rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio [{lo:.4f}, {hi:.4f}] (in {lim['ratio']})"
        f" worst row {row:.5f} (<= {lim['row']})"
    )
    fails = []
    if "coef" in lim:  # <got, want> / <want, want>: a uniform scale error
        coef = ((got * w).sum() / (w * w).sum()).item()
        metrics.record(f"coef_swap_{step}_{tag}", coef)
        msg += f" coefficient {coef:.5f} (in {lim['coef']})"
        if not (lim["coef"][0] <= coef <= lim["coef"][1]):
            fails.append(f"{step} {tag} coefficient {coef:.5f} outside {lim['coef']}")
    print(msg)
    if not rel <= lim["rel"]:  # NaN fails
        fails.append(f"{step} {tag} rel L2 {rel:.4f} > {lim['rel']}")
    if not (lim["ratio"][0] <= lo and hi <= lim["ratio"][1]):
        fails.append(f"{step} {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {lim['ratio']}")
    if not row <= lim["row"]:
        fails.append(f"{step} {tag} worst row rel L2 {row:.4f} > {lim['row']}")
    return fails


def _hc_checks(got, want):
    fails = []
    if got.numel() != want.numel():
        return [f"attn_hc: shape {tuple(got.shape)} vs golden {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return ["attn_hc: non-finite output"]
    for name, sl in PARTS.items():
        a, b = got[:, sl], w[:, sl]
        rel = ((a - b).norm() / b.norm()).item()
        mx = (a - b).abs().max().item()
        metrics.record(f"rel_l2_swap_attn_hc_{name}", rel)
        metrics.record(f"max_abs_swap_attn_hc_{name}", mx)
        print(f"attn_hc {name}: rel_l2={rel:.5f} (<= {MAX_REL_L2[name]}) max_abs={mx:.2e} (<= {MAX_ABS[name]})")
        if not rel <= MAX_REL_L2[name]:
            fails.append(f"attn_hc {name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if not mx <= MAX_ABS[name]:
            fails.append(f"attn_hc {name} max abs {mx:.3g} > {MAX_ABS[name]}")
    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record("worst_col_rel_l2_swap_attn_hc", worst)
    print(f"attn_hc worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {MAX_COL_REL_L2})")
    if not worst <= MAX_COL_REL_L2:
        fails.append(f"attn_hc column {col_rel.argmax().item()} rel L2 {worst:.4f} > {MAX_COL_REL_L2}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record("comb_col_sum_err_swap_attn_hc", col_err)
    print(f"attn_hc comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if not col_err <= COL_SUM_TOL:
        fails.append(f"attn_hc comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"attn_hc pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"attn_hc post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("attn_hc negative comb entries")
    return fails


def _share(tag, steps, ref, layer, g, c, gl, seen, out, overrides, max_flips, ratio, ratio_flipped, max_rel):
    """Block out vs a CPU block that differs from the swapped one in a single step."""
    blk = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: blk.__setitem__(n, t),
        overrides=overrides,
    )
    flips = _flipped(seen["router"], blk["router"])
    n = int(flips.sum())
    fails = []
    if not n <= max_flips:
        fails.append(f"{n} tokens route to another top-8 than the {tag} block (> {max_flips})")
    return fails + _row_checks(tag, out, blk["out"].float().reshape(out.shape), flips, ratio, ratio_flipped, max_rel)


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides, muts = {}, {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        muts[name] = mut
        overrides[name] = lambda ctx, *x, mut=mut: _f32(mut(ctx, dctx, *x))
    gl = g.layer(c, layer)
    seen = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        rctx,
        gl["in"].float(),
        rec=lambda n, t: seen.__setitem__(n, t),
        overrides=overrides,
    )
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    thr = threshold(S, "block") if THRESHOLD is None else THRESHOLD
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", thr)

    failures = [] if ok else [f"pcc_swap_out below {thr}"]
    out, w = seen["out"].float().reshape(gl["out"].shape), gl["out"].float()
    out_rel = ((out - w).norm() / w.norm()).item()
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"block out vs golden: rel_l2={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not torch.isfinite(out).all():
        failures.append("block out non-finite")
    if not out_rel <= OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2}")
    failures += _row_checks(
        "golden", out, w, _flipped(seen["router"], gl["router"]), OUT_ROW_RATIO_SAME, OUT_ROW_RATIO_FLIPPED
    )
    hc_dev = seen["attn_hc"].float()
    in_dev = seen["attn_in"].float().reshape(gl["attn_in"].shape)
    blk = (steps, ref, layer, g, c, gl, seen, out)

    # All three device steps' share: the all-CPU block of the same golden `in`.
    failures += _share("cpu", *blk, {}, CPU_MAX_FLIPS, CPU_ROW_RATIO, CPU_ROW_RATIO_FLIPPED, CPU_MAX_REL_L2)

    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])

    # attn_collapse vs golden (its attn_hc input is the device one), and vs the CPU collapse of the same inputs.
    failures += _out_checks("attn_collapse", "vs_golden", seen["attn_in"], gl["attn_in"], COL_GOLD)
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), hc_dev)
    failures += _out_checks("attn_collapse", "vs_cpu_same_input", seen["attn_in"], cpu_in, COL_SAME)

    # The attn_collapse share alone: the CPU block with the device attn_hc, the CPU collapse and the device norm.
    dev_norm = lambda ctx, x: _f32(muts["attn_norm"](ctx, dctx, x))  # noqa: E731
    failures += _share(
        "collapse_share",
        *blk,
        {"attn_hc": lambda ctx, x: hc_dev, "attn_norm": dev_norm},
        SHARE_MAX_FLIPS,
        SHARE_RATIO,
        SHARE_RATIO_FLIPPED,
        SHARE_MAX_REL_L2,
    )

    # attn_norm vs golden (its attn_in input is the device one), and vs the fp32 CPU norm of the same input.
    failures += _out_checks("attn_norm", "vs_golden", seen["attn_norm"], gl["attn_norm"], NORM_GOLD)
    cpu_norm = ref.component(layer, "attn_norm")(rctx, in_dev)
    failures += _out_checks("attn_norm", "vs_cpu_same_input", seen["attn_norm"], cpu_norm, NORM_SAME)

    # The attn_norm share alone: the CPU block with the device attn_hc and attn_in and the CPU norm.
    failures += _share(
        "norm_share",
        *blk,
        {"attn_hc": lambda ctx, x: hc_dev, "attn_collapse": lambda ctx, x, h: in_dev},
        NORM_SHARE_MAX_FLIPS,
        NORM_SHARE_RATIO,
        NORM_SHARE_RATIO_FLIPPED,
        NORM_SHARE_MAX_REL_L2,
    )
    assert not failures, "; ".join(failures)

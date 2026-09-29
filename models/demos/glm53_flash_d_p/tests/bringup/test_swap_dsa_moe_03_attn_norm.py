# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 3: block type dsa_moe (layer 3) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 3 (dsa_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm

Reviewed (S.dsa_moe.03.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attn_norm is input_layernorm, w * x * rsqrt(mean(x^2) + 1e-5), on attn_in
[S, H]; its output feeds q_a, the indexer and attention, whose own norms (q_a_layernorm, kv_a_layernorm) remove a
row scale, and attention reaches block out only through post (<= 0.024 at layer 3), so block out hardly sees norm bugs.
Measured on the CPU (golden s4096 chunk 1, CPU attn_hc and collapse, attn_norm output perturbed, rest CPU). Columns:
PCC of block out vs golden, then the norm share of block out (vs the all-CPU block: flips / same-routing rel L2 /
ratio), then attn_norm vs the fp32 CPU norm of the same input (rel L2 / per-token ratio / worst row):
  fp32 CPU: PCC 0.999996, 36 flips vs golden. Noise: bf16 output share 1 / 0.0002 / [0.9995, 1.0006], norm 0.0017 /
  [0.9999, 1.0002] / 0.0017; bf16 input, weight and output 5 / 0.0002, 0.0023 / 0.0025; 0.3% element noise 7 /
  0.0003 / [0.9986, 1.0007], 0.0034 / 0.0036; squares accumulated in bf16 9 / 0.0002, 0.0045 / [0.990, 1.025] /
  0.025; rsqrt 5e-3 row noise 5 / 0.0002, 0.0053 / [0.984, 1.019] / 0.019.
  Bugs: x1.005 share 2 / 0.0001, norm ratio 1.0050; x1.01 / x1.02 / x0.98 share <= 11 / 0.0003 (invisible in block
  out), norm ratio 1.01 / 1.02 / 0.98; eps 1.1e-5 13 / 0.0003, norm 0.0205 / [0.961, 0.995]; eps 9e-6 8 / 0.0003,
  0.022 / 1.044; eps 1.2e-5 24 / 0.0005, 0.040; eps 0 84 / 0.0023, 0.43; eps 1e-6 71 / 0.0021, 0.34; mean subtraction
  29 / 0.0011, 0.0139 / [0.9995, 1.0001] / worst row 0.043; w reversed 410 / 0.0122; no weight, 1 + w, ffn_norm's
  weight 367..385 / 0.011 (PCC >= 0.99989: the gate passes all of them); row 100 x1.05 share 0 / 0.0000, norm worst
  row 0.050; last row zeroed share 1 / 0.0000, norm ratio 0; last 32 rows zeroed 29 / 0.0079 / [0.885, 1.093]; last
  32 columns zeroed 234 / 0.0065, norm 0.086.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 02 asserts: block out vs golden (rel L2 <= 0.01; per-row ratio [0.975, 1.025] on rows with the
    golden's top-8, [0.9, 1.1] on flipped rows); block out vs the all-CPU block of the same golden `in` (now all three
    device steps' share: flips <= 64, same-routing rel L2 <= 0.005, ratio [0.98, 1.02]); the attn_hc part checks;
    attn_collapse vs golden attn_in and vs the fp32 CPU collapse of its own inputs;
  - the collapse share now runs the device attn_norm on the CPU collapse, so only the collapse differs (limits as
    swap 02: flips <= 24, same-routing rel L2 <= 0.0015, ratio [0.995, 1.005]);
  - attn_norm vs golden attn_norm (its input is the device attn_in): the component test's limits, rel L2 <= 0.01,
    per-token ratio [0.99, 1.01], worst row <= 0.015;
  - attn_norm vs the fp32 CPU norm of the attn_in it got: rel L2 <= 0.0045, per-token ratio [0.996, 1.004], worst row
    <= 0.008 (fails x1.005, eps off by 10%, mean subtraction, bf16 square accumulation, rsqrt row noise, any zeroed
    or rescaled row; bf16 rounding and 0.3% element noise pass);
  - the attn_norm share of block out: block out vs the CPU block with the device attn_hc and attn_in and the CPU norm:
    flips <= 20, same-routing rel L2 <= 0.001, ratio [0.995, 1.005] (catches eps 0 / 1e-6, wrong or missing weight,
    mean subtraction, zeroed rows or columns).
Device (first run): norm same-input 0.0017 / [0.9991, 1.0005] / 0.0019; vs golden 0.0043 / [0.9985, 1.0020] /
0.0068; norm share 4 flips / 0.00015 / [0.9997, 1.0007]; collapse share 0 flips / 0.00002; block out vs CPU 29 flips /
0.0020 / [0.9924, 1.0075]; PCC 0.999994.
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
BLOCK_TYPE = "dsa_moe"
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
N = 4  # hc_mult
OUT_MAX_REL_L2 = 0.01  # block out vs golden
OUT_ROW_RATIO_SAME = (0.975, 1.025)  # per-row ||got|| / ||want||, rows whose top-8 equals the golden's
OUT_ROW_RATIO_FLIPPED = (0.9, 1.1)  # rows with another top-8 (a whole expert changes)
CPU_MAX_FLIPS = 64  # rows whose top-8 differs from the CPU block of the same `in`
CPU_MAX_REL_L2 = 0.005  # block out vs CPU block, same-routing rows
CPU_ROW_RATIO = (0.98, 1.02)
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.01, "post": 0.01, "comb": 0.01}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.02, "post": 5e-4, "comb": 0.02}
MAX_COL_REL_L2 = 0.07
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|
COL_GOLD = {"rel": 0.01, "ratio": (0.985, 1.015), "row": 0.02}  # attn_collapse vs golden attn_in (device attn_hc in)
COL_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.0065}  # vs the fp32 CPU collapse of the same inputs
SHARE_MAX_FLIPS = 24  # collapse share: rows whose top-8 differs from the CPU block with device attn_hc, CPU collapse
SHARE_MAX_REL_L2 = 0.0015  # collapse share, same-routing rows
SHARE_RATIO = (0.995, 1.005)
NORM_GOLD = {"rel": 0.01, "ratio": (0.99, 1.01), "row": 0.015}  # attn_norm vs golden (device attn_in in)
NORM_SAME = {"rel": 0.0045, "ratio": (0.996, 1.004), "row": 0.008}  # vs the fp32 CPU norm of the same input
NORM_SHARE_MAX_FLIPS = 20  # norm share: rows whose top-8 differs from the CPU block with device hc + attn_in
NORM_SHARE_MAX_REL_L2 = 0.001
NORM_SHARE_RATIO = (0.995, 1.005)


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
            f" ratio=[{rmin:.4f}, {rmax:.4f}] (limit {same_lim})"
        )
        if not (same_lim[0] <= rmin and rmax <= same_lim[1]):
            fails.append(f"block out vs {tag}: same-routing per-row ratio [{rmin:.4f}, {rmax:.4f}] outside {same_lim}")
        if max_rel is not None and rel > max_rel:
            fails.append(f"block out vs {tag}: same-routing rel L2 {rel:.4f} > {max_rel}")
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
    print(
        f"{step} {tag}: rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio [{lo:.4f}, {hi:.4f}] (in {lim['ratio']})"
        f" worst row {row:.5f} (<= {lim['row']})"
    )
    fails = []
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
        if rel > MAX_REL_L2[name]:
            fails.append(f"attn_hc {name} rel L2 {rel:.4f} > {MAX_REL_L2[name]}")
        if mx > MAX_ABS[name]:
            fails.append(f"attn_hc {name} max abs {mx:.3g} > {MAX_ABS[name]}")
    col_rel = (got - w).norm(dim=0) / w.norm(dim=0)
    worst = col_rel.max().item()
    metrics.record("worst_col_rel_l2_swap_attn_hc", worst)
    print(f"attn_hc worst column rel L2 {worst:.4f} at column {col_rel.argmax().item()} (<= {MAX_COL_REL_L2})")
    if worst > MAX_COL_REL_L2:
        fails.append(f"attn_hc column {col_rel.argmax().item()} rel L2 {worst:.4f} > {MAX_COL_REL_L2}")
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col = comb.sum(dim=-2)
    col_err = (col - 1).abs().max().item()
    metrics.record("comb_col_sum_err_swap_attn_hc", col_err)
    print(f"attn_hc comb column sums [{col.min().item():.5f}, {col.max().item():.5f}] (|. - 1| <= {COL_SUM_TOL})")
    if col_err > COL_SUM_TOL:
        fails.append(f"attn_hc comb column sums off 1 by {col_err:.4f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    if not ((pre > 0).all() and (pre <= 1 + 1e-3).all()):
        fails.append(f"attn_hc pre outside (0, 1]: [{pre.min().item():.3e}, {pre.max().item():.4f}]")
    if not ((post >= 0).all() and (post <= 2 + 1e-3).all()):
        fails.append(f"attn_hc post outside [0, 2]: [{post.min().item():.3e}, {post.max().item():.4f}]")
    if (comb < 0).any():
        fails.append("attn_hc negative comb entries")
    return fails


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
    if out_rel > OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2}")
    failures += _row_checks(
        "golden", out, w, _flipped(seen["router"], gl["router"]), OUT_ROW_RATIO_SAME, OUT_ROW_RATIO_FLIPPED
    )

    # Both device steps' share: the all-CPU block of the same golden `in`.
    cpu = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: cpu.__setitem__(n, t),
    )
    flips = _flipped(seen["router"], cpu["router"])
    n_flips = int(flips.sum())
    if n_flips > CPU_MAX_FLIPS:
        failures.append(f"{n_flips} tokens route to another top-8 than the CPU block (> {CPU_MAX_FLIPS})")
    cpu_out = cpu["out"].float().reshape(w.shape)
    failures += _row_checks("cpu", out, cpu_out, flips, CPU_ROW_RATIO, None, CPU_MAX_REL_L2)

    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])

    # attn_collapse vs golden (its attn_hc input is the device one), and vs the CPU collapse of the same inputs.
    hc_dev = seen["attn_hc"].float()
    failures += _out_checks("attn_collapse", "vs_golden", seen["attn_in"], gl["attn_in"], COL_GOLD)
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), hc_dev)
    failures += _out_checks("attn_collapse", "vs_cpu_same_input", seen["attn_in"], cpu_in, COL_SAME)

    # The attn_collapse share alone: the CPU block with the device attn_hc, the CPU collapse and the device norm.
    share = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: share.__setitem__(n, t),
        overrides={"attn_hc": lambda ctx, x: hc_dev, "attn_norm": lambda ctx, x: _f32(muts["attn_norm"](ctx, dctx, x))},
    )
    s_flips = _flipped(seen["router"], share["router"])
    n_s = int(s_flips.sum())
    if n_s > SHARE_MAX_FLIPS:
        failures.append(f"{n_s} tokens route to another top-8 than the collapse-share block (> {SHARE_MAX_FLIPS})")
    share_out = share["out"].float().reshape(w.shape)
    failures += _row_checks("collapse_share", out, share_out, s_flips, SHARE_RATIO, None, SHARE_MAX_REL_L2)

    # attn_norm vs golden (its attn_in input is the device one), and vs the CPU norm of the same input.
    in_dev = seen["attn_in"].float().reshape(gl["attn_in"].shape)
    failures += _out_checks("attn_norm", "vs_golden", seen["attn_norm"], gl["attn_norm"], NORM_GOLD)
    cpu_norm = ref.component(layer, "attn_norm")(rctx, in_dev)
    failures += _out_checks("attn_norm", "vs_cpu_same_input", seen["attn_norm"], cpu_norm, NORM_SAME)

    # The attn_norm share alone: the CPU block with the device attn_hc and attn_in and the CPU norm.
    nshare = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: nshare.__setitem__(n, t),
        overrides={"attn_hc": lambda ctx, x: hc_dev, "attn_collapse": lambda ctx, x, h: in_dev},
    )
    n_flips_n = _flipped(seen["router"], nshare["router"])
    n_n = int(n_flips_n.sum())
    if n_n > NORM_SHARE_MAX_FLIPS:
        failures.append(f"{n_n} tokens route to another top-8 than the norm-share block (> {NORM_SHARE_MAX_FLIPS})")
    nshare_out = nshare["out"].float().reshape(w.shape)
    failures += _row_checks("norm_share", out, nshare_out, n_flips_n, NORM_SHARE_RATIO, None, NORM_SHARE_MAX_REL_L2)
    assert not failures, "; ".join(failures)

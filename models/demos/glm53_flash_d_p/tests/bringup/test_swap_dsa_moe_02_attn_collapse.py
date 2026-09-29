# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type dsa_moe (layer 3) with attn_collapse swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 3 (dsa_moe) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse

Reviewed (S.dsa_moe.02.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). attn_collapse is attn_in [S, H] = sum_n pre[:, n] * in[:, n], pre =
attn_hc[:, 0:4]. attn_norm follows it and removes a row scale, so block out hardly sees collapse bugs. Measured on
the CPU (golden s4096 chunk 1, CPU attn_hc, attn_collapse output perturbed, rest CPU). Columns: collapse vs the fp32
CPU collapse of the same inputs rel L2 / per-token ratio / worst row, then the collapse share of block out (vs the
CPU block with the same attn_hc and the CPU collapse: flips / same-routing rel L2 / ratio):
  bf16 output 0.0017 / [0.9999, 1.0002] / 0.0018, 4 / 0.0001 / [0.9996, 1.0003]; bf16 products and accumulation
  0.0031 / [0.9970, 1.0026] / 0.0042, 11 / 0.0007 / [0.9970, 1.0027]; 0.3% element noise 0.0030 / 0.0032, 4 / 0.0002.
  Bugs: x1.005 0.0050 / 1.0050, share 1 / 0.0001; x1.01 0.0100 (share 2 / 0.0001: attn_norm removes it); x0.99 the
  same; pre column 0 x1.01 0.0062 / [1.0012, 1.0069] / 0.0081, share 40 / 0.0026 / 0.9939; pre normalized to sum 1
  0.069 / [0.71, 1.29]; one row's pre reversed 0.30 / ratio 11.95, share ratio 1.0059; last row zeroed 0.0124 / ratio
  0; last 32 rows zeroed 0.114, share 0.0079 / [0.885, 1.093]; last 32 columns zeroed 0.087 / worst row 0.13, share
  235 flips; pre reversed 10.1 and stream-major rows 6.7 (PCC 0.973 / 0.926, the only ones the gate catches).
  With 0.3% noise on attn_hc (about the device's part error), the collapse vs golden attn_in scores 0.0036 /
  [0.9934, 1.0075] / worst row 0.0084: the component test's worst-row limit 0.008 does not fit a device attn_hc input.
Extra asserted checks (informational metrics, not in the runner's list):
  - everything swap 01 asserts, unchanged: block out vs golden (rel L2 <= 0.01; per-row ratio [0.975, 1.025] on rows
    with the golden's top-8, [0.9, 1.1] on flipped rows); block out vs the all-CPU block of the same golden `in` (now
    both device steps' share: flips <= 64, same-routing rel L2 <= 0.005, ratio [0.98, 1.02]); the attn_hc part checks;
  - attn_collapse vs golden attn_in: rel L2 <= 0.01, per-token ratio [0.985, 1.015], worst row <= 0.02;
  - attn_collapse vs the fp32 CPU collapse of its own inputs (golden `in`, device attn_hc): rel L2 <= 0.0045,
    per-token ratio [0.996, 1.004], worst row <= 0.0065 (x1.005 and pre column 0 x1.01 fail; bf16 accumulation passes);
  - the collapse share of block out: block out vs the CPU block with the device attn_hc and the CPU collapse: flips
    <= 24, same-routing rel L2 <= 0.0015, ratio [0.995, 1.005] (catches pre column 0 x1.01, one row's pre reversed,
    zeroed rows or columns).
Device (first run): collapse same-input 0.0017 / [0.9998, 1.0001] / 0.0017; vs golden 0.0039 / [0.9979, 1.0033] /
0.0070; share 4 flips / 0.0002 / [0.9995, 1.0004]; block out vs CPU 27 flips / 0.0020 / [0.9924, 1.0075].
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
SWAPPED = ["attn_hc", "attn_collapse"]
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


def _collapse_checks(tag, got, want, lim):
    if got.numel() != want.numel():
        return [f"attn_collapse {tag}: {got.numel()} elements, want {tuple(want.shape)}"]
    got, w = got.float().reshape(want.shape), want.float()
    if not torch.isfinite(got).all():
        return [f"attn_collapse {tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_swap_attn_collapse_{tag}", rel)
    metrics.record(f"norm_ratio_min_swap_attn_collapse_{tag}", lo)
    metrics.record(f"norm_ratio_max_swap_attn_collapse_{tag}", hi)
    metrics.record(f"row_rel_l2_max_swap_attn_collapse_{tag}", row)
    print(
        f"attn_collapse {tag}: rel_l2={rel:.5f} (<= {lim['rel']}) norm ratio [{lo:.4f}, {hi:.4f}] (in {lim['ratio']})"
        f" worst row {row:.5f} (<= {lim['row']})"
    )
    fails = []
    if not rel <= lim["rel"]:  # NaN fails
        fails.append(f"attn_collapse {tag} rel L2 {rel:.4f} > {lim['rel']}")
    if not (lim["ratio"][0] <= lo and hi <= lim["ratio"][1]):
        fails.append(f"attn_collapse {tag} norm ratio [{lo:.4f}, {hi:.4f}] outside {lim['ratio']}")
    if not row <= lim["row"]:
        fails.append(f"attn_collapse {tag} worst row rel L2 {row:.4f} > {lim['row']}")
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
    overrides = {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
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
    failures += _collapse_checks("vs_golden", seen["attn_in"], gl["attn_in"], COL_GOLD)
    cpu_in = ref.component(layer, "attn_collapse")(rctx, gl["in"].float(), hc_dev)
    failures += _collapse_checks("vs_cpu_same_input", seen["attn_in"], cpu_in, COL_SAME)

    # The attn_collapse share alone: the CPU block with the device attn_hc and the CPU collapse.
    share = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        reference_ctx(ref, layer, g, c),
        gl["in"].float(),
        rec=lambda n, t: share.__setitem__(n, t),
        overrides={"attn_hc": lambda ctx, x: hc_dev},
    )
    s_flips = _flipped(seen["router"], share["router"])
    n_s = int(s_flips.sum())
    if n_s > SHARE_MAX_FLIPS:
        failures.append(f"{n_s} tokens route to another top-8 than the collapse-share block (> {SHARE_MAX_FLIPS})")
    share_out = share["out"].float().reshape(w.shape)
    failures += _row_checks("collapse_share", out, share_out, s_flips, SHARE_RATIO, None, SHARE_MAX_REL_L2)
    assert not failures, "; ".join(failures)

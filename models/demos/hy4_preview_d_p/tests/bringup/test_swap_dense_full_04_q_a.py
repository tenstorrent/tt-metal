# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 4: block type dense_full (layer 0) with q_a swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense_full) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a

Reviewed (S.dense_full.04.test.1). The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). q_resid [S, 2048] = q_a_layernorm(q_a_proj(attn_norm)) (eps 1e-6) feeds only
the indexer queries (top-2048) and the attention queries (q_b); the block output is the residual stream plus the
attention, so q_a bugs move it very little. Measured on the CPU (golden s4096 chunk 1, 2048 rows, 2048-key prefix;
q_a replaced by mutations of the fp32 step, every other step the fp32 CPU reference):

    variant                         q_resid rel / row ratio / worst row | topk ov | attn_out rel/row | h_mid rel/row  | out PCC   rel
    fp32 reference                  0.0017 / [0.9996, 1.0004] / 0.0019  | 0.9993  | 0.0017 / 0.002   | 0.0017 / 0.0017 | 1.000000  0.0017
    bf16 x / W, bf16 pre and out    0.0029 / [0.9994, 1.0006] / 0.0038  | 0.9992  | 0.0017 / 0.002   | 0.0017 / 0.0017 | 1.000000  0.0017
    eps 1e-5                        0.0017 / [0.9996, 1.0004] / 0.0019  | 0.9993  | 0.0017 / 0.002   | 0.0017 / 0.0017 | 1.000000  0.0017 (passes)
    x 1.01                          0.0101 / [1.0096, 1.0104] / 0.0105  | 0.9993  | 0.0024 / 0.004   | 0.0024 / 0.0039 | 1.000000  0.0030 (passes)
    x 1.02                          0.0201 / [1.0196, 1.0204] / 0.0205  | 0.9993  | 0.0039 / 0.007   | 0.0037 / 0.0072 | 1.000000  0.0053 (passes)
    RMS over half the columns       0.0348 / [0.9656, 1.0720] / 0.0720  | 0.9993  | 0.0056 / 0.016   | 0.0043 / 0.0194 | 1.000000  0.0062 (passes)
    last row zeroed                 0.0218 / [0.0,    1.0004] / 1.0     | 0.9991  | 0.0053 / 0.205   | 0.0066 / 0.183  | 1.000000  0.0093 (passes)
    last tile row (32) zeroed       0.1249 / [0.0,    1.0004] / 1.0     | 0.9951  | 0.0213 / 0.259   | 0.0223 / 0.286  | 0.999611  0.0311 (passes)
    only one K half (no reduce)     0.411  / [0.9544, 1.0139] / 0.580   | 0.9807  | 0.0288 / 0.106   | 0.0326 / 0.0853 | 0.998973  0.0473 (passes)
    norm w halves swapped           0.263  / [0.8549, 0.9819] / 0.383   | 0.9748  | 0.0402 / 0.063   | 0.0383 / 0.0698 | 0.998573  0.0556 (passes)
    norm per K partial, then sum    0.807  / [1.6469, 1.8736] / 0.874   | 0.9991  | 0.0945 / 0.143   | 0.0794 / 0.181  | 0.993518  0.1194 (passes)
    row halves swapped (SP order)   0.929  / [0.9175, 1.0905] / 1.24    | 0.9263  | 0.145  / 0.379   | 0.110  / 0.491  | 0.988632  0.1560 (passes)
    no norm weight                  3.37   / [4.1187, 4.7294] / 3.76    | 0.9750  | 0.189  / 0.320   | 0.172  / 0.293  | 0.971434  0.2533
    zero stub                       1.0    / [0.0,    0.0]    / 1.0     | 0.7754  | 0.166  / 0.300   | 0.162  / 0.329  | 0.974783  0.2322

"(passes)" = passes the 0.98 out gate. Only "no norm weight" and the zero stub fail it. So the test also asserts
(informational metrics):
  - attn_hc (gates), attn_x and attn_norm vs golden at the swap 01 / 02 / 03 limits, and the attn_norm module on the
    device attn_x x 0.1 vs the CPU step (swap 03's eps check), so upstream regressions still fail;
  - q_resid (the swapped step) vs golden at the component limits (test_c_dense_full_q_a.py): finite, shape, rel L2
    <= 0.008, every row's norm ratio in [0.994, 1.006], worst row rel L2 <= 0.015 (catches x 1.01, RMS subsets,
    zeroed rows, a missing or per-partial reduce, the norm-weight bugs and everything worse);
  - q_resid vs the CPU q_a on the same input (the device attn_norm): rel L2 <= 0.008, worst row <= 0.015 (the step's
    own error, apart from the device attn_norm error);
  - the q_a module once more on the device attn_norm x 0.01 (bf16) vs the CPU step on the same input: rel L2
    <= 0.01, worst row <= 0.02. The golden cannot see eps (pre-norm mean square ~1e5 x eps); at x 0.01 the component
    test measured eps 1e-5 0.173, eps 0 0.027, eps 2e-6 0.025 vs bf16 noise 0.0023;
  - indexer top-k overlap vs golden >= 0.995 (reference 0.9993: near-tie flips); attn_out rel L2 <= 0.01, worst
    row <= 0.05; h_mid rel L2 <= 0.01, worst row <= 0.05; block out finite, rel L2 <= 0.01 (as swaps 01-03).
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
BLOCK_TYPE = "dense_full"
SWAPPED = ["attn_hc", "attn_hc_pre", "attn_norm", "q_a"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_ABS_ERR = 0.015  # attn_hc, per column: max |got - want|
X_MAX_REL_L2 = 0.004  # attn_x vs golden
X_RATIO = (0.995, 1.005)  # attn_x per-row ||got|| / ||want||
X_MAX_ROW_REL_L2 = 0.01  # attn_x worst row
N_MAX_REL_L2 = 0.008  # attn_norm vs golden, and vs the CPU attn_norm on the same input
N_RATIO = (0.993, 1.007)  # attn_norm per-row norm ratio vs golden
N_MAX_ROW_REL_L2 = 0.015  # attn_norm worst row, vs golden and vs the CPU step on the same input
SYN_SCALE = 0.1  # eps check: the device attn_x x SYN_SCALE (bf16) through the module vs the CPU step
SYN_MAX_REL_L2 = 0.01
SYN_MAX_ROW_REL_L2 = 0.02
Q_MAX_REL_L2 = 0.008  # q_resid vs golden, and vs the CPU q_a on the same input
Q_RATIO = (0.994, 1.006)  # q_resid per-row norm ratio vs golden
Q_MAX_ROW_REL_L2 = 0.015  # q_resid worst row, vs golden and vs the CPU step on the same input
Q_SYN_SCALE = 0.01  # eps check: the device attn_norm x Q_SYN_SCALE (bf16) through the q_a module vs the CPU step
Q_SYN_MAX_REL_L2 = 0.01
Q_SYN_MAX_ROW_REL_L2 = 0.02
TOPK_MIN_OVERLAP = 0.995  # indexer top-k selection vs golden (mean per-row set overlap, -1 pads ignored)
ATT_MAX_REL_L2 = 0.01  # attn_out, whole tensor
ATT_MAX_ROW_REL_L2 = 0.05  # attn_out, worst token row
MID_MAX_REL_L2 = 0.01  # h_mid, whole tensor
MID_MAX_ROW_REL_L2 = 0.05  # h_mid, worst token row
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor


def _errors(got, want):
    """rel L2, per-row norm ratio (min, max), worst row rel L2."""
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    wn = want.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    return rel, ratio.min().item(), ratio.max().item(), ((got - want).norm(dim=-1) / wn).max().item()


def _topk_overlap(got, want):
    """Mean per-row |got & want| / |want| over the valid (non -1) indices."""
    got, want = got.reshape(want.shape).long(), want.long()
    tot = 0.0
    for a, b in zip(got, want):
        b = b[b >= 0]
        if b.numel():
            tot += torch.isin(b, a[a >= 0]).float().mean().item()
        else:
            tot += 1.0
    return tot / want.shape[0]


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
        assert not getattr(mut, "cpu_bridge", False), f"device_component returned a CPU bridge for {name}"
        muts[name] = mut
        overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
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

    # Extra checks (see the module docstring). Recorded as informational metrics, asserted here.
    failures = [] if ok else [f"pcc_swap_out below {thr}"]

    def finite_shape(n):
        got, want = seen[n], gl[n]
        if got.numel() != want.numel() or got.shape[-1] != want.shape[-1]:
            failures.append(f"{n}: shape {tuple(got.shape)} vs golden {tuple(want.shape)}")
            return False
        if not torch.isfinite(got.float()).all():
            failures.append(f"{n}: non-finite")
            return False
        return True

    def rel_row(n, max_rel, max_row, ratio=None, what="vs golden", want=None):
        rel, rmin, rmax, row = _errors(seen[n], gl[n] if want is None else want)
        tag = n if what == "vs golden" else f"{n}_{what.replace(' ', '_')}"
        metrics.record(f"rel_l2_swap_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_{tag}", row)
        msg = f"{n} {what}: rel_l2={rel:.6f} (<= {max_rel})"
        if ratio is not None:
            msg += f" row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(ratio)})"
        if max_row is not None:
            msg += f" worst_row_rel_l2={row:.5f} (<= {max_row})"
        print(msg)
        if rel > max_rel:
            failures.append(f"{n} {what}: rel L2 {rel:.5f} > {max_rel}")
        if ratio is not None and not (ratio[0] <= rmin and rmax <= ratio[1]):
            failures.append(f"{n} {what}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if max_row is not None and row > max_row:
            failures.append(f"{n} {what}: worst row rel L2 {row:.5f} > {max_row}")

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7).
    if finite_shape("attn_hc"):
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _errors(got, want)[0]
        col_err = (got - want).abs().amax(dim=0)
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_abs_err_swap_attn_hc", col_err.max().item())
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) max_abs_err per column="
            f"{[round(v, 5) for v in col_err.tolist()]} (<= {GATES_MAX_ABS_ERR})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_err.tolist()) if v > GATES_MAX_ABS_ERR]
        if bad:
            failures.append(f"attn_hc: max abs error > {GATES_MAX_ABS_ERR} in columns {bad}")

    # attn_x vs golden.
    x_ok = finite_shape("attn_x")
    if x_ok:
        rel_row("attn_x", X_MAX_REL_L2, X_MAX_ROW_REL_L2, X_RATIO)

    # attn_norm (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    n_ok = finite_shape("attn_norm")
    if n_ok:
        rel_row("attn_norm", N_MAX_REL_L2, N_MAX_ROW_REL_L2, N_RATIO)
        if x_ok:
            cpu = ref.component(layer, "attn_norm")
            x = seen["attn_x"].float().reshape(gl["attn_x"].shape)
            rel_row("attn_norm", N_MAX_REL_L2, N_MAX_ROW_REL_L2, what="vs cpu", want=cpu(rctx, x).float())

            xs = (x * SYN_SCALE).bfloat16().float()
            syn_want = cpu(rctx, xs).float()
            syn_out = muts["attn_norm"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"attn_norm scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, ymin, ymax, yrow = _errors(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_attn_norm", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_attn_norm", yrow)
                print(
                    f"attn_norm on attn_x x{SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {SYN_MAX_ROW_REL_L2})"
                )
                if yrel > SYN_MAX_REL_L2 or yrow > SYN_MAX_ROW_REL_L2:
                    failures.append(f"attn_norm scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # q_resid (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    if finite_shape("q_resid"):
        rel_row("q_resid", Q_MAX_REL_L2, Q_MAX_ROW_REL_L2, Q_RATIO)
        if n_ok:
            cpu = ref.component(layer, "q_a")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            rel_row("q_resid", Q_MAX_REL_L2, Q_MAX_ROW_REL_L2, what="vs cpu", want=cpu(rctx, xn).float())

            xs = (xn * Q_SYN_SCALE).bfloat16().float()
            syn_want = cpu(rctx, xs).float()
            syn_out = muts["q_a"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"q_a scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, ymin, ymax, yrow = _errors(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_q_resid", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_q_resid", yrow)
                print(
                    f"q_a on attn_norm x{Q_SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {Q_SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {Q_SYN_MAX_ROW_REL_L2})"
                )
                if yrel > Q_SYN_MAX_REL_L2 or yrow > Q_SYN_MAX_ROW_REL_L2:
                    failures.append(f"q_a scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # Downstream of q_a: the indexer's top-k, attn_out.
    if "topk" in gl:
        if seen["topk"].numel() != gl["topk"].numel():
            failures.append(f"topk: shape {tuple(seen['topk'].shape)} vs golden {tuple(gl['topk'].shape)}")
        else:
            ov = _topk_overlap(seen["topk"], gl["topk"])
            metrics.record("topk_overlap_swap", ov)
            print(f"topk overlap vs golden: {ov:.5f} (>= {TOPK_MIN_OVERLAP})")
            if ov < TOPK_MIN_OVERLAP:
                failures.append(f"topk overlap {ov:.5f} < {TOPK_MIN_OVERLAP}")
    if finite_shape("attn_out"):
        rel_row("attn_out", ATT_MAX_REL_L2, ATT_MAX_ROW_REL_L2)

    # h_mid.
    if finite_shape("h_mid"):
        rel_row("h_mid", MID_MAX_REL_L2, MID_MAX_ROW_REL_L2)

    # Block out.
    out_rel = _errors(seen["out"], gl["out"])[0]
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"].float()).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)

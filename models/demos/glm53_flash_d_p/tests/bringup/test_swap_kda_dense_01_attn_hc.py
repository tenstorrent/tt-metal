# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type kda_dense (layer 0) with attn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (kda_dense) with these steps on the device and the rest on the CPU reference:
    attn_hc

Reviewed (S.kda_dense.01.test.1): the gated metric is pcc_swap_out (PCC, float [8192, 4096] = 4-stream residual
[S * 4, H], spec block threshold 0.98). PCC alone is blind to mHC coefficient bugs: the comb mix of the 4 residual
streams dominates out. Measured on the CPU (golden s4096 chunk 1, attn_hc output perturbed, rest CPU), block out
PCC / rel L2 / per-token norm ratio: reference 0.999999 / 0.0017 / [0.9998, 1.0002]; zero stub 0 / 1.0; comb
transposed 0.9985 / 0.075 / [0.90, 1.21]; comb rows instead of columns normalized 0.9985 / 0.070; post x1.05
0.9991 / 0.045; pre x1.05 0.99986 / 0.017 / [0.93, 1.12]; last row zeroed 0.99993 / 0.012 / min 0; random error at
the component test's limits 0.9997 / 0.026. All but the stub pass the 0.98 gate.
Extra asserted checks (informational metrics, not in the runner's list):
  - block out: finite, rel L2 <= 0.01, per-token norm ratio in [0.97, 1.03];
  - the swapped step's own output vs golden (its input is the golden `in`, so this is the component test's check):
    rel L2 and max abs per part, comb columns sum to 1, pre in (0, 1], post in [0, 2].
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
BLOCK_TYPE = "kda_dense"
SWAPPED = ["attn_hc"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
OUT_MAX_REL_L2 = 0.01  # block out: ||got - want|| / ||want||
OUT_ROW_NORM_RATIO = (0.97, 1.03)  # block out: per-row ||got|| / ||want||
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
MAX_REL_L2 = {"pre": 0.03, "post": 0.03, "comb": 0.02}  # attn_hc per part, as the component test
MAX_ABS = {"pre": 0.15, "post": 0.2, "comb": 0.1}
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1|


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


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
            fails.append(f"attn_hc {name} max abs {mx:.3f} > {MAX_ABS[name]}")
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

    failures = [] if ok else [f"pcc_swap_out below {thr}"]
    out, w = seen["out"].float().reshape(gl["out"].shape), gl["out"].float()
    out_rel = _rel(out, w)
    ratio = out.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("row_norm_ratio_min_swap_out", rmin)
    metrics.record("row_norm_ratio_max_swap_out", rmax)
    print(f"block out: rel_l2={out_rel:.6f} (<= {OUT_MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}]")
    if not torch.isfinite(out).all():
        failures.append("block out non-finite")
    if out_rel > OUT_MAX_REL_L2:
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2}")
    if not (OUT_ROW_NORM_RATIO[0] <= rmin and rmax <= OUT_ROW_NORM_RATIO[1]):
        failures.append(f"block out per-row norm ratio [{rmin:.4f}, {rmax:.4f}] outside {OUT_ROW_NORM_RATIO}")
    failures += _hc_checks(seen["attn_hc"], gl["attn_hc"])
    assert not failures, "; ".join(failures)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type full_dense (layer 0) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (full_dense) with these steps on the device and the rest on the CPU reference:
    attn_norm

Reviewed (mimo_v2_6_d_p_2x2 S.full_dense.01.test.1): copied from the 1x4 prior's frozen test
(models/demos/mimo_v2_6_d_p/tests/bringup/, same golden); only this docstring differs. The gated metric is pcc_swap_out (PCC, float [2048, 4096] golden, spec block
threshold 0.98). PCC is scale-invariant and the residual dominates `out`, so the gate alone is weak for a norm swap
(see known_issues "What a swap test can and cannot see"). Extra asserted checks, as in
gemma4_a4b_d_p/tests/bringup/test_swap_sliding_01_attn_norm.py (informational metrics, not in the runner's list):
  - the swapped step's own output vs golden: finite, PCC >= spec component threshold, relative L2 <= 0.03 and
    per-token norm ratio in [0.97, 1.03] (catches sum-instead-of-mean, eps, ``1 + w`` / no weight);
  - block out: finite, relative L2 <= 0.01.
Measured on the CPU by the 1x4 prior (block out PCC / rel L2 | step rel L2 / norm ratio): reference 0.999999 / 0.0017 | 0.0016 /
[0.998, 1.002]; zero stub 0.9884 / 0.155 (PASSES the 0.98 gate: at layer 0 the residual and the attention over the
golden KV prefix dominate out); sum-instead-of-mean 0.9890 / 0.151 (passes the gate) | 0.98 / 0.016; eps 1e-2 0.9916 /
0.135 (passes the gate) | 0.89; x2 0.9775 / 0.30; x1.05 0.99996 / 0.0100 | 0.050; 5% noise 0.999998 / 0.0021 | 0.051 /
[0.93, 1.08]; ``1 + w`` or no weight 0.67 / 54; last row zeroed 0.999998 / 0.0018 | 0.017 / min 0.0 (caught only by
the norm ratio). All of these fail the extra checks; the reference passes them with margin.
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
BLOCK_TYPE = "full_dense"
SWAPPED = ["attn_norm"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||
STEP_ROW_NORM_RATIO = (0.97, 1.03)  # swapped step output: per-token ||got|| / ||want||
OUT_MAX_REL_L2 = 0.01  # block output: ||got - want|| / ||want||


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


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

    # Extra checks (see the module docstring). Recorded as informational metrics, asserted here.
    failures = [] if ok else [f"pcc_swap_out below {thr}"]
    out_rel = _rel(seen["out"], gl["out"])
    metrics.record("rel_l2_swap_out", out_rel)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2})")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    comp_thr = threshold(S, "component")
    for name in SWAPPED:
        o = _step(ref, layer, name).output
        want = gl[o]
        if not want.is_floating_point():
            continue
        if seen[o].numel() != want.numel():
            failures.append(f"swapped step {name}: shape {tuple(seen[o].shape)} vs golden {tuple(want.shape)}")
            continue
        got, w = seen[o].float().reshape(want.shape), want.float()
        v = metrics.pcc(got, w)
        rel = _rel(got, w)
        ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
        rmin, rmax = ratio.min().item(), ratio.max().item()
        metrics.record(f"rel_l2_swap_{o}", rel)
        metrics.record(f"row_norm_ratio_min_swap_{o}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_{o}", rmax)
        print(
            f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f} (<= {STEP_MAX_REL_L2}) "
            f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {STEP_ROW_NORM_RATIO})"
        )
        if not torch.isfinite(got).all():
            failures.append(f"swapped step {name}: non-finite output")
        if v < comp_thr or rel > STEP_MAX_REL_L2:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} (scale or weight bug, e.g. 1 + w)")
        if not (STEP_ROW_NORM_RATIO[0] <= rmin and rmax <= STEP_ROW_NORM_RATIO[1]):
            failures.append(f"swapped step {name}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}]")
    assert not failures, "; ".join(failures)

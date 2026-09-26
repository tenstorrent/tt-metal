# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type sliding_moe (layer 1) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (sliding_moe) with these steps on the device and the rest on the CPU reference:
    attn_norm

Reviewed (S.sliding_moe.01.test.1): same body as test_swap_full_dense_01_attn_norm.py with BLOCK_TYPE sliding_moe.
The gated metric is pcc_swap_out (PCC, float [2048, 4096] golden, spec block threshold 0.98). The residual dominates
`out`, so the gate alone is weak for a norm swap (known_issues "What a swap test can and cannot see", "A zero stub
passes the layer-0 swap gate"). Extra asserted checks (informational metrics, not in the runner's list):
  - the swapped step's own output vs golden: finite, PCC >= spec component threshold, relative L2 <= 0.03 and
    per-token norm ratio in [0.97, 1.03] (the C.sliding_moe.attn_norm limits);
  - block out: finite, relative L2 <= 0.01.
Measured (block out PCC gate / rel L2 | step rel L2 / norm ratio): BRINGUP_IMPL=reference 0.999996 / 0.0030 |
0.0024 / [0.9954, 1.0043]; device (TtRMSNorm) 0.999995 / 0.0032 | 0.0030 / [0.9934, 1.0058]; zero stub fails the gate
(out PCC 0, rel 16.9: unlike layer 0, a zero norm blows up this block). Mutations of the CPU step (temporary env
hook, since removed): x1.05 passes the gate, out rel 0.0100 | 0.050 / [1.045, 1.055]; eps 1e-2 passes the gate, out rel
0.014 | 2.94; 5% noise passes the gate and out rel | 0.050; sum-instead-of-mean, no weight, zeroed last row fail
the gate (and the extras). Every mutation fails at least one extra check.
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
BLOCK_TYPE = "sliding_moe"
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

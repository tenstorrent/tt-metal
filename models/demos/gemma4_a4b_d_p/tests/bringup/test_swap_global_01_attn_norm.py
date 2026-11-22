# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 1: block type global (layer 5) with attn_norm swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 5 (global) with these steps on the device and the rest on the CPU reference:
    attn_norm

Reviewed (S.global.01.test.1): same checks as test_swap_sliding_01_attn_norm.py (see known_issues "Block-out PCC
barely sees a pre-attention norm"). The gated metric is pcc_swap_out (PCC, float [2048, 2816] golden, spec block
threshold 0.98). The global attention also normalizes its input per head (q_norm, k_norm, and V = unscaled RMS norm of
k_proj), so a uniform scale error in attn_norm cannot reach block out, and the residual dominates `out`.
Measured on the CPU, layer 5, chunk 1 of s4096 (block out PCC / rel L2 | attn_norm out PCC / rel L2):
reference 0.999996 / 0.0029 | 1.0 / 0.0024; ``1 + w`` 0.9878 / 0.156 | 0.818 / 1.69 (passes 0.98!); no weight
0.9782 / 0.208 | 0.574 / 1.39; wrong weight (post_attention) 0.9766 / 0.215 | 0.501 / 1.56; 5% noise on attn_norm
0.99996 / 0.0094 | 0.9987 / 0.050; x2 identical to the reference at block out | 1.0 / 1.0; zero 0.962 / 0.274 | 0 / 1.0.
Extra asserted checks (informational metrics, not in the runner's threshold list):
  - each swapped float step's own output vs golden: PCC >= spec component threshold, relative L2 <= 0.03
    (catches 5% noise rel 0.050, x2 rel 1.0, ``1 + w``, no weight and wrong weight);
  - block out relative L2 <= 0.01 (catches ``1 + w`` rel 0.156; reference 0.0029).
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
BLOCK_TYPE = "global"
SWAPPED = ["attn_norm"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
STEP_MAX_REL_L2 = 0.03  # swapped step output: ||got - want|| / ||want||
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
        v = metrics.pcc(seen[o].float().reshape(want.shape), want.float()) if seen[o].numel() == want.numel() else 0.0
        rel = _rel(seen[o], want) if seen[o].numel() == want.numel() else float("inf")
        metrics.record(f"rel_l2_swap_{o}", rel)
        print(f"swapped {name}: pcc={v:.6f} (>= {comp_thr}) rel_l2={rel:.6f} (<= {STEP_MAX_REL_L2})")
        if v < comp_thr or rel > STEP_MAX_REL_L2:
            failures.append(f"swapped step {name}: pcc {v:.4f} / rel L2 {rel:.4f} (scale or weight bug, e.g. 1 + w)")
    assert not failures, "; ".join(failures)

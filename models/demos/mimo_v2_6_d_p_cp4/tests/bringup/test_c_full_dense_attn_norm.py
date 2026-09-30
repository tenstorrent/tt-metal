# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type full_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_dense.attn_norm.test.1), starting from the prior's frozen mimo_v2_6_d_p test (same golden). PCC is
scale-invariant, so a uniform-scale bug (sum instead of mean in the RMS) or a weight-form bug (``1 + w`` instead of
MiMo's plain ``w``) can pass 0.99. Extra checks vs the golden ([2048, 4096], s4096 chunk 1): output finite, relative
L2 <= 0.03, per-token norm ratio in [0.97, 1.03], worst row rel L2 <= 0.01 (a wrong or zeroed CP slice or a few rows).
Measured: fp32 CPU reference rel 0.0016, worst row 0.0029; sum-instead-of-mean rel 0.98; ``1 + w`` rel 54.
The golden's smallest row mean square (3.95e-5) is 40x eps (1e-6), so eps 0 / 2e-6 score rel 0.006 there and pass.
Hence also: the module on the golden input x 0.1 vs the CPU step on the same input: rel <= 0.02, worst row <= 0.04
(bf16 in/out 0.0022 / 0.0044; eps 0 0.40, eps 1e-7 0.33, eps 2e-6 0.17, eps 1e-5 0.54).
"""

import torch

from models.demos.common.bringup.core import metrics
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
STEP = "attn_norm"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_ROW_REL = 0.01  # worst per-token rel L2 vs golden
EPS_SCALE = 0.1  # input scale for the eps check
EPS_MAX_REL = 0.02  # module vs CPU step on the scaled input
EPS_MAX_ROW = 0.04


def _errors(got, want):
    got = got.float().reshape(want.shape)
    w = want.float()
    rel = ((got - w).norm() / w.norm()).item()
    wn = w.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    row = ((got - w).norm(dim=-1) / wn).max().item()
    return torch.isfinite(got).all().item(), rel, ratio.min().item(), ratio.max().item(), row


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
    finite, rel, rmin, rmax, row = _errors(out, want)
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"worst_row_rel_{STEP}_L{LAYER:02d}", row)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"worst_row={row:.6f} (<= {MAX_ROW_REL})"
    )
    assert finite, "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale or weight bug, e.g. 1 + w)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.4f} > {MAX_ROW_REL} (wrong rows or CP slice)"

    # eps check: on a scaled input the row mean square is comparable to eps, so a wrong eps shows.
    xs = [inputs[0] * EPS_SCALE] + inputs[1:]
    cpu = ref.component(LAYER, STEP)
    want_s = cpu(reference_ctx(ref, LAYER, g, c), *xs).float()
    out_s = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *xs)
    finite_s, rel_s, smin, smax, row_s = _errors(out_s, want_s)
    metrics.record(f"eps_rel_l2_{STEP}_L{LAYER:02d}", rel_s)
    metrics.record(f"eps_worst_row_{STEP}_L{LAYER:02d}", row_s)
    print(
        f"input x{EPS_SCALE} vs CPU step: rel_l2={rel_s:.6f} (<= {EPS_MAX_REL}) worst_row={row_s:.6f} "
        f"(<= {EPS_MAX_ROW}) ratio=[{smin:.4f}, {smax:.4f}]"
    )
    assert finite_s, "non-finite output on the scaled input"
    assert rel_s <= EPS_MAX_REL, f"scaled-input rel L2 {rel_s:.4f} > {EPS_MAX_REL} (wrong RMSNorm eps?)"
    assert row_s <= EPS_MAX_ROW, f"scaled-input worst row {row_s:.4f} > {EPS_MAX_ROW} (wrong RMSNorm eps?)"

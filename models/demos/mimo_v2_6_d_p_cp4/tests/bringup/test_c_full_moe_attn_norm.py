# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.attn_norm.test.1). The same checks as the frozen cp4 test_c_sliding_moe_attn_norm.py (the step
is the same RMSNorm: plain ``w``, eps 1e-6). The prior's frozen mimo_v2_6_d_p test for this step uses the same
golden. PCC is scale-invariant, so the test also checks against the golden (in [2048, 4096] -> attn_norm, layer 5
input_layernorm w in [-0.005, 1.14]): output finite, relative L2 <= 0.03, per-token norm ratio in [0.97, 1.03],
worst row rel L2 <= 0.015, rel L2 per CP slice (4 x 512 rows) <= 0.01.
Measured on the CPU (float64 PCC / rel / worst row / worst slice):
- fp32 reference: 0.999997 / 0.0024 / 0.0045 / 0.0024.
- bf16 math: 0.999996 / 0.0028 / 0.0050 / 0.0028.
- Sum instead of mean: PCC 0.999997, but rel 0.98.
- eps 1e-3: PCC 0.998, but rel 0.21.
- ``1 + w``: PCC 0.41. No weight: PCC 0.38.
- Last 32 rows zeroed: PCC 0.993, but worst row 1.0.
- CP slices 1 and 2 swapped: PCC 0.65.
The golden's smallest row mean square (7.5e-4) is 750x eps, so eps 0 / 1e-7 / 2e-6 score rel 0.0024 there and pass
(eps 1e-5 rel 0.0037). Hence also the eps check: the module on the golden input x 0.1 vs the CPU step on the same
input, rel <= 0.02 and worst row <= 0.04. Measured there: bf16 in/out 0.0023 / 0.0038; eps 0 0.031 / 0.064;
eps 1e-7 0.028 / 0.057; eps 2e-6 0.028 / 0.054; eps 1e-5 0.19 / 0.30.
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
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_ROW_REL = 0.015  # worst per-token rel L2 vs golden
MAX_SLICE_REL = 0.01  # rel L2 per CP slice vs golden
CP = 4
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

    # Per CP slice (chip c holds rows c*S/4 ..): a wrong, swapped or stale slice.
    got = out.float().reshape(want.shape)
    w = want.float()
    L = w.shape[-2] // CP
    for r in range(CP):
        gs, ws = got[..., r * L : (r + 1) * L, :], w[..., r * L : (r + 1) * L, :]
        srel = ((gs - ws).norm() / ws.norm()).item()
        metrics.record(f"rel_l2_cp_slice{r}_{STEP}_L{LAYER:02d}", srel)
        print(f"cp slice {r}: rel_l2={srel:.6f} (<= {MAX_SLICE_REL})")
        assert srel <= MAX_SLICE_REL, f"CP slice {r} rel L2 {srel:.4f} > {MAX_SLICE_REL}"

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

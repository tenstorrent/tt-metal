# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.ffn_residual.test.1, ported from the prior mimo_v2_6_d_p frozen test; same golden, [2048, 4096]):
out = h_mid + experts_out (plain add, no scale, no post-MoE norm). ||h_mid|| 126.25, ||experts_out|| 8.47,
||out|| 127.52: the experts term is only ~6.7% of the output, so PCC barely sees it. Prior CPU measurements (PCC /
rel L2): bf16 add 0.999997 / 0.0023; experts_out dropped 0.9978 (passes) / 0.066; h_mid + 0.5 experts_out 0.99946
(passes) / 0.033; 2 (h_mid + experts_out) 0.999998 (passes) / 1.0; experts_out shifted one row 0.9957 (passes) /
0.092; last row zeroed 0.99985 (passes) / 0.0173; residual dropped 0.18.
Asserted checks (limits as the prior test and the cp4 sliding_moe layer 1 test): output size, finite, rel L2 <= 0.01,
per-token norm ratio in [0.99, 1.01], and on delta = out - h_mid: coefficient <delta, experts_out> / ||experts_out||^2
in [0.97, 1.03] and ||delta - experts_out|| / ||experts_out|| <= 0.1. The add must stay bf16 or better.
CP=4 (this bring-up): chip r holds rows [r*S/4, (r+1)*S/4). rel L2 and the experts term are also asserted per CP slice.
Measured on this golden (whole rel / per-slice rel / per-slice coef, experts rel): fp32 add 0.0019 / 0.0019 / 1.0, 0;
bf16 add 0.0023 / 0.0023 / 1.0, 0.017-0.021; golden out 0 / 0 / 1.0, 0.026-0.032; experts_out slices 1 and 2 swapped
0.064 / 0.090 / 0.04-0.05, 1.3-1.5; slice 3's experts_out dropped 0.037 / 0.074 / 0, 1.0; slice 3's experts_out halved
0.018 / 0.037 / 0.5, 0.5; slice 0's experts_out written to slice 1 0.043 / 0.084 / 0.03, 1.39; experts_out shifted
one row 0.092 / 0.085-0.10 / 0.01-0.05, 1.38-1.40.
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
STEP = "ffn_residual"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||, whole chunk and per CP slice
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
EXP_COEF = (0.97, 1.03)  # <out - h_mid, experts_out> / ||experts_out||^2, whole chunk and per CP slice
MAX_EXP_REL = 0.1  # ||(out - h_mid) - experts_out|| / ||experts_out||, whole and per slice (bf16 add: 0.019)
CP = 4


def _exp_term(delta, exp):
    coef = ((delta * exp).sum() / (exp * exp).sum().clamp_min(1e-30)).item()
    erel = ((delta - exp).norm() / exp.norm().clamp_min(1e-30)).item()
    return coef, erel


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

    # Scale, per-row and per-slice checks (PCC is scale-invariant and barely sees a few bad rows).
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()

    # The experts term, checked directly on out - h_mid (it is small next to h_mid).
    res, exp = (x.float().reshape(want.shape) for x in inputs)
    delta = got - res
    coef, erel = _exp_term(delta, exp)

    rows = w.shape[0]
    assert rows % CP == 0, f"chunk rows {rows} not divisible by CP={CP}"
    L = rows // CP
    sl = [slice(r * L, (r + 1) * L) for r in range(CP)]
    slice_rel = [((got[s] - w[s]).norm() / w[s].norm()).item() for s in sl]
    slice_exp = [_exp_term(delta[s], exp[s]) for s in sl]

    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"rel_l2_slice_max_{STEP}_L{LAYER:02d}", max(slice_rel))
    metrics.record(f"experts_coef_{STEP}_L{LAYER:02d}", coef)
    metrics.record(f"experts_rel_l2_{STEP}_L{LAYER:02d}", erel)
    metrics.record(f"experts_rel_l2_slice_max_{STEP}_L{LAYER:02d}", max(e for _, e in slice_exp))
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"slice_rel={[round(x, 5) for x in slice_rel]}"
    )
    print(
        f"experts term: coef={coef:.4f} (in {EXP_COEF}) rel={erel:.4f} (<= {MAX_EXP_REL}) "
        f"per slice={[(round(a, 4), round(b, 4)) for a, b in slice_exp]}"
    )

    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"
    worst = max(range(CP), key=lambda r: slice_rel[r])
    assert (
        slice_rel[worst] <= MAX_REL_L2
    ), f"CP slice {worst} relative L2 {slice_rel[worst]:.4f} > {MAX_REL_L2} (slice order or gather bug?)"
    assert (
        EXP_COEF[0] <= coef <= EXP_COEF[1]
    ), f"experts_out coefficient {coef:.4f} outside {EXP_COEF} (dropped or scaled experts_out?)"
    assert erel <= MAX_EXP_REL, f"experts term rel L2 {erel:.4f} > {MAX_EXP_REL} (experts_out misaligned or corrupted)"
    for r, (sc, se) in enumerate(slice_exp):
        assert EXP_COEF[0] <= sc <= EXP_COEF[1], f"CP slice {r}: experts_out coefficient {sc:.4f} outside {EXP_COEF}"
        assert se <= MAX_EXP_REL, f"CP slice {r}: experts term rel L2 {se:.4f} > {MAX_EXP_REL} (slice misplaced?)"

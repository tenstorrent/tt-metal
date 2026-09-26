"""Component test: ffn_residual of block type sliding_moe (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding_moe.ffn_residual.test.1): out = h_mid + experts_out (plain add, no scale, no post-MoE norm).
Golden [2048, 4096], out stored bf16: ||h_mid|| 79.58, ||experts_out|| 12.44, ||out|| 82.24. Measured as PCC / rel L2 /
per-token norm ratio / experts coef / experts rel: fp32 reference 0.999997 / 0.0024 / [0.9991, 1.0010] / 1.0 / 0;
bf16 add 0.999996 / 0.0029 / [0.9989, 1.0012] / 1.0 / 0.011; experts_out dropped 0.9887 (passes) / 0.151 /
[0.786, 1.007]; h_mid + 0.5 experts_out 0.9973 (passes) / 0.076; 2 (h_mid + experts_out) 0.999997 (passes) / 1.0;
experts_out shifted one row 0.980 / 0.201; last row zeroed 0.99981 (passes) / 0.0196 / min 0; residual dropped 0.286.
Extra asserted checks: output size, finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01] (as the other
residuals), and on the experts term delta = out - h_mid: coefficient <delta, experts_out> / ||experts_out||^2 in
[0.97, 1.03] and ||delta - experts_out|| / ||experts_out|| <= 0.1 (bf16 add 0.011, golden ~0.016; last row zeroed
0.128). The add must stay bf16 or better.
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
EXP_COEF = (0.97, 1.03)  # <out - h_mid, experts_out> / ||experts_out||^2
MAX_EXP_REL = 0.1  # ||(out - h_mid) - experts_out|| / ||experts_out|| (bf16 add alone gives 0.011)


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

    # Scale and per-row checks (PCC is scale-invariant and barely sees a few bad rows). Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO})")
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"

    # The experts term, checked directly on out - h_mid.
    res, exp = (x.float().reshape(want.shape) for x in inputs)
    delta = got - res
    coef = ((delta * exp).sum() / (exp * exp).sum().clamp_min(1e-30)).item()
    erel = ((delta - exp).norm() / exp.norm().clamp_min(1e-30)).item()
    metrics.record(f"experts_coef_{STEP}_L{LAYER:02d}", coef)
    metrics.record(f"experts_rel_l2_{STEP}_L{LAYER:02d}", erel)
    print(f"experts term: coef={coef:.4f} (in {EXP_COEF}) rel={erel:.4f} (<= {MAX_EXP_REL})")
    assert (
        EXP_COEF[0] <= coef <= EXP_COEF[1]
    ), f"experts_out coefficient {coef:.4f} outside {EXP_COEF} (dropped or scaled experts_out?)"
    assert erel <= MAX_EXP_REL, f"experts term rel L2 {erel:.4f} > {MAX_EXP_REL} (experts_out misaligned or corrupted)"

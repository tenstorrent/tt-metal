# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.attn_residual.test.1): h_mid = in + attn_out (plain add, no scale). Layer 5 is full attention with
no sink, so attn_out is large here, unlike sliding layer 1 (||in|| 121.4, ||attn_out|| 78.9, ||h_mid|| 126.3,
[2048, 4096], all bf16). Measured as PCC / rel L2 / per-token norm ratio / attn coef / attn rel: fp32 reference
0.999997 / 0.0025 / [0.9989, 1.0009] / 1.0000 / 0.0040 vs the golden; bf16 add 0.999996 / 0.0030 / [0.9989, 1.0012] /
1.0000 / 0.0028 (vs its own inputs); 2 * (in + attn_out) 0.999997 (passes) / 1.0 / 2.0; last row zeroed 0.99985
(passes) / 0.0176 / min 0; row 0 zeroed 0.99980 (passes) / 0.020; last 32 columns zeroed 0.99833 (passes) / 0.058;
last 32 rows zeroed 0.99247 (passes) / 0.12; attn_out shifted one row 0.9863 / 0.165 / coef 0.965 / attn rel 0.265;
attn_out dropped 0.798; in + 0.5 attn_out 0.950. Extra asserted checks, as sliding_moe: output size, finite, rel L2 <=
0.01, per-token norm ratio in [0.99, 1.01], and on delta = out - in: coef <delta, attn_out> / ||attn_out||^2 in
[0.98, 1.02] and ||delta - attn_out|| / ||attn_out|| <= 0.05 (tighter than layer 1's [0.95, 1.05] / 0.3 because the
bf16 output rounding is small next to this attn_out: 0.003). The add must stay bf16 or better.
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
STEP = "attn_residual"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
ATTN_COEF = (0.98, 1.02)  # <out - in, attn_out> / ||attn_out||^2
MAX_ATTN_REL = 0.05  # ||(out - in) - attn_out|| / ||attn_out|| (bf16 output rounding alone gives 0.003)


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

    # Check the attention term directly on out - in (catches a scaled or misaligned attn_out).
    res, attn = (x.float().reshape(want.shape) for x in inputs)
    delta = got - res
    coef = ((delta * attn).sum() / (attn * attn).sum().clamp_min(1e-30)).item()
    arel = ((delta - attn).norm() / attn.norm().clamp_min(1e-30)).item()
    metrics.record(f"attn_coef_{STEP}_L{LAYER:02d}", coef)
    metrics.record(f"attn_rel_l2_{STEP}_L{LAYER:02d}", arel)
    print(f"attn term: coef={coef:.4f} (in {ATTN_COEF}) rel={arel:.4f} (<= {MAX_ATTN_REL})")
    assert (
        ATTN_COEF[0] <= coef <= ATTN_COEF[1]
    ), f"attn_out coefficient {coef:.4f} outside {ATTN_COEF} (dropped or scaled attn_out?)"
    assert arel <= MAX_ATTN_REL, f"attn term rel L2 {arel:.4f} > {MAX_ATTN_REL} (attn_out misaligned or corrupted)"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: mlp_residual of block type full_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_dense.mlp_residual.test.1, ported from the prior mimo_v2_6_d_p; same golden, s4096 chunk 1,
[2048, 4096]): out = h_mid + mlp_out (plain add, no scale). The gated metric is PCC, which misses scale and a few bad
rows. Prior CPU measurements (PCC / rel L2 / per-token norm ratio): fp32 reference 0.999998 / 0.0021 /
[0.9991, 1.0009]; bf16 add 0.999996 / 0.0028 / [0.9987, 1.0015]; 2x 0.999998 (passes) / 1.0 / 2.0; row 0 zeroed
0.99976 (passes) / 0.022 / min 0; last row zeroed 0.99981 (passes) / 0.020; last 32 rows zeroed 0.9922 (passes) /
0.125; last 32 columns zeroed 0.9876 / 0.157; mlp_out shifted one row 0.979; mlp_out dropped 0.914; residual dropped
0.867; h_mid + 0.5 mlp_out 0.987; h_mid - mlp_out 0.249. Extra asserted checks: output size, finite, rel L2 <= 0.01
and per-token norm ratio within [0.99, 1.01]. The add must stay bf16 or better; bfp8 on the residual stream is not
budgeted.
CP=4 (this bring-up): chip r holds rows [r*S/4, (r+1)*S/4) of the chunk, so a slice-ordering or slice-dropping bug in
the device-to-host gather hits whole slices; the test also asserts rel L2 <= 0.01 per CP slice (a swap of two slices
or one slice written twice is ~1.4 rel on that slice).
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
STEP = "mlp_residual"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||, whole chunk and per CP slice
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
CP = 4


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

    # Scale, per-row and per-slice checks (PCC is scale-invariant and barely sees a few bad rows). Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    rows = w.shape[0]
    assert rows % CP == 0, f"chunk rows {rows} not divisible by CP={CP}"
    L = rows // CP
    slice_rel = [
        ((got[r * L : (r + 1) * L] - w[r * L : (r + 1) * L]).norm() / w[r * L : (r + 1) * L].norm()).item()
        for r in range(CP)
    ]
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"rel_l2_slice_max_{STEP}_L{LAYER:02d}", max(slice_rel))
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"slice_rel={[round(x, 5) for x in slice_rel]}"
    )
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"
    worst = max(range(CP), key=lambda r: slice_rel[r])
    assert (
        slice_rel[worst] <= MAX_REL_L2
    ), f"CP slice {worst} relative L2 {slice_rel[worst]:.4f} > {MAX_REL_L2} (slice order or gather bug?)"

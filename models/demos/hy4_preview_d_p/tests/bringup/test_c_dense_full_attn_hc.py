# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.attn_hc.test.1). The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from the
streams [S, 4H]; the golden is stored in bf16 (s4096 chunk 1, 2048 rows). The gated metric is PCC over the whole
[S, 8] matrix, and that is weak here: the 8 columns have very different means (pre ~0.994..0.99996, post
0.11..0.34), so the column means dominate the correlation. Measured on this golden (CPU, fp32 math on the golden
input; mutations of the reference):

    variant                           PCC        rel L2    max abs error per column
    fp32 reference                    1.000000   0.00052   <= 0.0019 (the golden's bf16 rounding)
    bf16 or tf32 streams and fn       1.000000   0.00052   <= 0.0020
    sigmoid with 1e-3 abs error       0.999993   0.0022    <= 0.0039
    post = 1 * sigmoid (not 2x)       0.990992   0.113     0.39
    TP: one chip's partial mixes      0.993925   0.117     0.72
    TP: one chip's partial sum sq     0.998913   0.052     0.13
    RMS over one stream, not four     0.994563   0.112     0.25
    pre gate 3 = pre gate 2           0.998665   0.030     1.0
    fn rows 4 / 5 swapped (post)      0.999544   0.0176    0.10
    base 4 / 5 swapped (post)         0.999861   0.0098    0.030
    fn rows 0 / 1 swapped (pre)       0.999998   0.0011    0.035
    fn chip-major column order        0.994121   0.114     0.55
    rows shifted by 1 (SP offset)     0.956196   0.173     1.0

Every row above from "post = 1 * sigmoid" down passes the 0.99 PCC gate except the last. So the test also checks,
against the golden: output finite, shape [S, 8], relative L2 <= 0.01 (reference 0.00052, smallest bug caught by it
0.0176) and max abs error per column <= 0.015 (reference 0.0019, smallest bug 0.030). This leaves the device room for
about 0.005 of sigmoid / rsqrt error on post (2 * sigmoid doubles it). The rel L2 alone misses the swapped pre and
base rows; the per-column max abs error catches them.

Blind spot: at layer 0 the four input streams are identical (each one is the embedding), and the four stream blocks of
hc_attn_layer.fn are nearly equal (rel diff < 1% per row). A stream-order bug (fn's stream blocks permuted) moves the
gates by <= 1e-4 even on synthetic distinct streams, so no test at layer 0 can see it; the swap tests and later
layers (distinct streams) must.
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
STEP = "attn_hc"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_ABS_ERR = 0.015  # per column, max |got - want| (the golden is bf16: rounding up to 0.002)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; attn_hc is not on the device"
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # PCC over [S, 8] is dominated by the per-column means; check the error itself. Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    col_err = (got - w).abs().amax(dim=0)
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"max_abs_err_{STEP}_L{LAYER:02d}", col_err.max().item())
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) max_abs_err per column (pre 0-3 | post 4-7)="
        f"{[round(v, 5) for v in col_err.tolist()]} (<= {MAX_ABS_ERR})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2}"
    bad = [j for j, v in enumerate(col_err.tolist()) if v > MAX_ABS_ERR]
    assert not bad, f"max abs error above {MAX_ABS_ERR} in gate columns {bad} (pre 0-3, post 4-7): {col_err.tolist()}"

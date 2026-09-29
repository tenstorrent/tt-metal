# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.ffn_hc.test.1). The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from h_mid
[S, 4H] (hc_mlp_layer weights); the golden is stored in bf16 (s4096 chunk 1, 2048 rows). Unlike attn_hc, the four
input streams are distinct here (h_mid_j = embedding + post_j x attn_out; streams 1 / 2 / 3 differ from stream 0 by
18 % / 6 % / 111 %), and the pre gates are small and unsaturated: column means 0.094, 2e-6..1.8e-5, 0.0009, 0.018
(post 0.49..0.72). Because pre_0 is only ~0.09, an absolute error of 1e-3 on any pre gate moves the FFN input
ffn_x = sum_j pre_j x stream_j by ~3 %: the pre gates need absolute accuracy of a few 1e-4, which neither PCC nor
a whole-matrix rel L2 can see (both are dominated by the post columns). Measured on this golden (CPU, fp32 math on
the golden input; mutations of the reference):

    variant                        PCC       rel L2   post col rel  ffn_x rel / worst row  post worst row
    fp32 reference                 0.999997  0.00167  <= 0.0017     0.0012 / 0.0035        0.0031
    bf16 input and fn              0.999997  0.00168  <= 0.0017     0.0012 / 0.0035
    sigmoid abs err 3e-4           -         0.00202  <= 0.0021     0.0085 / 0.021
    sigmoid abs err 1e-3           0.999983  0.0041   <= 0.0045     0.029 / 0.067          0.0069
    post = 1 * sigmoid (not 2x)    0.993356  0.498    0.5           -
    pre gate 3 = pre gate 2        0.999835  0.0147   -             0.42 / 0.56
    pre gate 2 = pre gate 1        -         0.0019   -             0.018 / 0.072
    fn rows 1 / 2 swapped (pre)    -         0.0019   -             0.013 / 0.051
    fn rows 0 / 1 swapped (pre)    0.992777  0.090    -             0.47 / 0.68
    fn rows 4 / 5 swapped (post)   0.991554  0.092    0.156         -
    base 4 / 6 swapped (post)      -         0.0152   0.027         -                      0.021
    base 4 / 5 swapped (post)      0.998973  0.032    0.053         -
    fn streams 0 / 2 swapped       -         0.0129   0.010         0.019 / 0.074
    fn streams 0 / 1 swapped       0.999797  0.0184   0.033         0.056 / 0.084
    pre x 1.01                     -         0.0019   -             0.0101 / 0.0135
    post x 1.01                    -         0.0101   0.0101        -
    rows 1023 / 1024 swapped (SP)  -         0.0056   0.0078        0.0041 / 0.21          0.19
    last row = previous row        -         0.0076   0.0097        0.0086 / 0.29          0.25
    TP: one chip's partial sumsq   0.986894  0.177    -             0.53 / 0.60
    chip-major column order        0.975555  0.287    -             1.6 / 2.0

Most of these pass the 0.99 PCC gate. So the test also checks, against the golden: output finite, element count,
rel L2 over [S, 8] <= 0.01; per post column (4-7) rel L2 <= 0.006 and worst row rel L2 over the post columns
<= 0.015; and, for the pre gates, the downstream FFN input the CPU ffn_hc_pre builds from h_mid with the device gates
vs with the golden gates: rel L2 <= 0.005, worst row <= 0.02. The pre side is judged through ffn_x, not per column,
because pre column 1 is ~1e-5 (hc_eps 1e-6 included): its relative error is meaningless (dropping hc_eps moves it
22 %) but its absolute error is what reaches ffn_x.
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
STEP = "ffn_hc"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC_MULT = 4
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_POST_COL_REL = 0.006  # per post column (4-7), rel L2 vs the golden
MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
MAX_FFN_X_REL = 0.005  # ffn_x = hc_pre(h_mid, gates): device gates vs golden gates, rel L2 over [S, H]
MAX_FFN_X_ROW_REL = 0.02  # same, worst row


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; ffn_hc is not on the device"
    rctx = reference_ctx(ref, LAYER, g, c)
    out = fn(rctx, device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # PCC over [S, 8] is dominated by the post columns; check the error itself. Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    err = got - w
    rel = (err.norm() / w.norm()).item()
    col_rel = (err.norm(dim=0) / w.norm(dim=0)).tolist()
    post_col = max(col_rel[HC_MULT:])
    post_row = (err[:, HC_MULT:].norm(dim=-1) / w[:, HC_MULT:].norm(dim=-1)).max().item()
    col_err = err.abs().amax(dim=0).tolist()

    # Pre gates through the step that consumes them: ffn_x from the golden h_mid.
    hc_pre = ref.component(LAYER, "ffn_hc_pre")
    h_mid = inputs[0]
    fx_want = hc_pre(rctx, h_mid, w).float()
    fx_got = hc_pre(rctx, h_mid, got).float()
    fx_rel = ((fx_got - fx_want).norm() / fx_want.norm()).item()
    fx_row = ((fx_got - fx_want).norm(dim=-1) / fx_want.norm(dim=-1)).max().item()

    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"post_col_rel_l2_{STEP}_L{LAYER:02d}", post_col)
    metrics.record(f"post_worst_row_rel_l2_{STEP}_L{LAYER:02d}", post_row)
    metrics.record(f"ffn_x_rel_l2_{STEP}_L{LAYER:02d}", fx_rel)
    metrics.record(f"ffn_x_worst_row_rel_l2_{STEP}_L{LAYER:02d}", fx_row)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) col rel (pre 0-3 | post 4-7)={[round(v, 5) for v in col_rel]} "
        f"max abs per column={[f'{v:.2e}' for v in col_err]}"
    )
    print(
        f"post: max col rel={post_col:.6f} (<= {MAX_POST_COL_REL}) worst row rel={post_row:.5f} (<= {MAX_POST_ROW_REL}); "
        f"pre via ffn_x: rel={fx_rel:.6f} (<= {MAX_FFN_X_REL}) worst row={fx_row:.5f} (<= {MAX_FFN_X_ROW_REL})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2}"
    bad = [HC_MULT + j for j, v in enumerate(col_rel[HC_MULT:]) if v > MAX_POST_COL_REL]
    assert not bad, f"post gate columns {bad} rel L2 above {MAX_POST_COL_REL}: {col_rel}"
    assert post_row <= MAX_POST_ROW_REL, f"post gates worst row rel L2 {post_row:.5f} > {MAX_POST_ROW_REL}"
    assert fx_rel <= MAX_FFN_X_REL, f"pre gates: ffn_x rel L2 {fx_rel:.5f} > {MAX_FFN_X_REL} (pre gate abs error?)"
    assert fx_row <= MAX_FFN_X_ROW_REL, f"pre gates: ffn_x worst row rel L2 {fx_row:.5f} > {MAX_FFN_X_ROW_REL}"

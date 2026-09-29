# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.attn_hc.test.1). Built from the layer-1 test (test_c_moe_full_attn_hc.py); attn_hc is the same
op on every block type. The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from the streams [S, 4H]
(hc_attn_layer weights of layer 2); the golden is stored in bf16 (s4096 chunk 1, 2048 rows). The streams are distinct
(streams 1 / 2 / 3 differ from stream 0 by 29 % / 34 % / 475 %; stream norms 71 / 66 / 74 / 340) and so are fn's
stream blocks (79 % / 110 % / 185 %). Row RMS 0.007 .. 0.14, so rms_norm_eps 1e-5 matters. The gate columns span
six decades: column RMS pre 2.1e-6, 0.96, 0.26, 0.046 and post 0.27, 0.32, 0.29, 0.0061. Pre gate 0 sits at hc_eps
(1.03e-6 .. 1.6e-5, mean 1.7e-6), so its relative error measures the eps and the last bits of a ~7e-7 sigmoid, and
it has no effect downstream (pre_0 x stream_0 is 2e-6 of attn_x). Post gate 7 is small too (mean 0.0028) and weights
the large stream 3, but its addend is only 0.2 % of h_mid's stream 3, so post-7 bugs are visible only on the gates.
Measured on this golden (CPU, fp32 math on the golden input; mutations of the reference). "attn_x" and "h_mid" are
the CPU attn_hc_pre / attn_residual built from the golden streams (and golden attn_out) with these gates vs with the
golden gates:

    variant                        PCC       rel L2   col 0 rel  col 1-7 rel  post row  attn_x rel/row  h_mid stream/row
    fp32 reference                 0.999999  0.00132  0.0017     0.0018       0.0030    0.0012 / 0.0027  0.0018 / 0.0045
    device TtHcGates (gate run)    0.999999  0.00136  0.0045     0.0034       0.0048    0.0012 / 0.0032  0.0018 / 0.0058
    bf16 output                    1.000000  0.00037  0.0010     0.0011       0.0047    0.0002 / 0.0040  0.0006 / 0.0075
    sigmoid abs err 1e-4           0.999999  0.00138  47         0.033        0.0066    0.0014 / 0.0028  0.0018 / 0.0061
    sigmoid abs err 3e-4           0.999999  0.00178  141        0.099        0.0165    0.0023 / 0.0054  0.0025 / 0.0118
    post = 1 * sigmoid (not 2x)    0.968341  0.228    0.0017     0.50         0.50      -                0.54 / 0.69
    post x 1.01                    0.999987  0.00475  -          0.0103       0.0129    -                0.0108 / 0.0166
    pre x 1.01                     0.999987  0.00895  0.0103     0.0102       -         0.0101 / 0.0123  -
    post 5 = post 4                0.997770  0.0554   -          0.19         0.32      -                0.21 / 0.43
    base 4 / 5 swapped (post)      0.999684  0.0203   -          0.055        0.063     -                0.057 / 0.087
    fn rows 4 / 5 swapped (post)   0.992920  0.0961   -          0.29         0.49      -                0.25 / 0.47
    fn streams 0 / 1 swapped       0.996082  0.0724   0.0115     0.31         0.12      0.036 / 0.32     0.032 / 0.12
    fn streams 1 / 2 swapped       0.989078  0.120    0.053      0.49         0.27      0.052 / 0.46     0.095 / 0.22
    TP: one chip's partial sumsq   0.990660  0.128    0.63       2.5          0.99      0.084 / 0.24     0.23 / 0.61
    rms eps 1e-6 (not 1e-5)        0.999927  0.0101   0.016      0.26         0.23      0.0036 / 0.055   0.0086 / 0.15
    rows 1023 / 1024 swapped (SP)  0.999992  0.00319  -          0.0077       0.47      0.0014 / 0.056   0.0060 / 0.45
    last row = previous row        0.999926  0.00980  0.0094     0.024        0.62      0.0042 / 0.13    0.022 / 0.71
    no hc_eps                      0.999999  0.00132  0.47       0.0018       0.0030    0.0012 / 0.0027  0.0018 / 0.0045

Every mutation above except "post = 1 x sigmoid" passes rel L2 0.01, and all but three pass the 0.99 PCC gate. So the
test also checks, against the golden: output finite, element count, rel L2 over [S, 8] <= 0.01, rel L2 per column (all
8) <= 0.01, worst row rel L2 over the post columns <= 0.015; the pre gates through attn_x (rel <= 0.005, worst row <=
0.02) and the post gates through h_mid (per stream rel <= 0.003, worst row <= 0.02). Only the per-column check catches
a dropped hc_eps (column 0 rel 0.47; downstream change < 1e-5). Column 0 asks the device for ~1e-8 absolute accuracy
on a gate of ~2e-6 (the fp32 sigmoid of TtHcGates reached 1.05e-7 max abs, rel 0.0045: the tightest margin, 2.2x).
Every mutation in the table fails at least one check.
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC_MULT = 4
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs the golden; pre 0 sits at hc_eps (~2e-6)
MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
MAX_ATTN_X_REL = 0.005  # attn_x = hc_pre(streams, gates): device gates vs golden gates, rel L2 over [S, H]
MAX_ATTN_X_ROW_REL = 0.02  # same, worst row
MAX_H_MID_STREAM_REL = 0.003  # h_mid = hc_post(streams, gates, golden attn_out): per stream rel L2
MAX_H_MID_ROW_REL = 0.02  # same, worst (row, stream)


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
    rctx = reference_ctx(ref, LAYER, g, c)
    out = fn(rctx, device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # PCC over [S, 8] is dominated by the large columns; check the error itself. Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    err = got - w
    rel = (err.norm() / w.norm()).item()
    col_rel = (err.norm(dim=0) / w.norm(dim=0)).tolist()
    col_err = err.abs().amax(dim=0).tolist()
    post_row = (err[:, HC_MULT:].norm(dim=-1) / w[:, HC_MULT:].norm(dim=-1)).max().item()

    # Pre gates through attn_hc_pre, post gates through attn_residual, both on the golden streams (and attn_out).
    streams = inputs[0]
    hc_pre = ref.component(LAYER, "attn_hc_pre")
    ax_want = hc_pre(rctx, streams, w).float()
    ax_got = hc_pre(rctx, streams, got).float()
    ax_rel = ((ax_got - ax_want).norm() / ax_want.norm()).item()
    ax_row = ((ax_got - ax_want).norm(dim=-1) / ax_want.norm(dim=-1)).max().item()
    hc_post = ref.component(LAYER, "attn_residual")
    y = gl["attn_out"].float()
    hm_want = hc_post(rctx, streams, w, y).float().view(w.shape[0], HC_MULT, -1)
    hm_got = hc_post(rctx, streams, got, y).float().view(w.shape[0], HC_MULT, -1)
    dh = hm_got - hm_want
    hm_stream = (dh.norm(dim=(0, 2)) / hm_want.norm(dim=(0, 2))).tolist()
    hm_row = (dh.norm(dim=-1) / hm_want.norm(dim=-1)).max().item()

    tag = f"{STEP}_L{LAYER:02d}"
    metrics.record(f"rel_l2_{tag}", rel)
    metrics.record(f"max_col_rel_l2_{tag}", max(col_rel))
    metrics.record(f"post_worst_row_rel_l2_{tag}", post_row)
    metrics.record(f"attn_x_rel_l2_{tag}", ax_rel)
    metrics.record(f"attn_x_worst_row_rel_l2_{tag}", ax_row)
    metrics.record(f"h_mid_max_stream_rel_l2_{tag}", max(hm_stream))
    metrics.record(f"h_mid_worst_row_rel_l2_{tag}", hm_row)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) col rel (pre 0-3 | post 4-7)={[round(v, 5) for v in col_rel]} "
        f"(<= {MAX_COL_REL}) max abs per column={[f'{v:.2e}' for v in col_err]}"
    )
    print(
        f"post worst row rel={post_row:.5f} (<= {MAX_POST_ROW_REL}); pre via attn_x: rel={ax_rel:.6f} "
        f"(<= {MAX_ATTN_X_REL}) worst row={ax_row:.5f} (<= {MAX_ATTN_X_ROW_REL}); post via h_mid: per stream="
        f"{[round(v, 6) for v in hm_stream]} (<= {MAX_H_MID_STREAM_REL}) worst row={hm_row:.5f} (<= {MAX_H_MID_ROW_REL})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2}"
    bad = [j for j, v in enumerate(col_rel) if v > MAX_COL_REL]
    assert not bad, f"gate columns {bad} (pre 0-3, post 4-7) rel L2 above {MAX_COL_REL}: {col_rel}"
    assert post_row <= MAX_POST_ROW_REL, f"post gates worst row rel L2 {post_row:.5f} > {MAX_POST_ROW_REL}"
    assert ax_rel <= MAX_ATTN_X_REL, f"pre gates: attn_x rel L2 {ax_rel:.5f} > {MAX_ATTN_X_REL}"
    assert ax_row <= MAX_ATTN_X_ROW_REL, f"pre gates: attn_x worst row rel L2 {ax_row:.5f} > {MAX_ATTN_X_ROW_REL}"
    bad = [j for j, v in enumerate(hm_stream) if v > MAX_H_MID_STREAM_REL]
    assert not bad, f"post gates: h_mid streams {bad} rel L2 above {MAX_H_MID_STREAM_REL}: {hm_stream}"
    assert hm_row <= MAX_H_MID_ROW_REL, f"post gates: h_mid worst row rel L2 {hm_row:.5f} > {MAX_H_MID_ROW_REL}"

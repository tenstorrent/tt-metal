# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.attn_hc.test.1). The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from the
streams [S, 4H] (hc_attn_layer weights of layer 1); the golden is stored in bf16 (s4096 chunk 1, 2048 rows). Unlike
layer 0, the four input streams are distinct (streams 1 / 2 / 3 differ from stream 0 by 29 % / 10 % / 163 %, and
fn's four stream blocks by 48 % / 42 % / 188 %), so stream-order bugs are visible here. The row RMS of the streams is
small (0.005 .. 0.073), so rms_norm_eps 1e-5 matters. Several gates are small: column means are pre 0.026, 0.013,
0.83, 0.49 and post 0.0003, 0.0005, 0.042, 0.23. The whole-matrix PCC and rel L2 are dominated by the large
columns. Measured on this golden (CPU, fp32 math on the golden input; mutations of the reference). "attn_x" and
"h_mid" are the CPU attn_hc_pre / attn_residual built from the golden streams (and golden attn_out) with these gates
vs with the golden gates:

    variant                        PCC       rel L2   max col rel  attn_x rel / row   h_mid max stream / row
    fp32 reference                 0.999999  0.00143  0.0019       0.0014 / 0.0043    0.0011 / 0.0029
    bf16 input and fn              0.999999  0.00143  0.0019       0.0014 / 0.0044    0.0011 / 0.0029
    sigmoid abs err 1e-4           0.999998  0.00150  0.26         0.0014 / 0.0044    0.0013 / 0.012
    sigmoid abs err 3e-4           0.999998  0.00194  0.78         0.0016 / 0.0047    0.0038 / 0.037
    post = 1 * sigmoid (not 2x)    0.989338  0.129    0.50         -                  0.34 / 0.42
    post x 1.01                    0.999994  0.00296  0.0103       -                  0.0068 / 0.010
    pre x 1.01                     0.999994  0.00977  0.0102       0.0101 / 0.013     -
    post gate 4 zeroed             0.999998  0.00161  1.0          -                  0.0042 / 0.017
    post gate 5 = post gate 4      0.999997  0.00186  1.05         -                  0.0075 / 0.11
    pre gate 1 = pre gate 0        0.999714  0.0204   1.26         0.014 / 0.081      -
    base 0 / 1 swapped (pre)       0.999966  0.00687  0.24         0.0031 / 0.014     -
    fn rows 0 / 1 swapped (pre)    0.999528  0.0254   0.99         0.0053 / 0.030     -
    fn rows 4 / 5 swapped (post)   0.999755  0.0185   24           -                  0.11 / 1.7
    base 4 / 5 swapped (post)      0.999781  0.0175   15           -                  0.11 / 1.6
    base 6 / 7 swapped (post)      0.998973  0.0386   0.19         -                  0.099 / 0.13
    fn streams 0 / 1 swapped       0.999923  0.0138   0.098        0.012 / 0.023      0.015 / 0.066
    fn streams 1 / 2 swapped       0.994605  0.0877   0.26         0.056 / 0.28       0.066 / 0.38
    TP: one chip's partial sumsq   0.991108  0.151    0.91         0.16 / 0.22        0.24 / 0.41
    RMS over one stream            0.989329  0.167    0.57         0.091 / 0.51       0.25 / 0.64
    rms eps 1e-6 (not 1e-5)        0.999501  0.0312   0.081        0.0077 / 0.096     0.034 / 0.27
    rows 1023 / 1024 swapped (SP)  0.999989  0.00382  0.022        0.0037 / 0.31      0.0025 / 0.095
    last row = previous row        0.999963  0.00715  0.019        0.011 / 0.32       0.0097 / 0.29
    no hc_eps                      0.999999  0.00143  0.0021       0.0014 / 0.0043    0.0011 / 0.0029

Every row above passes the 0.99 PCC gate, and most pass rel L2 0.01. So the test also checks, against the golden:
output finite, element count, rel L2 over [S, 8] <= 0.01, rel L2 per column (all 8) <= 0.01, worst row rel L2 over
the post columns <= 0.015; the pre gates through attn_x (rel <= 0.005, worst row <= 0.02) and the post gates through
h_mid (per stream rel <= 0.003, worst row <= 0.02). Unlike layer 0's ffn_hc, per-column rel L2 is meaningful on every
column here: the smallest column (post 4, ~3e-4) is 300x hc_eps (dropping hc_eps moves it 0.2 %). The per-column
limit asks the device for about 3e-6 absolute accuracy on post gates 4 / 5, i.e. about 1 % relative error on a
sigmoid of ~1.5e-4; an accurate fp32 sigmoid does that (it was 1.6e-7 abs on 1e-5 gates at layer 0's ffn_hc).
Caught by nothing: dropping hc_eps (downstream change < 1e-5).
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC_MULT = 4
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs the golden
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

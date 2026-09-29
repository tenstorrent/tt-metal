# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.ffn_hc.test.1). Built from the layer-1 test (test_c_moe_full_ffn_hc.py); ffn_hc is the same op
on every block type. The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from h_mid [S, 4H]
(hc_mlp_layer weights of layer 2); the golden is stored in bf16 (s4096 chunk 1, 2048 rows). The streams are distinct
(streams 1 / 2 / 3 differ from stream 0 by 44 % / 48 % / 668 %, fn's stream blocks by 71 % / 115 % / 165 %), and the
row RMS is small (0.007 .. 0.14), so rms_norm_eps 1e-5 matters. Column means are pre 1.1e-3, 1.4e-6, 1.0, 0.10 and
post 5.4e-4, 9.6e-4, 3.5e-4, 0.049: pre gate 1 sits at hc_eps (1e-6 .. 5.3e-5), pre gate 2 is saturated at 1.0, and
post gates 4 / 5 / 6 are small but 300x+ hc_eps. The two gate scales are equal (0.0398), so swapping them is a no-op.
Post gate j scaled by mlp_out is 6 % / 10 % / 4 % / 59 % of stream j's norm. Measured on this golden (CPU, fp32 math
on the golden input; mutations of the reference). "ffn_x" and "out" are the CPU ffn_hc_pre / ffn_residual on the
golden h_mid (and golden mlp_out) with these gates vs with the golden gates:

    variant                     PCC       rel L2   col rel (worst)     post row  ffn_x rel / row   out max stream / row
    fp32 reference              1.000000  0.00026  0.0019 (1)          0.0040    0.0006 / 0.0047   0.0007 / 0.0023
    bf16 input and fn           1.000000  0.00026  0.0020 (1)          0.0042    0.0007 / 0.0048   0.0007 / 0.0025
    bf16 output                 1.000000  0.00015  0.0016 (4)          0.0077    0.0004 / 0.0092   0.0004 / 0.0037
    sigmoid abs err 1e-4        1.000000  0.00051  42 (1), 0.22 (6)    0.048     0.0009 / 0.0051   0.013 / 0.14
    post = 1 * sigmoid (not 2x) 0.999438  0.0321   0.50 (4-7)          0.50      -                 0.22 / 0.41
    no hc_eps                   1.000000  0.00026  0.42 (1)            0.0040    0.0006 / 0.0047   0.0007 / 0.0023
    post x 1.01                 1.000000  0.00069  0.0102 (6, 7)       0.0141    -                 0.0044 / 0.0095
    pre x 1.01                  1.000000  0.00998  0.0103 (0)          -         0.010 / 0.014     -
    gate 0 zeroed               0.999995  0.00301  1.0 (0)             -         0.0032 / 0.049    -
    gate 1 zeroed               1.000000  0.00026  1.0 (1)             -         0.0006 / 0.0047   -
    gate 1 = gate 0             0.999995  0.00301  1261 (1)            -         0.0026 / 0.041    -
    gate 4 / 5 / 6 zeroed       0.999999  <=0.0016  1.0                 0.05-0.17 -                 0.04-0.10 / 0.15-0.34
    gate 7 zeroed               0.997751  0.0641   1.0 (7)             1.0       -                 0.43 / 0.83
    base 0 / 1 swapped          1.000000  0.00048  0.14 (0, 1)         -         0.0008 / 0.0064   -
    base 1 / 2 swapped          1.000000  0.00026  3.6 (1)             -         0.0007 / 0.0047   -
    base 4 / 5 swapped          0.999999  0.00108  0.62 (5)            0.11      -                 0.061 / 0.21
    base 6 / 7 swapped          0.999621  0.0262   0.41 (7)            0.44      -                 0.18 / 0.34
    fn rows 4 / 5 swapped       0.999998  0.00196  1.5 (4)             0.32      -                 0.083 / 0.35
    fn streams 0 / 1 swapped    0.999954  0.00907  0.19 (0)            0.10      0.030 / 0.12      0.019 / 0.076
    fn streams 0 / 2 swapped    0.999748  0.0206   0.35 (0)            0.37      0.070 / 0.27      0.041 / 0.22
    TP: one chip's partial sumsq 0.999016 0.0428   0.90 (6)            0.81      0.10 / 0.33       0.22 / 0.45
    RMS over one stream         0.992929  0.113    1.0                 1.0       0.31 / 1.95       0.40 / 0.82
    rms eps 1e-6 (not 1e-5)     0.999991  0.00391  0.028 (3)           0.30      0.0034 / 0.080    0.0038 / 0.071
    rows 1023 / 1024 swapped    1.000000  0.00085  0.0069 (5)          0.34      0.0027 / 0.24     0.0015 / 0.084
    last row = previous row     0.999998  0.00170  0.024 (4)           0.52      0.012 / 0.42      0.0061 / 0.18
    scales swapped              1.000000  0.00026  0.0021 (1)          0.0042    0.0006 / 0.0047   0.0007 / 0.0021

All but four rows pass the 0.99 PCC gate, and all but eight pass rel L2 0.01. So the test also checks, against the
golden: output finite, element count, rel L2 over [S, 8] <= 0.01, rel L2 per column <= 0.01 (<= 0.02 on the hc_eps
column 1), post worst row rel L2 <= 0.015; the pre gates through ffn_x (rel <= 0.005, worst row <= 0.02) and the post
gates through out (per stream rel <= 0.003, worst row <= 0.02). Every mutation above fails at least one of these,
except bf16 rounding and the scale swap (a no-op on this layer). A dropped hc_eps, gate 1 zeroed and base 1 / 2
swapped change ffn_x and out by < 1e-5 and are caught only by the per-column check on column 1, which asks for ~2 %
relative accuracy on a gate of ~1.4e-6. The out per-stream limit is 0.003 (layer 1: 0.005): post 7 is smaller here
(59 % of stream 3, not 173 %), and at 0.005 post x 1.01 would fail only the per-column limit, by a hair (0.0102).
The device TtHcGates (gate run): PCC 1.000000, rel 0.00029, col rel [0.0030, 0.0064, 0.0, 0.0018, 0.0042, 0.0041,
0.0050, 0.0023], post row 0.0081, ffn_x 0.00070 / 0.0052, out per stream <= 0.00099 / row 0.0033. Tightest margins:
post columns 4-6 (~0.0045 vs 0.01) and the post worst row (0.0081 vs 0.015), from the sigmoid on ~5e-4 gates.
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC_MULT = 4
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_COL_REL = 0.01  # per gate column, rel L2 vs the golden: columns 0, 2, 3, 4, 5, 6, 7
SMALL_COLS = (1,)  # gate at hc_eps (col mean 1.4e-6)
MAX_SMALL_COL_REL = 0.02  # per gate column rel L2 on SMALL_COLS
MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
MAX_FFN_X_REL = 0.005  # ffn_x = hc_pre(h_mid, gates): device gates vs golden gates, rel L2 over [S, H]
MAX_FFN_X_ROW_REL = 0.02  # same, worst row
MAX_OUT_STREAM_REL = 0.003  # out = hc_post(h_mid, gates, golden mlp_out): per stream rel L2 (layer 1: 0.005)
MAX_OUT_ROW_REL = 0.02  # same, worst (row, stream)


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

    # PCC over [S, 8] is dominated by the large columns (pre 2 / 3, post 7); check the error itself.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    err = got - w
    rel = (err.norm() / w.norm()).item()
    col_rel = (err.norm(dim=0) / w.norm(dim=0)).tolist()
    col_err = err.abs().amax(dim=0).tolist()
    post_row = (err[:, HC_MULT:].norm(dim=-1) / w[:, HC_MULT:].norm(dim=-1)).max().item()

    # Pre gates through ffn_hc_pre, post gates through ffn_residual, both on the golden h_mid (and mlp_out).
    h_mid = inputs[0]
    hc_pre = ref.component(LAYER, "ffn_hc_pre")
    fx_want = hc_pre(rctx, h_mid, w).float()
    fx_got = hc_pre(rctx, h_mid, got).float()
    fx_rel = ((fx_got - fx_want).norm() / fx_want.norm()).item()
    fx_row = ((fx_got - fx_want).norm(dim=-1) / fx_want.norm(dim=-1)).max().item()
    hc_post = ref.component(LAYER, "ffn_residual")
    y = gl["mlp_out"].float()
    o_want = hc_post(rctx, h_mid, w, y).float().view(w.shape[0], HC_MULT, -1)
    o_got = hc_post(rctx, h_mid, got, y).float().view(w.shape[0], HC_MULT, -1)
    do = o_got - o_want
    o_stream = (do.norm(dim=(0, 2)) / o_want.norm(dim=(0, 2))).tolist()
    o_row = (do.norm(dim=-1) / o_want.norm(dim=-1)).max().item()

    tag = f"{STEP}_L{LAYER:02d}"
    metrics.record(f"rel_l2_{tag}", rel)
    metrics.record(f"max_col_rel_l2_{tag}", max(col_rel))
    metrics.record(f"post_worst_row_rel_l2_{tag}", post_row)
    metrics.record(f"ffn_x_rel_l2_{tag}", fx_rel)
    metrics.record(f"ffn_x_worst_row_rel_l2_{tag}", fx_row)
    metrics.record(f"out_max_stream_rel_l2_{tag}", max(o_stream))
    metrics.record(f"out_worst_row_rel_l2_{tag}", o_row)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) col rel (pre 0-3 | post 4-7)={[round(v, 5) for v in col_rel]} "
        f"(<= {MAX_COL_REL}, {MAX_SMALL_COL_REL} on {SMALL_COLS}) max abs per column={[f'{v:.2e}' for v in col_err]}"
    )
    print(
        f"post worst row rel={post_row:.5f} (<= {MAX_POST_ROW_REL}); pre via ffn_x: rel={fx_rel:.6f} "
        f"(<= {MAX_FFN_X_REL}) worst row={fx_row:.5f} (<= {MAX_FFN_X_ROW_REL}); post via out: per stream="
        f"{[round(v, 6) for v in o_stream]} (<= {MAX_OUT_STREAM_REL}) worst row={o_row:.5f} (<= {MAX_OUT_ROW_REL})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2}"
    bad = [j for j, v in enumerate(col_rel) if v > (MAX_SMALL_COL_REL if j in SMALL_COLS else MAX_COL_REL)]
    assert not bad, (
        f"gate columns {bad} (pre 0-3, post 4-7) rel L2 above {MAX_COL_REL} "
        f"({MAX_SMALL_COL_REL} on columns {SMALL_COLS}): {col_rel}"
    )
    assert post_row <= MAX_POST_ROW_REL, f"post gates worst row rel L2 {post_row:.5f} > {MAX_POST_ROW_REL}"
    assert fx_rel <= MAX_FFN_X_REL, f"pre gates: ffn_x rel L2 {fx_rel:.5f} > {MAX_FFN_X_REL}"
    assert fx_row <= MAX_FFN_X_ROW_REL, f"pre gates: ffn_x worst row rel L2 {fx_row:.5f} > {MAX_FFN_X_ROW_REL}"
    bad = [j for j, v in enumerate(o_stream) if v > MAX_OUT_STREAM_REL]
    assert not bad, f"post gates: out streams {bad} rel L2 above {MAX_OUT_STREAM_REL}: {o_stream}"
    assert o_row <= MAX_OUT_ROW_REL, f"post gates: out worst row rel L2 {o_row:.5f} > {MAX_OUT_ROW_REL}"

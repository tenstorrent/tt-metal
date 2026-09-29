# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.ffn_hc.test.1). The step's output is the iHC gates [S, 8] fp32 (pre 4 | post 4) from h_mid
[S, 4H] (hc_mlp_layer weights of layer 1); the golden is stored in bf16 (s4096 chunk 1, 2048 rows). The four streams
are distinct (streams 1 / 2 / 3 differ from stream 0 by 29 % / 34 % / 222 %, fn's stream blocks by 58 % / 122 % /
179 %), and the row RMS is small (0.006 .. 0.08), so rms_norm_eps 1e-5 matters. The gate columns span six orders of
magnitude: column means are pre 1.05e-5, 1.45e-6, 0.93, 0.33 and post 1.05e-4, 8.0e-5, 1.6e-6, 0.023. Columns 0, 1
and 6 are within ~10x of hc_eps (1e-6). Post gate j scaled by mlp_out is 2.7 % / 1.0 % / 0.04 % / 173 % of stream j's
norm, so only post 7 moves out much. Measured on this golden (CPU, fp32 math on the golden input; mutations of the
reference). "ffn_x" and "out" are the CPU ffn_hc_pre / ffn_residual on the golden h_mid (and golden mlp_out) with
these gates vs with the golden gates:

    variant                     PCC       rel L2   col rel (worst)   ffn_x rel / row   out max stream / row
    fp32 reference              0.999999  0.00131  0.0028 (6)        0.0013 / 0.0048   0.0013 / 0.0033
    bf16 input and fn           0.999999  0.00131  0.0035 (6)        0.0013 / 0.0048   0.0013 / 0.0031
    bf16 output                 1.000000  0.00031  0.0051 (6)        0.0005 / 0.0087   0.0007 / 0.0065
    sigmoid abs err 1e-4        0.999999  0.00138  66 (1)            0.0013 / 0.0047   0.025 / 0.27
    post = 1 * sigmoid (not 2x) 0.999888  0.0141   0.50 (4-7)        -                 0.40 / 0.48
    no hc_eps                   0.999999  0.00131  0.66 (1), 0.077 (0) -               0.0013 / 0.0033
    post x 1.01                 0.999999  0.00134  0.012 (6)         -                 0.0080 / 0.012
    pre x 1.01                  0.999999  0.0102   0.0102            0.010 / 0.014     -
    gate 0 / 1 / 6 zeroed       0.999999  0.00131  1.0               0.0013 / 0.0048   0.0013 / 0.0033
    gate 4 zeroed               0.999999  0.00133  1.0 (4)           -                 0.028 / 0.16
    gate 7 zeroed               0.999558  0.0280   1.0 (7)           -                 0.79 / 0.95
    gate 3 = gate 2             0.851312  0.639    1.77 (3)          1.16 / 3.7        -
    base 0 / 1 swapped          0.999999  0.00131  6.8 (1)           0.0013 / 0.0048   -
    base 2 / 3 swapped          0.992779  0.110    0.28 (3)          0.16 / 0.41       -
    base 5 / 6 swapped          0.999999  0.00131  0.21 (6)          -                 0.0018 / 0.040
    fn rows 4 / 5 swapped       0.999999  0.00141  3.4 (5)           -                 0.067 / 0.36
    fn rows 6 / 7 swapped       0.999412  0.0310   3156 (6)          -                 1.7 / 6.6
    fn streams 0 / 1 swapped    0.999939  0.0105   0.55 (5)          0.016 / 0.043     0.029 / 0.14
    TP: one chip's partial sumsq 0.996754 0.0959   0.92              0.15 / 0.28       0.59 / 0.74
    RMS over one stream         0.989663  0.162    0.95              0.19 / 0.87       0.56 / 0.88
    rms eps 1e-6 (not 1e-5)     0.999860  0.0166   0.40 (5)          0.0073 / 0.089    0.021 / 0.23
    rows 1023 / 1024 swapped    0.999997  0.00219  0.021 (0)         0.0026 / 0.18     0.0014 / 0.019
    last row = previous row     0.999998  0.00181  0.018 (7)         0.0032 / 0.10     0.017 / 0.49

Almost every row passes the 0.99 PCC gate. So the test also checks, against the golden: output finite, element
count, rel L2 over [S, 8] <= 0.01, rel L2 per column <= 0.01 (<= 0.02 on the near-hc_eps columns 0, 1, 6), post worst
row rel L2 <= 0.015; the pre gates through ffn_x (rel <= 0.005, worst row <= 0.02) and the post gates through out
(per stream rel <= 0.005, worst row <= 0.02). Every mutation above fails at least one of these, bf16 rounding
excepted. Gates 0 / 1 / 6 zeroed, base 0 / 1 swapped and a dropped hc_eps change ffn_x and out by < 1e-4 and are
caught only by the per-column check on the small columns: that check asks for ~1 % relative accuracy on sigmoid
outputs of 1e-6 .. 1e-5, which the fp32 SFPU sigmoid gives (the device TtHcGates: col rel 0.0072 / 0.0035 / 0.0075).
The out per-stream limit is 0.005, not attn_hc's 0.003: out stream 3 carries post 7 x mlp_out at 1.7x the stream
norm, so it tracks post column 7 (device 0.0032 -> out stream 3 0.0026).
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC_MULT = 4
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| over [S, 8]
MAX_COL_REL = 0.01  # per gate column, rel L2 vs the golden: columns 2, 3, 4, 5, 7
SMALL_COLS = (0, 1, 6)  # gates within ~10x of hc_eps (col means 1e-5, 1.5e-6, 1.6e-6)
MAX_SMALL_COL_REL = 0.02  # per gate column rel L2 on SMALL_COLS
MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
MAX_FFN_X_REL = 0.005  # ffn_x = hc_pre(h_mid, gates): device gates vs golden gates, rel L2 over [S, H]
MAX_FFN_X_ROW_REL = 0.02  # same, worst row
MAX_OUT_STREAM_REL = 0.005  # out = hc_post(h_mid, gates, golden mlp_out): per stream rel L2
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

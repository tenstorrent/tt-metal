# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.ffn_residual.test.1). The step is iHC's post after the FFN: out_j = h_mid_j + post_j * mlp_out
for each of the 4 streams j (h_mid / out [S, 4 x 6144], mlp_out [S, 6144], post = ffn_hc gate columns 4-7, fp32
math; the same hc_post as attn_residual). The golden stores every tensor in bf16 (s4096 chunk 1, 2048 rows).
||h_mid|| 285, ||mlp_out|| 137, post 0.33..1.53 (column means 0.49 / 0.58 / 0.48 / 0.72), ||out|| 167 (the addend
partly cancels h_mid). Unlike attn_residual, the h_mid streams differ at layer 0 (max |stream_j - stream_0| up to
0.71). Measured on this golden (CPU, on the golden inputs; per stream j the addend check is delta_j = out_j - h_mid_j
vs t_j = post_j * mlp_out):

    variant                          PCC       rel L2    stream norm ratio   addend coef      addend rel / worst row
    fp32 reference                   0.999990  0.00437   [0.9985, 1.0016]    1.0              0.000 / 0.000
    bf16 output                      0.999989  0.00465   [0.9985, 1.0016]    1.0              <= 0.002 / 0.006
    1.02 x mlp_out                   0.999714  0.024     [0.9896, 1.0050]    1.02             0.02 / 0.02
    1.05 x mlp_out                   0.998240  0.060     [0.9758, 1.0137]    1.05             0.05 / 0.05
    1.1 x mlp_out                    0.992868  0.120     [0.9549, 1.0326]    1.1              0.1 / 0.1
    last row zeroed                  0.999472  0.033     [0.0, 1.0016]       1.0002           <= 0.03 / 0.86
    last 32 columns per stream 0     0.997634  0.069     [0.9943, 1.0000]    1.001            <= 0.066 / 0.25
    mlp_out first / last tile row 0  0.9901 / 0.9874  0.14 / 0.16  [0.9985, 2.19]  0.985     0.12 / 1.0
    h_mid streams 0 / 1 swapped      0.984     0.179     [0.9135, 1.1222]    0.86 / 1.11      0.28 / 0.47
    post columns reversed / one col  0.87 / 0.93  0.57 / 0.40                0.55..1.77       up to 0.84
    attn_hc post gates instead       0.865     0.67      [1.07, 2.02]        0.34..0.55       0.66
    1 x sigmoid (post halved)        0.886     0.60      [1.06, 1.48]        0.5              0.5
    mlp_out or post shifted 1 row    0.70 / 0.92  1.12 / 0.45                0.33..0.85       up to 0.98
    mlp_out SP row halves swapped    0.705     1.12      [0.98, 16.4]        0.37..0.41       0.97
    mlp_out dropped                  0.728     1.20      [1.13, 2.28]        0.0              1.0
    h_mid dropped / layer in used    -0.20 / 0.27  1.71 / 1.48               1.46..2.25       >= 0.92
    zero stub                        nan       1.0       0                   -                -

1.02 x, 1.05 x, 1.1 x, the zeroed last row, the zeroed last 32 columns and a zeroed first / last tile row of mlp_out
pass the 0.99 PCC gate. So the test also checks, against the golden: device output (not a CPU bridge), size,
finite, rel L2 <= 0.01, per-token per-stream norm ratio in [0.99, 1.01]; and per stream, on the addend: coefficient
<delta_j, t_j> / ||t_j||^2 in [0.97, 1.03], ||delta_j - t_j|| / ||t_j|| <= 0.03, and the worst row of that <= 0.05
(bf16 output 0.006; last row zeroed 0.86, last 32 columns 0.25). 1.02 x mlp_out fails rel L2 (0.024).
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0044)
STREAM_NORM_RATIO = (0.99, 1.01)  # per token and stream ||got|| / ||want|| (reference [0.9985, 1.0016])
ADD_COEF = (0.97, 1.03)  # per stream <out_j - h_mid_j, post_j * mlp_out> / ||post_j * mlp_out||^2
MAX_ADD_REL = 0.03  # per stream ||(out_j - h_mid_j) - post_j * mlp_out|| / ||post_j * mlp_out|| (bf16 output 0.002)
MAX_ADD_ROW_REL = 0.05  # the same per token, worst row (bf16 output 0.006)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; ffn_residual is not on the device"
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale and per-row checks (PCC is scale-invariant and barely sees a few bad rows). Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    n = w.shape[0]
    gs, ws = got.view(n, HC, -1), w.view(n, HC, -1)
    ratio = gs.norm(dim=-1) / ws.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) stream_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {STREAM_NORM_RATIO})")
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        STREAM_NORM_RATIO[0] <= rmin and rmax <= STREAM_NORM_RATIO[1]
    ), f"per-token stream norm ratio [{rmin:.4f}, {rmax:.4f}] outside {STREAM_NORM_RATIO} (zeroed rows or streams?)"

    # The addend on each stream: delta_j = out_j - h_mid_j vs post_j * mlp_out (post = ffn_hc columns HC..2HC-1).
    streams, gates, mlp = inputs
    post = gates.float()[:, HC : 2 * HC]
    delta = gs - streams.float().view(n, HC, -1)
    tgt = post.unsqueeze(-1) * mlp.float().view(n, 1, -1)
    err = delta - tgt
    coef = ((delta * tgt).sum(dim=(0, 2)) / (tgt * tgt).sum(dim=(0, 2)).clamp_min(1e-30)).tolist()
    arel = (err.norm(dim=(0, 2)) / tgt.norm(dim=(0, 2)).clamp_min(1e-30)).tolist()
    worst = (err.norm(dim=-1) / tgt.norm(dim=-1).clamp_min(1e-30)).max().item()
    metrics.record(f"add_coef_min_{STEP}_L{LAYER:02d}", min(coef))
    metrics.record(f"add_coef_max_{STEP}_L{LAYER:02d}", max(coef))
    metrics.record(f"add_rel_l2_{STEP}_L{LAYER:02d}", max(arel))
    metrics.record(f"add_worst_row_rel_{STEP}_L{LAYER:02d}", worst)
    print(
        f"addend per stream: coef={[round(v, 4) for v in coef]} (in {ADD_COEF}) rel={[round(v, 4) for v in arel]} "
        f"(<= {MAX_ADD_REL}) worst row={worst:.4f} (<= {MAX_ADD_ROW_REL})"
    )
    bad = [j for j, v in enumerate(coef) if not ADD_COEF[0] <= v <= ADD_COEF[1]]
    assert not bad, f"post * mlp_out coefficient outside {ADD_COEF} on streams {bad}: {coef} (post gates wrong?)"
    bad = [j for j, v in enumerate(arel) if v > MAX_ADD_REL]
    assert not bad, f"addend rel L2 above {MAX_ADD_REL} on streams {bad}: {arel} (mlp_out or post misaligned?)"
    assert worst <= MAX_ADD_ROW_REL, f"worst row addend rel L2 {worst:.4f} > {MAX_ADD_ROW_REL} (a bad row?)"

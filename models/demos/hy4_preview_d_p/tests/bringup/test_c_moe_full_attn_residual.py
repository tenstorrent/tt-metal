# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.attn_residual.test.1). The step is iHC's post: h_mid_j = in_j + post_j * attn_out for each of
the 4 streams j (in / h_mid [S, 4 x 6144], attn_out [S, 6144], post = attn_hc gate columns 4-7, fp32 math, as HF). The
golden stores every tensor in bf16 (2048 rows). Unlike layer 0, the four input streams differ here (stream norms
70 / 65 / 71 / 115), and the post gates of streams 0 and 1 are tiny (column means 3e-4 / 5e-4 / 0.042 / 0.23): the
addend t_j = post_j * attn_out has norm 0.29 / 0.47 / 23 / 103 per stream. On streams 0 and 1 that is below the bf16
resolution of the stream itself: a bf16 output rounds away 32% / 15% of the addend (r_j / ||t_j||, r_j = ||bf16(in_j +
t_j) - (in_j + t_j)||), and even the bf16 golden h_mid scores addend rel 0.50 / 0.26 there. A fixed addend limit (layer
0's 0.03) would fail any bf16-output module, so the addend limits here add twice that rounding budget. Measured on this
golden (CPU, on the golden inputs; "excess" = ||delta_j - t_j|| / (0.01 ||t_j|| + 2 r_j), fail > 1; "row" = the same
per token, 0.05 ||t|| + 2 r, worst row):

    variant                          PCC       rel L2   stream norm ratio   excess per stream           row
    fp32 reference                   0.999997  0.00243  [0.9980, 1.0022]    0 / 0 / 0 / 0               0
    bf16 output                      0.999996  0.00281  [0.9980, 1.0023]    0.49 / 0.48 / 0.26 / 0.17   0.49
    bf16 addcmul (bf16 in, bf16 math)  -       0.0030   [0.9979, 1.0021]    0.49 / 0.48 / 0.28 / 0.21   0.49
    1.01 x attn_out                  -         0.0059   [0.9997, 1.0085]    0.02 / 0.03 / 0.49 / 0.67   0.19   (passes)
    1.02 x attn_out                  0.999955  0.0111   [0.9995, 1.0148]    - / - / 0.98 / 1.34         0.37
    0.995 x in                       -         0.0049   [0.9947, 1.0003]    1.86 / 2.18 / 0.75 / 0.38   24.6
    attn_out dropped on stream 0     -         0.0029   [0.9980, 1.0022]    1.55 / 0 / 0 / 0            4.2
    attn_out dropped on stream 1     -         0.0034   [0.9980, 1.0240]    0 / 3.19 / 0 / 0            12.6
    post columns 0 / 1 swapped       -         0.0043   [0.9781, 1.0235]    2.57 / 3.28 / 0 / 0         52
    addend dropped on the last row   0.999851  0.0174   [0.7568, 1.0022]    - / - / 1.61 / 2.14         18
    last row zeroed                  0.999471  0.0327   [0.0, 1.0022]       11.8 / 13.9 / 4.9 / 3.3     615
    last 32 columns per stream 0     0.997505  0.0710   [0.9944, 1.0001]    -                           731
    1.1 x attn_out                   0.998984  0.0545   [0.9979, 1.0689]    coef 1.1                    -
    input streams 0 / 1 swapped      0.989003  0.149    [0.8672, 1.1551]    108 / 138 / 0 / 0           1621
    1 x sigmoid (post halved)        0.965     0.272    [0.71, 1.02]        24 / 33 on streams 2 / 3    9.3
    attn_out shifted 1 row           0.870     0.512    [0.79, 1.27]        47 / 63 on streams 2 / 3    26
    attn_out dropped                 0.840     0.545    [0.58, 1.05]        -                           -
    zero stub                        nan       1.0      0                   -                           -

The gated PCC (0.99) misses every bug down to the input stream swap. So the test also checks, against the golden:
device output (not a CPU bridge), size, finite, rel L2 <= 0.01, per-token per-stream norm ratio in [0.99, 1.01]; and
per stream on the addend (delta_j = out_j - in_j vs t_j): |coef_j - 1| <= 0.01 + 2 r_j / ||t_j|| with coef_j =
<delta_j, t_j> / ||t_j||^2, ||delta_j - t_j|| <= 0.01 ||t_j|| + 2 r_j, and per token ||delta - t|| <= 0.05 ||t|| +
2 r + 1e-6. The rounding budget r is computed from the golden inputs in fp32, so a module that keeps the streams fp32
(as HF and TtHcPost do) scores 0 and a bf16-output module about 0.5 of the allowance.

Blind spot: 1.01 x attn_out (or post) passes (it is at the bf16 tolerance of the addend on streams 2 / 3).
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0024; 1.02 x attn_out 0.0111)
STREAM_NORM_RATIO = (0.99, 1.01)  # per token and stream ||got|| / ||want|| (reference [0.9980, 1.0022])
ADD_COEF_TOL = 0.01  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j||
MAX_ADD_REL = 0.01  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
MAX_ADD_ROW_REL = 0.05  # per token ||delta - t|| <= this * ||t|| + ROUND_MULT * r + ROW_FLOOR
ROUND_MULT = 2.0  # allowance for the bf16 rounding of the output (a bf16 output uses about half of it)
ROW_FLOOR = 1e-6


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
    ), "device_component returned a CPU bridge; attn_residual is not on the device"
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

    # The addend on each stream: delta_j = out_j - in_j vs t_j = post_j * attn_out (post = attn_hc columns HC..2HC-1).
    # Streams 0 / 1 have addends below the stream's bf16 resolution, so every limit adds ROUND_MULT x the bf16
    # rounding error r of the exact fp32 result (computed here from the golden inputs). The statistics are float64:
    # fp32 sums over 12.6M elements drift by ~2%, enough to move the coefficient of an exact output to 1.03.
    streams, gates, attn = inputs
    xs = streams.float().view(n, HC, -1)
    post = gates.float()[:, HC : 2 * HC]
    tgt32 = post.unsqueeze(-1) * attn.float().view(n, 1, -1)
    exact = xs + tgt32
    rnd = (exact.bfloat16().float() - exact).double()
    tgt = tgt32.double()
    delta = (gs - xs).double()
    err = delta - tgt
    tn = tgt.norm(dim=(0, 2)).clamp_min(1e-30)
    rs = rnd.norm(dim=(0, 2))
    coef = (delta * tgt).sum(dim=(0, 2)) / (tn * tn)
    coef_tol = ADD_COEF_TOL + ROUND_MULT * rs / tn
    coef_x = ((coef - 1).abs() / coef_tol).tolist()  # > 1 fails
    arel = (err.norm(dim=(0, 2)) / tn).tolist()
    excess = (err.norm(dim=(0, 2)) / (MAX_ADD_REL * tn + ROUND_MULT * rs)).tolist()  # > 1 fails
    row_x = err.norm(dim=-1) / (MAX_ADD_ROW_REL * tgt.norm(dim=-1) + ROUND_MULT * rnd.norm(dim=-1) + ROW_FLOOR)
    worst = row_x.max().item()
    worst_at = divmod(int(row_x.argmax().item()), HC)
    metrics.record(f"add_coef_min_{STEP}_L{LAYER:02d}", coef.min().item())
    metrics.record(f"add_coef_max_{STEP}_L{LAYER:02d}", coef.max().item())
    metrics.record(f"add_rel_l2_{STEP}_L{LAYER:02d}", max(arel))
    metrics.record(f"add_excess_{STEP}_L{LAYER:02d}", max(excess))
    metrics.record(f"add_worst_row_excess_{STEP}_L{LAYER:02d}", worst)
    print(
        f"addend per stream: coef={[round(v, 4) for v in coef.tolist()]} (|coef-1|/tol={[round(v, 3) for v in coef_x]}"
        f" <= 1) rel={[round(v, 4) for v in arel]} bf16 budget/||t||={[round(v, 4) for v in (rs / tn).tolist()]} "
        f"excess={[round(v, 3) for v in excess]} (<= 1) worst row excess={worst:.3f} (<= 1) at (row, stream)={worst_at}"
    )
    bad = [j for j, v in enumerate(coef_x) if v > 1]
    assert not bad, f"post * attn_out coefficient off on streams {bad}: {coef.tolist()} (post gates wrong?)"
    bad = [j for j, v in enumerate(excess) if v > 1]
    assert not bad, f"addend error above the limit on streams {bad}: excess {excess} (attn_out or post misaligned?)"
    assert worst <= 1, f"worst row addend error {worst:.3f} x the limit at (row, stream) {worst_at} (a bad row?)"

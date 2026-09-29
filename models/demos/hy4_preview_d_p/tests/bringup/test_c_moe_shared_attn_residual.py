# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.attn_residual.test.1). The step is iHC's post, the same as moe_full layer 1: h_mid_j = in_j +
post_j * attn_out for each of the 4 streams j (in / h_mid [S, 4 x 6144], attn_out [S, 6144], post = attn_hc gate
columns 4-7, fp32 math as in HF). The golden stores every tensor in bf16 (2048 rows). At layer 2 the streams differ
(norms 70 / 65 / 73 / 335). Post column means are 0.21 / 0.24 / 0.22 / 0.0028, so the addend t_j = post_j * attn_out
has norm 37 / 45 / 40 / 0.64 per stream. That is large on streams 0-2 (bf16 rounding budget r_j / ||t_j|| 0.0023 /
0.0015 / 0.0023). On stream 3 it is below the bf16 resolution of the stream (budget 0.12). So the layer-1
rounding-aware addend checks apply unchanged in form, and the fixed parts are tightened to 0.005 (layer 1: 0.01).
That closes layer 1's blind spot (1.01 x attn_out). Measured on this golden (CPU, golden inputs; "excess" =
||delta_j - t_j|| / (0.005 ||t_j|| + 2 r_j), fail > 1; "row" = the same per token, 0.05 ||t|| + 2 r, worst row):

    variant                          PCC       rel L2   stream norm ratio   excess per stream           row   verdict
    fp32 reference                   0.999999  0.00101  [0.9989, 1.0015]    0 / 0 / 0 / 0               0     pass
    bf16 output                      0.999999  0.00111  [0.9989, 1.0015]    0.24 / 0.19 / 0.24 / 0.49   0.49  pass
    bf16 addcmul (bf16 in and math)  0.999999  0.00116  [0.9989, 1.0014]    0.30 / 0.28 / 0.30 / 0.49   0.49  pass
    1.005 x attn_out                 0.999999  0.00144  [0.9979, 1.0037]    0.53 / 0.62 / 0.52 / 0.02   0.10  pass
    1.01 x attn_out (0.99 x same)    0.999997  0.00228  [0.9967, 1.0058]    1.05 / 1.24 / 1.04 / 0.04   0.19  fail
    1.02 x attn_out                  0.999991  0.00422  [0.9943, 1.0103]    2.1 / 2.5 / 2.1 / 0.08      0.38  fail
    0.999 x in                       0.999999  0.00144  [0.9978, 1.0009]    0.13 / 0.11 / 0.12 / 2.1    33    fail
    attn_out dropped on stream 3     0.999998  0.00210  [0.9989, 1.0180]    0 / 0 / 0 / 4.0             9.1   fail
    addend dropped on the last row   0.999976  0.00697  [0.9989, 1.5925]    2.2 / 2.6 / 2.4 / 0         19    fail
    post columns 0 / 2 swapped       0.999832  0.0184   [0.9128, 1.1294]    8.3 / 0 / 7.7 / 0           6.6   fail
    last row zeroed                  0.999421  0.0342   [0.0, 1.0015]       2.8 / 2.2 / 2.9 / 72        19422 fail
    post columns 0 / 1 swapped       0.999370  0.0356   [0.8871, 1.1695]    16 / 15 / 0 / 0             13    fail
    1.1 x attn_out                   0.999790  0.0205   [0.9755, 1.0520]    6.9 / 7.6 / 6.8 / 0.4       1.9   fail
    last 32 columns per stream 0     0.997426  0.0721   [0.9943, 0.9993]    6.7 / 5.2 / 6.9 / 150       2384  fail
    input streams 0 / 1 swapped      0.996614  0.0824   [0.7516, 1.3308]    37 / 34 / 0 / 0             18    fail
    output streams 0 / 1 swapped     0.995788  0.0920   [0.7108, 1.4067]    41 / 38 / 0 / 0             22    fail
    1 x sigmoid (post halved)        0.994851  0.102    [0.9008, 1.2448]    34 / 38 / 34 / 2.0          9.5   fail
    attn_out dropped on stream 0     0.994274  0.108    [0.9834, 1.5587]    69 / 0 / 0 / 0              19    fail
    post col 3 = col 2               0.993459  0.116    [0.9192, 1.0015]    0 / 0 / 0 / 250             5507  fail
    attn_out dropped on stream 1     0.991673  0.131    [0.9990, 1.6632]    0 / 76 / 0 / 0              19    fail
    post columns 2 / 3 swapped       0.986608  0.163    [0.9131, 1.4383]    0 / 0 / 68 / 250            5507  fail
    post shifted 1 row               0.984413  0.180    [0.8969, 11.646]    59 / 68 / 60 / 6.1          282   fail
    attn_out dropped                 0.980030  0.205    [0.9165, 1.6632]    69 / 76 / 68 / 4.0          19    fail
    attn_out shifted 1 row           0.977276  0.220    [0.9999, 1.8569]    74 / 82 / 73 / 4.8          26    fail
    attn_out row halves swapped (SP) 0.976921  0.222    [0.9899, 1.9077]    75 / 83 / 74 / 4.5          26    fail
    attn_out column halves swapped   0.961105  0.290    [1.0000, 2.1255]    97 / 108 / 97 / 5.6         28    fail
    pre gates used as post           0.956769  0.306    [0.8978, 15.46]     69 / 153 / 69 / 28          538   fail
    input streams 2 / 3 swapped      0.072749  1.36     [0.0658, 16.17]     0 / 0 / 566 / 2073          33561 fail
    zero stub                        nan       1.0      0                   94 / 71 / 95 / 2098         32994 fail

The gated PCC (0.99) passes every bug above "post columns 2 / 3 swapped" in the table. So the test also
checks, against the golden: device output (not a CPU bridge), size, finite, rel L2 <= 0.005, per-token per-stream norm
ratio in [0.995, 1.005]; and per stream on the addend (delta_j = out_j - in_j vs t_j): |coef_j - 1| <= 0.005 +
2 r_j / ||t_j|| with coef_j = <delta_j, t_j> / ||t_j||^2, ||delta_j - t_j|| <= 0.005 ||t_j|| + 2 r_j, and per token
||delta - t|| <= 0.05 ||t|| + 2 r + 1e-6. The rounding budget r is the bf16 rounding error of the exact fp32 result on
the golden inputs (float64 statistics), so an fp32-output module (TtHcPost) scores 0 and a bf16-output module about
0.5 of the allowance. The bf16 golden h_mid itself fails the addend checks (stream 3 excess 1.18, row 10.6): it was
rounded from unrounded fp32 inputs. It is not a module output and is not the bar.

Blind spot: 1.005-1.007 x attn_out (or post) passes (inside the bf16 tolerance of streams 0-2).
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.005  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0010; 0.995 x in 0.0052)
STREAM_NORM_RATIO = (0.995, 1.005)  # per token and stream ||got|| / ||want|| (reference [0.9989, 1.0015])
ADD_COEF_TOL = 0.005  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j||
MAX_ADD_REL = 0.005  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
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
    # Stream 3 has an addend below the stream's bf16 resolution, so every limit adds ROUND_MULT x the bf16
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

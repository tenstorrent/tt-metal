# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.ffn_residual.test.1). The step is iHC's post after the MoE: out_j = h_mid_j + post_j * mlp_out
for each of the 4 streams j (h_mid / out [S, 4 x 6144], mlp_out = moe_combine [S, 6144], post = ffn_hc gate columns
4-7, fp32 math; the same hc_post as attn_residual). The golden stores every tensor in bf16 (s4096 chunk 1, 2048 rows).
Stream norms 70 / 65 / 73 / 153, ||mlp_out|| 8216, post column means 1.1e-4 / 8.0e-5 / 1.6e-6 / 0.023 (max 0.081):
the addend t_j = post_j * mlp_out has norm 1.94 / 0.63 / 0.031 / 265 per stream. On streams 0-2 the addend is at or
below the bf16 resolution of the stream: a bf16 output rounds away 6% / 15% / 72% of it (r_j / ||t_j||, r_j =
||bf16(h_mid_j + t_j) - (h_mid_j + t_j)||). So the addend limits add twice that rounding budget, as in the layer-1
attn_residual test. Measured on this golden (CPU, on the golden inputs; "excess" = ||delta_j - t_j|| / (0.01 ||t_j||
+ 2 r_j), fail > 1; "coef" = |coef_j - 1| / (0.01 + 2 r_j / ||t_j||), fail > 1; "row" = per token ||delta - t|| /
(0.05 ||t|| + 2 r + 1e-6), worst row, fail > 1; statistics in float64):

    variant                        PCC       rel L2   stream norm ratio   coef (worst)  excess per stream         row
    fp32 reference                 0.999997  0.00253  [0.9973, 1.0030]    0             0 / 0 / 0 / 0             0
    bf16 output                    0.999996  0.00293  [0.9972, 1.0030]    0.26          0.46 / 0.48 / 0.50 / 0.15 0.49
    bf16 addcmul (bf16 math)       0.999995  0.00328  [0.9970, 1.0028]    0.26          0.46 / 0.48 / 0.50 / 0.20 0.49
    1.01 x mlp_out                 0.999989  0.0079   [0.9997, 1.0113]    0.70          - / - / - / 0.70          0.19
    1.02 x mlp_out                 0.999966  0.0152   [0.9996, 1.0203]    1.41          - / - / - / 1.41          0.37
    1.1 x mlp_out                  0.999295  0.0746   [0.9983, 1.0954]    7.0           - / - / - / 7.0           1.87
    0.995 x h_mid                  0.999995  0.0037   [0.9948, 1.0020]    0.06          1.4 / 1.7 / 8.0 / 0.2     58
    mlp_out dropped on stream 0    0.999982  0.0060   [0.9877, 1.0055]    7.7           7.7 / 0 / 0 / 0           14
    mlp_out dropped on stream 1    0.999995  0.0031   [0.9828, 1.0273]    3.3           0 / 3.3 / 0 / 0           15
    mlp_out dropped on stream 2    0.999997  0.0025   [0.9973, 1.0030]    0.69          0 / 0 / 0.69 / 0        0.99 (!)
    post columns 0 / 1 swapped     0.999973  0.0074   [0.9541, 1.0271]    6.6           7.0 / 9.3 / 0 / 0         95
    rows 1023 / 1024 swapped       0.999804  0.0199   [0.7448, 1.3427]    0.01          4.6 / 5.6 / 27 / 1.8      2446
    addend dropped on the last row 0.999664  0.0260   [0.4400, 1.0030]    0.09          - / - / - / 2.45          18
    last row zeroed                0.999421  0.0342   [0.0, 1.0030]       0.1           8.8 / 11 / 50 / 3.1       3349
    last 32 columns per stream 0   0.997442  0.0718   [0.9936, 1.0008]    0.41          20 / 23 / 112 / 6.4       801
    mlp_out first / last tile row 0  0.9971 / 0.9939  0.077 / 0.111  [0.34, 1.03]  1.56  up to 10.5  18.6
    h_mid streams 0 / 1 swapped    0.996723  0.0811   [0.8586, 1.1498]    17.5          81 / 108 / 0 / 0          440
    output streams 0 / 1 swapped   0.996800  0.0801   [0.8406, 1.1897]    16            80 / 107 / 0 / 0          448
    1 x sigmoid (post halved)      0.960429  0.373    [0.54, 1.01]        35            up to 35                  9.4
    mlp_out dropped on stream 3    0.683653  0.746    [0.19, 1.00]        70            0 / 0 / 0 / 70            19
    mlp_out or post shifted 1 row  0.53 / 0.85  0.91 / 0.53  -            63 / 33       up to 87                  95+
    mlp_out SP row halves swapped  0.514244  0.926    [0.25, 2.58]        64            up to 87                  105
    post s2 / s3 swapped, reversed, ffn_hc pre gates as post, post = 1: PCC <= 0.42, rel >= 1.05
    zero stub                      nan       1.0      0                   -             -                         -

The gated PCC (0.99) misses every bug down to the h_mid stream swap. So the test also checks, against the golden:
device output (not a CPU bridge), size, finite, rel L2 <= 0.01, per-token per-stream norm ratio in [0.99, 1.01], and
the rounding-aware addend checks above (limits as attn_residual layer 1). 1.01 x mlp_out fails the stream ratio
(1.0113); 1.02 x fails rel L2, the ratio and the addend.

Stream 2 (post ~1.6e-6, addend 0.031) is invisible on the golden: dropping its addend scores excess 0.69 / row 0.99,
inside the limits. So the test runs the module a second time with each row's post gates rotated by (row mod 4): every
stream meets the large gate 3 on a quarter of the rows (addend norm 131 / 139 / 133 / 133, bf16 budget 0.2% of it),
and compares with the CPU step on the same inputs, with the same addend checks and rel L2 <= 0.005, per-token stream
norm ratio [0.995, 1.005]:

    variant (rotated gates)        PCC       rel L2   stream norm ratio   coef (worst)  excess (worst)  row
    bf16 output                    0.999999  0.00155  [0.9994, 1.0005]    0             0.16            0.49
    bf16 addcmul                   0.999998  0.00207  [0.9983, 1.0005]    0             0.20            0.49
    post x 1.005                   0.999997  0.0039   [0.9998, 1.0051]    0.37          0.37            0.09  (passes)
    post x 1.01                    0.999990  0.0079   [0.9995, 1.0102]    0.73          0.73            0.19
    mlp_out dropped on stream 0-3  <= 0.924  >= 0.386  min <= 0.28        >= 68         >= 68           >= 18.7
    h_mid streams 0 / 1 swapped    0.996354  0.0856   [0.8586, 1.1606]    1.49          11.5            2972
    output streams 1 / 2 swapped   0.676569  0.806    [0.07, 14.8]        74            107             23195
    post 0 / 1 or 1 / 2 swapped    <= 0.69   >= 0.79  -                   >= 72         >= 100          >= 23166
    rotation ignored               0.564018  0.964    [0.06, 5.38]        73            119             23845

Blind spot: a uniform post (or mlp_out) scale of 1.005 passes both runs (at the bf16 tolerance of stream 3's addend).
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0025; 1.02 x mlp_out 0.0152)
STREAM_NORM_RATIO = (0.99, 1.01)  # per token and stream ||got|| / ||want|| (reference [0.9973, 1.0030])
ADD_COEF_TOL = 0.01  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j||
MAX_ADD_REL = 0.01  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
MAX_ADD_ROW_REL = 0.05  # per token ||delta - t|| <= this * ||t|| + ROUND_MULT * r + ROW_FLOOR
ROUND_MULT = 2.0  # allowance for the bf16 rounding of the output (a bf16 output uses about half of it)
ROW_FLOOR = 1e-6
ROT_MAX_REL_L2 = 0.005  # rotated post gates, vs the CPU step on the same inputs (bf16 addcmul 0.0021)
ROT_STREAM_NORM_RATIO = (0.995, 1.005)  # rotated, per token and stream (bf16 addcmul [0.9983, 1.0005])


def _rotated(gates: torch.Tensor) -> torch.Tensor:
    """Each row's post gates rotated by (row mod 4), so every stream meets the large gate 3 on a quarter of the rows."""
    n = gates.shape[0]
    idx = (torch.arange(HC)[None, :] + torch.arange(n)[:, None]) % HC
    gs = gates.clone()
    gs[:, HC : 2 * HC] = torch.gather(gates[:, HC : 2 * HC], 1, idx)
    return gs


def _check(tag: str, out, want: torch.Tensor, inputs, max_rel: float, ratio_lim) -> None:
    """Size, finite, rel L2, per-token stream norm ratio vs want; rounding-aware addend checks vs the inputs."""
    assert out.numel() == want.numel(), f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), f"{tag}: non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    n = w.shape[0]
    gs, ws = got.view(n, HC, -1), w.view(n, HC, -1)
    ratio = gs.norm(dim=-1) / ws.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    sfx = f"{STEP}_L{LAYER:02d}"
    metrics.record(f"{tag}rel_l2_{sfx}", rel)
    metrics.record(f"{tag}row_norm_ratio_min_{sfx}", rmin)
    metrics.record(f"{tag}row_norm_ratio_max_{sfx}", rmax)
    print(
        f"{tag or 'golden_'}: pcc={metrics.pcc(got, w):.6f} rel_l2={rel:.6f} (<= {max_rel}) "
        f"stream_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ratio_lim})"
    )
    assert rel <= max_rel, f"{tag}: relative L2 error {rel:.4f} > {max_rel} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ratio_lim[0] <= rmin and rmax <= ratio_lim[1]
    ), f"{tag}: per-token stream norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ratio_lim} (zeroed rows or streams?)"

    # The addend on each stream: delta_j = out_j - h_mid_j vs t_j = post_j * mlp_out (post = ffn_hc columns 4-7).
    # Small-gate streams have addends below the stream's bf16 resolution, so every limit adds ROUND_MULT x the bf16
    # rounding error r of the exact fp32 result (computed from the inputs). The statistics are float64: fp32 sums over
    # 12.6M elements drift by ~2%, enough to move the coefficient of an exact output to 1.03.
    streams, gates, mlp = inputs
    xs = streams.float().view(n, HC, -1)
    post = gates.float()[:, HC : 2 * HC]
    tgt32 = post.unsqueeze(-1) * mlp.float().view(n, 1, -1)
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
    metrics.record(f"{tag}add_coef_min_{sfx}", coef.min().item())
    metrics.record(f"{tag}add_coef_max_{sfx}", coef.max().item())
    metrics.record(f"{tag}add_rel_l2_{sfx}", max(arel))
    metrics.record(f"{tag}add_excess_{sfx}", max(excess))
    metrics.record(f"{tag}add_worst_row_excess_{sfx}", worst)
    print(
        f"{tag or 'golden_'} addend per stream: coef={[round(v, 4) for v in coef.tolist()]} "
        f"(|coef-1|/tol={[round(v, 3) for v in coef_x]} <= 1) rel={[round(v, 4) for v in arel]} "
        f"bf16 budget/||t||={[round(v, 4) for v in (rs / tn).tolist()]} excess={[round(v, 3) for v in excess]} (<= 1) "
        f"worst row excess={worst:.3f} (<= 1) at (row, stream)={worst_at}"
    )
    bad = [j for j, v in enumerate(coef_x) if v > 1]
    assert not bad, f"{tag}: post * mlp_out coefficient off on streams {bad}: {coef.tolist()} (post gates wrong?)"
    bad = [j for j, v in enumerate(excess) if v > 1]
    assert (
        not bad
    ), f"{tag}: addend error above the limit on streams {bad}: excess {excess} (mlp_out or post misaligned?)"
    assert worst <= 1, f"{tag}: worst row addend error {worst:.3f} x the limit at (row, stream) {worst_at} (a bad row?)"


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
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale, per-row and addend checks vs the golden (PCC is scale-invariant and barely sees a few bad rows).
    _check("", out, want, inputs, MAX_REL_L2, STREAM_NORM_RATIO)

    # Stream 2's post gate is ~1.6e-6 on this golden, so its addend is invisible: rotate the post gates per row so
    # every stream meets the large gate 3, and compare with the CPU step on the same inputs.
    streams, gates, mlp = inputs
    rot = (streams, _rotated(gates), mlp)
    rot_want = ref.component(LAYER, STEP)(rctx, *rot).float()
    rot_out = fn(rctx, dctx, *rot)
    _check("rot_", rot_out, rot_want, rot, ROT_MAX_REL_L2, ROT_STREAM_NORM_RATIO)

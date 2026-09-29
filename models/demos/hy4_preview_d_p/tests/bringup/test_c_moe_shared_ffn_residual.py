# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.ffn_residual.test.1). The step is iHC's post after the MoE, the same as moe_full layer 1:
out_j = h_mid_j + post_j * mlp_out for each of the 4 streams j (h_mid / out [S, 4 x 6144], mlp_out = moe_combine
[S, 6144], post = ffn_hc gate columns 4-7, fp32 math). The golden stores every tensor in bf16 (s4096 chunk 1, 2048
rows). Stream norms 51 / 42 / 56 / 335, ||mlp_out|| 2646, post column means 5.4e-4 / 9.6e-4 / 3.5e-4 / 0.049 (max
0.26): the addend t_j = post_j * mlp_out has norm 3.1 / 4.1 / 2.4 / 199 per stream. The bf16 rounding budget
r_j / ||t_j|| is 0.027 / 0.017 / 0.038 / 0.0038, so unlike layer 1 (stream 2 budget 0.72) every stream's addend is
visible on the golden, and the fixed parts of the layer-1 limits are tightened to 0.005 (as attn_residual layer 2).
Measured on this golden (CPU, golden inputs; "excess" = ||delta_j - t_j|| / (0.005 ||t_j|| + 2 r_j), fail > 1;
"coef" = |coef_j - 1| / (0.005 + 2 r_j / ||t_j||), fail > 1; "row" = per token ||delta - t|| / (0.05 ||t|| + 2 r +
1e-6), worst row, fail > 1; statistics in float64):

    variant                        PCC       rel L2   stream norm ratio   coef (worst)  excess per stream          row
    fp32 reference                 0.999997  0.00229  [0.9983, 1.0024]    0             0 / 0 / 0 / 0              0
    bf16 output                    0.999996  0.00269  [0.9983, 1.0024]    0             0.46 / 0.44 / 0.47 / 0.30  0.49
    bf16 addcmul (bf16 math)       0.999996  0.00292  [0.9982, 1.0023]    0             0.46 / 0.44 / 0.47 / 0.35  0.49
    1.005 x mlp_out                0.999995  0.00354  [0.9997, 1.0060]    0.39          0.47 / 0.45 / 0.47 / 0.50  0.49
    1.01 x mlp_out                 0.999992  0.00512  [0.9996, 1.0096]    0.79          0.49 / 0.51 / 0.49 / 0.84  0.49
    1.02 x mlp_out                 0.999980  0.00896  [0.9992, 1.0168]    1.58          - / - / - / 1.60           0.49
    1.1 x mlp_out                  0.999623  0.0427   [0.9962, 1.0767]    7.9           1.75 / 2.6 / 1.3 / 7.9     1.85
    0.999 x h_mid                  0.999996  0.00282  [0.9978, 1.0021]    0.06          0.54 / 0.51 / 0.55 / 0.33  0.61
    0.995 x h_mid                  0.999995  0.00471  [0.9943, 1.0010]    0.30          1.47 / 1.39 / 1.56 / 0.73  32
    mlp_out dropped on stream 0    0.999975  0.00711  [0.9914, 1.0296]    17            17 / 0.44 / 0.47 / 0.30    15
    mlp_out dropped on stream 1    0.999958  0.00913  [0.9548, 1.0305]    26            0.46 / 26 / 0.47 / 0.30    17
    mlp_out dropped on stream 2    0.999984  0.00569  [0.9983, 1.0458]    12            0.46 / 0.44 / 12 / 0.30    14
    post columns 0 / 1 swapped     0.999978  0.00661  [0.9602, 1.0284]    8.6           11 / 13 / 0.47 / 0.30      76
    post columns 0 / 2 swapped     0.999983  0.00583  [0.9766, 1.0215]    6.2           9.4 / 0.44 / 9.1 / 0.30    46
    addend dropped on the last row 0.999933  0.0116   [0.7699, 1.0173]    0.05          0.66 / 0.74 / 0.53 / 2.1   17
    rows 1023 / 1024 swapped       0.999829  0.0186   [0.7839, 1.2756]    0.02          4.1 / 4.0 / 4.6 / 3.4      377
    last row zeroed                0.999475  0.0326   [0.0, 1.0024]       0.12          8.3 / 8.0 / 9.0 / 5.9      214
    h_mid streams 0 / 1 swapped    0.997699  0.0680   [0.7117, 1.4024]    28            123 / 142 / 0.47 / 0.30    717
    last 32 columns per stream 0   0.997389  0.0725   [0.9947, 1.0002]    0.73          20 / 19 / 21 / 13          332
    1 x sigmoid (post halved)      0.986256  0.213    [0.6459, 1.0213]    39            8.5 / 13 / 6.2 / 39        9.3
    mlp_out dropped on stream 3    0.924512  0.426    [0.3776, 1.0006]    79            0.46 / 0.44 / 0.47 / 79    19
    mlp_out shifted 1 row          0.870068  0.495    [0.4267, 1.7212]    67            up to 92                   100
    mlp_out SP row halves swapped  0.866100  0.502    [0.4611, 1.6943]    68            up to 93                   102
    post columns 2 / 3 swapped     0.803990  0.597    [0.3776, 11.871]    624           up to 1046                 6166
    pre gates used as post         0.141550  5.61     [0.48, 503]         5641          up to 13838                340482
    zero stub                      nan       1.0      0                   -             -                          -

The gated PCC (0.99) misses every bug down to "post halved". So the test also checks, against the golden: device
output (not a CPU bridge), size, finite, rel L2 <= 0.005, per-token per-stream norm ratio in [0.995, 1.005], and the
rounding-aware addend checks above. 1.005 x mlp_out fails the stream ratio (1.0060), 0.995 x h_mid the ratio and the
addend. The bf16 golden itself scores excess 0.64 / row 3.3 against its inputs (rounded from unrounded fp32 inputs);
it is not a module output and is not the bar.

As at layer 1, the module runs a second time with each row's post gates rotated by (row mod 4), so every stream meets
the large gate 3 on a quarter of the rows, and is compared with the CPU step on the same inputs (rel L2 <= 0.004,
per-token stream norm ratio [0.995, 1.005], the same addend limits):

    variant (rotated gates)        PCC       rel L2   stream norm ratio   coef (worst)  excess (worst)  row
    bf16 output                    0.999999  0.00161  [0.9991, 1.0006]    0             0.35            0.49
    bf16 addcmul                   0.999998  0.00184  [0.9981, 1.0006]    0.01          0.38            0.49
    1.005 x mlp_out                0.999997  0.00289  [0.9991, 1.0053]    0.59          0.62            0.49  (fails ratio)
    1.01 x mlp_out                 0.999991  0.00507  [0.9991, 1.0106]    1.17          1.19            0.49
    0.995 x h_mid                  0.999996  0.00459  [0.9939, 1.0003]    0.23          1.10            218
    mlp_out dropped on stream 0-3  <= 0.974  >= 0.23   min <= 0.41        >= 58         >= 58           >= 18.5
    h_mid streams 0 / 1 swapped    0.997059  0.0769   [0.7117, 1.4023]    1.71          16.6            2003
    post columns swapped (any)     <= 0.890  >= 0.47  -                   >= 71         >= 100          >= 1832
    rotation ignored               0.855582  0.588    [0.07, 2.65]        73            80              5235

Blind spot: a uniform post (or mlp_out) scale of about 1.004 or less passes both runs (the bf16 tolerance of stream 3).
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.005  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0023; 1.01 x mlp_out 0.0051)
STREAM_NORM_RATIO = (0.995, 1.005)  # per token and stream ||got|| / ||want|| (reference [0.9983, 1.0024])
ADD_COEF_TOL = 0.005  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j||
MAX_ADD_REL = 0.005  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
MAX_ADD_ROW_REL = 0.05  # per token ||delta - t|| <= this * ||t|| + ROUND_MULT * r + ROW_FLOOR
ROUND_MULT = 2.0  # allowance for the bf16 rounding of the output (a bf16 output uses about half of it)
ROW_FLOOR = 1e-6
ROT_MAX_REL_L2 = 0.004  # rotated post gates, vs the CPU step on the same inputs (bf16 addcmul 0.0018)
ROT_STREAM_NORM_RATIO = (0.995, 1.005)  # rotated, per token and stream (bf16 addcmul [0.9981, 1.0006])


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

    # Streams 0-2 have post gates ~5e-4 on this golden (addend budget 2-4% in bf16): rotate the post gates per row so
    # every stream meets the large gate 3, and compare with the CPU step on the same inputs.
    streams, gates, mlp = inputs
    rot = (streams, _rotated(gates), mlp)
    rot_want = ref.component(LAYER, STEP)(rctx, *rot).float()
    rot_out = fn(rctx, dctx, *rot)
    _check("rot_", rot_out, rot_want, rot, ROT_MAX_REL_L2, ROT_STREAM_NORM_RATIO)

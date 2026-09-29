# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.attn_residual.test.1). The step is iHC's post: h_mid_j = in_j + post_j * attn_out for each
of the 4 streams j (in / h_mid [S, 4 x 6144], attn_out [S, 6144], post = attn_hc gate columns 4-7, fp32 math). The
golden stores every tensor in bf16 (s4096 chunk 1, 2048 rows). Unlike a plain residual, the addend is not small here:
||in|| 245, ||attn_out|| 510, post 0.03..0.77 (column means 0.12 / 0.15 / 0.11 / 0.34), ||h_mid|| 285. Measured on
this golden (CPU, on the golden inputs; per stream j the addend check is delta_j = out_j - in_j vs t_j = post_j *
attn_out):

    variant                          PCC       rel L2    stream norm ratio   addend coef      addend rel
    fp32 reference                   0.999996  0.00267   [0.9959, 1.0043]    1.0              0.000
    bf16 output                      0.999996  0.00286   [0.9958, 1.0043]    1.0              <= 0.003
    1.1 x attn_out                   0.997754  0.087     [0.9757, 1.1316]    1.1              0.10
    last row zeroed                  0.999398  0.035     [0.0, 1.0043]       0.999            <= 0.055
    last 32 columns per stream 0     0.997698  0.068     [0.9933, 1.0019]    0.997..0.999     <= 0.115
    1 x sigmoid (post halved)        0.904     0.43      [0.44, 1.19]        0.5              0.5
    post columns reversed / one col  0.78 / 0.89  0.66 / 0.47                0.36..2.7        up to 1.7
    attn_out or post shifted 1 row   0.70 / 0.81  0.85 / 0.62                0.46 / 0.65      0.98 / 0.72
    attn_out SP row halves swapped   0.710     0.84      [0.93, 2.58]        0.47             0.97
    attn_out dropped                 0.572     0.87      [0.46, 1.55]        0.0              1.0
    zero stub                        nan       1.0       0                   -                -

The first three bugs pass the 0.99 PCC gate. So the test also checks, against the golden: device output (not a CPU
bridge), size, finite, rel L2 <= 0.01, per-token per-stream norm ratio in [0.98, 1.02]; and per stream, on the addend:
coefficient <delta_j, t_j> / ||t_j||^2 in [0.97, 1.03], ||delta_j - t_j|| / ||t_j|| <= 0.03, and the worst row of
that <= 0.1 (bf16 output 0.004; last row zeroed 1.57, last 32 columns 0.19).

Blind spot: at layer 0 the four input streams are identical (each is the embedding), so a permutation of the input
streams is invisible; a permutation of the post columns is caught (per-stream coefficient).
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
HC = 4  # iHC streams
MAX_REL_L2 = 0.01  # ||got - want|| / ||want|| (fp32 reference vs the bf16 golden 0.0027)
STREAM_NORM_RATIO = (0.98, 1.02)  # per token and stream ||got|| / ||want|| (reference [0.9959, 1.0043])
ADD_COEF = (0.97, 1.03)  # per stream <out_j - in_j, post_j * attn_out> / ||post_j * attn_out||^2
MAX_ADD_REL = 0.03  # per stream ||(out_j - in_j) - post_j * attn_out|| / ||post_j * attn_out|| (bf16 output 0.003)
MAX_ADD_ROW_REL = 0.1  # the same per token, worst row (bf16 output 0.004)


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

    # The addend on each stream: delta_j = out_j - in_j vs post_j * attn_out (post = attn_hc columns HC..2HC-1).
    streams, gates, attn = inputs
    post = gates.float()[:, HC : 2 * HC]
    delta = gs - streams.float().view(n, HC, -1)
    tgt = post.unsqueeze(-1) * attn.float().view(n, 1, -1)
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
    assert not bad, f"post * attn_out coefficient outside {ADD_COEF} on streams {bad}: {coef} (post gates wrong?)"
    bad = [j for j, v in enumerate(arel) if v > MAX_ADD_REL]
    assert not bad, f"addend rel L2 above {MAX_ADD_REL} on streams {bad}: {arel} (attn_out or post misaligned?)"
    assert worst <= MAX_ADD_ROW_REL, f"worst row addend rel L2 {worst:.4f} > {MAX_ADD_ROW_REL} (a bad row?)"

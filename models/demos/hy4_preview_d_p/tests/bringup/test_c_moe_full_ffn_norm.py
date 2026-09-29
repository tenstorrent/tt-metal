# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_norm of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.ffn_norm.test.1). The step is ffn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) (HF
HYV4RMSNorm, post_attention_layernorm, plain w, eps = rms_norm_eps 1e-5), on ffn_x (the iHC pre-mix of h_mid); its
output feeds the router, the routed experts and the shared expert. The golden (s4096 chunk 1, 2048 x 6144) is bf16;
w in [0.085, 0.209], mean 0.144. Unlike layer 0 (row rms from 0.00083, 76% of rows eps-dominated), layer-1 ffn_x has
row rms in [0.0040, 0.072] (median 0.013): the smallest row's mean(x^2) is 1.6x eps, so both eps and the RMS reduction
are visible on the golden. Measured on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00234   [0.9999, 1.0002]     0.0025
    bf16 math (HF casts)                 0.999989   0.00329   [0.9983, 1.0018]     0.0038
    bf16 everywhere, bf16 rsqrt          0.999989   0.00372   [0.9958, 1.0036]     0.0054
    x 1.01                               0.999986   0.0103    [1.0099, 1.0102]     0.0104
    LayerNorm (centered) instead of RMS  0.999934   0.0104    [0.9998, 1.0002]     0.038
    RMS over half the columns            0.999943   0.0109    [0.9793, 1.0399]     0.040
    eps 1.2e-5 / 8e-6                    0.99994    0.013 / 0.014   [0.963, 1.041]   0.037 / 0.041
    RMS over a quarter of the columns    0.999881   0.0165    [0.9601, 1.0604]     0.060
    last row zeroed                      0.999754   0.0235    [0.0000, 1.0002]     1.0
    eps 2e-5                             0.999053   0.0563    [0.849, 0.999]       0.15
    eps 1e-6 (the latent-norm eps)       0.998501   0.0730    [1.0009, 1.2397]     0.24
    eps 0                                0.998070   0.0833    [1.0010, 1.2784]     0.28
    sum instead of mean                  0.998064   0.986     [0.0128, 0.0163]     0.99
    1 + w / no weight                    0.9966 / 0.9955   7.0 / 6.0   ~8x / ~7x     7.1 / 6.1
    w halves / quarters permuted (TP)    0.9922 / 0.9921   0.125 / 0.127   [0.99, 1.03]   0.13 / 0.14
    input_layernorm's w (wrong weight)   0.987      0.181     [0.888, 0.922]       0.19
    eps 1e-2                             0.838      0.807     [0.051, 0.586]       0.95

Everything above "input_layernorm's w" passes the 0.99 PCC gate. So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015 (the
layer-0 limits; pessimistic bf16 estimate 0.0037 / [0.9958, 1.0036] / 0.0054). Each mutation above fails one of them.

The module then runs twice more on scaled copies of the golden input (rounded to bf16), each compared with the CPU
ffn_norm on the same input, so a module that happens to fit this golden's row-rms range still has to get both regimes
right. x 0.1 (median mean(x^2) 0.17x eps, smallest 0.016x; as in the layer-1 attn_norm test):

    variant (x * 0.1)                    rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00285   [0.9963, 1.0036]     0.0044
    eps 2e-5                             0.188     [0.710, 0.928]       0.29
    eps 1e-4                             0.564     [0.319, 0.639]       0.68
    eps 1e-6                             0.612     [1.082, 2.962]       2.0
    eps 0                                1.17      [1.09, 8.03]         7.0

Limits there: rel L2 <= 0.01, worst row rel L2 <= 0.02. x 30 (mean(x^2) >= 1400x eps, the pure-RMS regime, as in
the layer-0 ffn_norm test):

    variant (x * 30)                     rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00287   [0.9960, 1.0035]     0.0047
    eps 1e-6 / 2e-5                      0.0001    [0.9996, 1.0003]     0.0004
    LayerNorm (centered) instead of RMS  0.00992   [0.9999, 1.0001]     0.038
    x 1.01                               0.0100    [1.0100, 1.0100]     0.010
    RMS over half the columns            0.0116    [0.9787, 1.0402]     0.040
    RMS over a quarter of the columns    0.0174    [0.9598, 1.0611]     0.061
    sum instead of mean                  0.987     [0.0128, 0.0128]     0.99

Limits there: rel L2 <= 0.006, per-row norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015.
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
STEP = "ffn_norm"
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.993, 1.007)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
EPS_SCALE = 0.1  # synthetic input = golden ffn_x * EPS_SCALE (bf16): eps dominates mean(x^2) + eps on most rows
EPS_MAX_REL_L2 = 0.01  # vs the CPU ffn_norm on the same input
EPS_MAX_ROW_REL = 0.02
SYN_SCALE = 30.0  # synthetic input = golden ffn_x * SYN_SCALE (bf16): mean(x^2) >> eps, the RMS reduction dominates
SYN_MAX_REL_L2 = 0.006  # vs the CPU ffn_norm on the same input
SYN_RATIO = (0.993, 1.007)
SYN_MAX_ROW_REL = 0.015


def _errors(got: torch.Tensor, want: torch.Tensor):
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm()).item()
    wn = want.norm(dim=-1).clamp_min(1e-30)
    ratio = got.norm(dim=-1) / wn
    row = ((got - want).norm(dim=-1) / wn).max().item()
    return rel, ratio.min().item(), ratio.max().item(), row


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; ffn_norm is not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    assert torch.isfinite(out.float()).all(), "non-finite output"
    rel, rmin, rmax, row = _errors(out, want)
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"worst_row_rel_l2_{STEP}_L{LAYER:02d}", row)
    print(
        f"golden: rel_l2={rel:.6f} (<= {MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(RATIO)}) "
        f"worst_row_rel_l2={row:.5f} (<= {MAX_ROW_REL})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2} (scale, eps or weight bug)"
    assert RATIO[0] <= rmin and rmax <= RATIO[1], f"row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.5f} > {MAX_ROW_REL}"

    (x,) = inputs

    # eps: the golden input scaled down so eps dominates mean(x^2) + eps on most rows; vs the CPU step on the same input
    xe = (x * EPS_SCALE).bfloat16().float()
    eps_want = ref.component(LAYER, STEP)(rctx, xe).float()
    eps_out = fn(rctx, dctx, xe)
    assert eps_out.numel() == eps_want.numel(), f"synthetic output has {eps_out.numel()} elements (x{EPS_SCALE})"
    assert torch.isfinite(eps_out.float()).all(), f"non-finite output (input x{EPS_SCALE})"
    erel, emin, emax, erow = _errors(eps_out, eps_want)
    metrics.record(f"eps_rel_l2_{STEP}_L{LAYER:02d}", erel)
    metrics.record(f"eps_worst_row_rel_l2_{STEP}_L{LAYER:02d}", erow)
    print(
        f"scaled input x{EPS_SCALE}: rel_l2={erel:.6f} (<= {EPS_MAX_REL_L2}) row norm ratio=[{emin:.5f}, {emax:.5f}] "
        f"worst_row_rel_l2={erow:.5f} (<= {EPS_MAX_ROW_REL})"
    )
    assert erel <= EPS_MAX_REL_L2, f"input x{EPS_SCALE}: relative L2 error {erel:.5f} > {EPS_MAX_REL_L2} (wrong eps?)"
    assert erow <= EPS_MAX_ROW_REL, f"input x{EPS_SCALE}: worst row rel L2 {erow:.5f} > {EPS_MAX_ROW_REL} (wrong eps?)"

    # RMS reduction: the golden input scaled up so mean(x^2) >> eps, vs the CPU step on the same input.
    xs = (x * SYN_SCALE).bfloat16().float()
    syn_want = ref.component(LAYER, STEP)(rctx, xs).float()
    syn_out = fn(rctx, dctx, xs)
    assert syn_out.numel() == syn_want.numel(), f"synthetic output has {syn_out.numel()} elements"
    assert torch.isfinite(syn_out.float()).all(), "non-finite output (scaled input)"
    srel, smin, smax, srow = _errors(syn_out, syn_want)
    metrics.record(f"syn_rel_l2_{STEP}_L{LAYER:02d}", srel)
    metrics.record(f"syn_row_norm_ratio_min_{STEP}_L{LAYER:02d}", smin)
    metrics.record(f"syn_row_norm_ratio_max_{STEP}_L{LAYER:02d}", smax)
    metrics.record(f"syn_worst_row_rel_l2_{STEP}_L{LAYER:02d}", srow)
    print(
        f"scaled input x{SYN_SCALE}: rel_l2={srel:.6f} (<= {SYN_MAX_REL_L2}) row norm ratio=[{smin:.5f}, {smax:.5f}] "
        f"(in {list(SYN_RATIO)}) worst_row_rel_l2={srow:.5f} (<= {SYN_MAX_ROW_REL})"
    )
    assert srel <= SYN_MAX_REL_L2, f"scaled input: relative L2 error {srel:.5f} > {SYN_MAX_REL_L2} (RMS reduction?)"
    assert (
        SYN_RATIO[0] <= smin and smax <= SYN_RATIO[1]
    ), f"scaled input: row norm ratio [{smin:.5f}, {smax:.5f}] outside {list(SYN_RATIO)} (RMS over part of the row?)"
    assert srow <= SYN_MAX_ROW_REL, f"scaled input: worst row rel L2 {srow:.5f} > {SYN_MAX_ROW_REL} (LayerNorm?)"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_norm of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.ffn_norm.test.1). The step is ffn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) (HF
HYV4RMSNorm, post_attention_layernorm, plain w, eps = rms_norm_eps 1e-5), on ffn_x (the iHC pre-mix of h_mid). The
golden (s4096 chunk 1, 2048 x 6144) is bf16; w in [0.022, 0.101], mean 0.028. Unlike attn_norm, ffn_x is small: row
rms in [0.00083, 0.0099], median 0.0017, so 76% of the rows have mean(x^2) < eps and eps dominates the scale. Measured
on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00234   [0.9995, 1.0005]     0.0028
    bf16 math (HF casts)                 1.000000   0.00329   [0.9984, 1.0016]     0.0040
    bf16 everywhere, bf16 rsqrt          1.000000   0.00371   [0.9960, 1.0038]     0.0053
    RMS over half the columns            1.000000   0.00438   [0.9859, 1.0161]     0.016
    LayerNorm (centered) instead of RMS  1.000000   0.00749   [0.9995, 1.0005]     0.023
    x 1.01                               1.000000   0.0103    [1.0095, 1.0105]     0.011
    RMS over a quarter of the columns    1.000000   0.0126    [0.9892, 1.0403]     0.040
    last row zeroed                      0.999601   0.0306    [0.0000, 1.0005]     1.0
    eps 1.2e-5 / 8e-6                    0.99987 / 0.99973   0.052 / 0.064   [0.92, 0.99] / [1.01, 1.11]   0.08 / 0.11
    eps 2e-5                             0.997215   0.193     [0.72, 0.96]         0.28
    eps 1e-6 (the latent-norm eps)       0.968      0.620     [1.04, 2.52]         1.5
    input_layernorm's w (wrong weight)   0.973      3.30      ~4.3x                3.4
    w halves / quarters permuted (TP)    0.957      0.30      [0.84, 0.90]         0.33
    sum instead of mean                  0.930      0.978     [0.013, 0.050]       0.99

Everything down to "eps 1.2e-5 / 8e-6" passes the 0.99 PCC gate. So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015 (bf16
device noise about 0.0037 / [0.996, 1.0038] / 0.0053). eps is well visible on the golden itself.

Because eps dominates the golden, the RMS reduction itself is damped there (RMS over half the columns, the missing
all-reduce of a column-split input, is only rel 0.0044). So the module runs a second time on the golden input scaled
by 30 (rounded to bf16; row rms >= 0.025, mean(x^2) >= 60x eps: the pure-RMS regime), compared with the CPU ffn_norm
on the same input:

    variant (x * 30)                     rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00289   [0.9957, 1.0035]     0.0048
    eps 1e-6 / 2e-5                      0.0034 / 0.0037  [1.0001, 1.0072] / [0.9922, 0.9999]  0.0072 / 0.0078
    RMS over half the columns            0.00614   [0.9807, 1.0234]     0.023
    LayerNorm (centered) instead of RMS  0.00635   [1.0000, 1.0001]     0.023
    x 1.01                               0.0100    [1.0100, 1.0100]     0.010
    RMS over a quarter of the columns    0.0229    [0.9840, 1.0543]     0.054
    sum instead of mean                  0.987     [0.0128, 0.0129]     0.99
    no RMS at all (w * x / sqrt(eps))    25        ~8x - 94x            93

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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.993, 1.007)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
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

    # RMS reduction: the golden input scaled up so mean(x^2) >> eps, vs the CPU step on the same input.
    (x,) = inputs
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

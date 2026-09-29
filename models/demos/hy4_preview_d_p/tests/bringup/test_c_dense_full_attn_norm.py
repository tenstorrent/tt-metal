# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.attn_norm.test.1). The step is attn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) (HF
HYV4RMSNorm, input_layernorm, plain w, eps = rms_norm_eps 1e-5; q_a / kv_a norms use 1e-6). The golden (s4096 chunk
1, 2048 x 6144) is bf16; w in [0.020, 0.225], mean 0.129; row rms of attn_x >= 0.0267. Measured on this golden (CPU;
mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00170   [0.9998, 1.0001]     0.0030
    bf16 math (HF casts)                 0.999982   0.00280   [0.9990, 1.0016]     0.0039
    bf16 everywhere, bf16 rsqrt          0.999966   0.00322   [0.9960, 1.0035]     0.0050
    eps 1e-6 (the latent-norm eps)       0.999992   0.00307   [0.9999, 1.0062]     0.0065
    x 1.01                               1.000000   0.0102    [1.0098, 1.0101]     0.0105
    RMS over half the columns            0.999915   0.0087    [0.9716, 1.0300]     0.030
    RMS over a quarter of the columns    0.999825   0.0180    [0.9458, 1.0464]     0.054
    LayerNorm (centered) instead of RMS  0.999912   0.0115    [0.9997, 1.0004]     0.035
    last row zeroed                      0.999771   0.0229    [0.0000, 1.0001]     1.0
    sum instead of mean                  0.999882   0.987     [0.0128, 0.0128]     0.99
    eps 1e-2                             0.932      0.449     [0.26, 0.97]         0.74
    1 + w / no weight                    0.98 / 0.98  7.8 / 6.8  ~8.5x            8
    w halves / quarters permuted (TP)    0.96       0.28-0.29 [0.98, 1.08]         0.33

Everything from "eps 1e-6" to "sum instead of mean" passes the 0.99 PCC gate. So the test also checks, against the
golden: output finite, shape [S, H], rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row rel L2
<= 0.015 (bf16 device noise about 0.0032 / [0.996, 1.0035] / 0.005).

The golden cannot separate eps 1e-6 from device noise (every row's mean square is >= 25x eps). So the module runs a
second time on the golden input scaled by 0.1 (rounded to bf16), compared with the CPU attn_norm on the same input;
there eps matters:

    variant (x * 0.1)                    rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00287   [0.9958, 1.0035]     0.0048
    eps 1e-6                             0.158     [1.0028, 1.4516]     0.45
    eps 2e-5                             0.090     [0.7946, 0.9970]     0.21
    eps 0                                0.187     [1.0031, 1.5501]     0.55

Limits there: rel L2 <= 0.01, worst row rel L2 <= 0.02.
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
STEP = "attn_norm"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.993, 1.007)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
SYN_SCALE = 0.1  # synthetic input = golden attn_x * SYN_SCALE (bf16), where eps matters
SYN_MAX_REL_L2 = 0.01  # vs the CPU attn_norm on the same input
SYN_MAX_ROW_REL = 0.02


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
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; attn_norm is not on the device"
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
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2} (scale or weight bug)"
    assert RATIO[0] <= rmin and rmax <= RATIO[1], f"row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.5f} > {MAX_ROW_REL}"

    # eps: the golden input scaled down so eps is a visible share of mean(x^2), vs the CPU step on the same input.
    (x,) = inputs
    xs = (x * SYN_SCALE).bfloat16().float()
    syn_want = ref.component(LAYER, STEP)(rctx, xs).float()
    syn_out = fn(rctx, dctx, xs)
    assert syn_out.numel() == syn_want.numel(), f"synthetic output has {syn_out.numel()} elements"
    assert torch.isfinite(syn_out.float()).all(), "non-finite output (scaled input)"
    srel, smin, smax, srow = _errors(syn_out, syn_want)
    metrics.record(f"syn_rel_l2_{STEP}_L{LAYER:02d}", srel)
    metrics.record(f"syn_worst_row_rel_l2_{STEP}_L{LAYER:02d}", srow)
    print(
        f"scaled input x{SYN_SCALE}: rel_l2={srel:.6f} (<= {SYN_MAX_REL_L2}) row norm ratio=[{smin:.5f}, {smax:.5f}] "
        f"worst_row_rel_l2={srow:.5f} (<= {SYN_MAX_ROW_REL})"
    )
    assert srel <= SYN_MAX_REL_L2, f"scaled input: relative L2 error {srel:.5f} > {SYN_MAX_REL_L2} (wrong eps?)"
    assert srow <= SYN_MAX_ROW_REL, f"scaled input: worst row rel L2 {srow:.5f} > {SYN_MAX_ROW_REL} (wrong eps?)"

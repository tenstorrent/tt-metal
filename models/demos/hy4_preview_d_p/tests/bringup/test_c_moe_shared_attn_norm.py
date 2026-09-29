# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.attn_norm.test.1). The step is attn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) (HF
HYV4RMSNorm, input_layernorm, plain w, eps = rms_norm_eps 1e-5; q_a / kv_a norms use 1e-6). The golden (s4096 chunk
1, 2048 x 6144) is bf16; w in [0.084, 0.159], mean 0.134; row rms of attn_x in [0.0031, 0.062] (median 0.0094), so
the smallest row's mean(x^2) is 0.94x eps (11 rows are below eps; layer 1: 2.1x). Measured on this golden (CPU;
mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00235   [0.9999, 1.0001]     0.0025
    bf16 everywhere, bf16 rsqrt          0.999940   0.00373   [0.9954, 1.0043]     0.0057
    eps 1e-6 (the latent-norm eps)       0.995413   0.133     [1.0011, 1.3649]     0.36
    eps 2e-5                             0.998015   0.0827    [0.8126, 0.9987]     0.19
    eps 0                                0.993821   0.155     [1.0013, 1.4355]     0.44
    x 1.01                               1.000000   0.0103    [1.0099, 1.0101]     0.0104
    RMS over half the columns            0.999983   0.0104    [0.9766, 1.0306]     0.031
    RMS over a quarter of the columns    0.999919   0.0160    [0.9681, 1.0605]     0.061
    LayerNorm (centered) instead of RMS  0.999966   0.0113    [0.9998, 1.0001]     0.036
    last row zeroed                      0.999760   0.0243    [0.0000, 1.0001]     1.0
    sum instead of mean                  0.993792   0.986     [0.0128, 0.0183]     0.99
    eps 1e-2                             0.830      0.847     [0.044, 0.53]        0.96
    1 + w / no weight                    0.998 / 0.998  7.5 / 6.5  ~8x            6.5-7.5
    w halves / quarters permuted (TP)    0.996      0.087-0.088 [1.00, 1.015]      0.09

Everything except eps 1e-2 passes the 0.99 PCC gate (w is flat on this layer, so 1 + w, no weight and a permuted w
too). So the test also checks, against the golden: output finite, shape [S, H], rel L2 <= 0.008, every row's norm
ratio in [0.993, 1.007], worst row rel L2 <= 0.015 (the layer 0/1 limits; pessimistic bf16 estimate 0.0037 /
[0.9954, 1.0043] / 0.0057). Each of those mutations fails at least one of them.

Eps is visible on this golden too. The module still runs a second time on the golden input scaled by 0.1 (rounded
to bf16), compared with the CPU attn_norm on the same input, where every row is eps-sensitive:

    variant (x * 0.1)                    rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00287   [0.9959, 1.0036]     0.0047
    eps 1e-6                             0.737     [1.108, 3.037]       2.0
    eps 2e-5                             0.211     [0.709, 0.911]       0.29
    eps 0                                1.61      [1.12, 10.3]         9.3
    RMS over half the columns            0.0046    [0.9904, 1.0195]     0.0195

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
LAYER = 2
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

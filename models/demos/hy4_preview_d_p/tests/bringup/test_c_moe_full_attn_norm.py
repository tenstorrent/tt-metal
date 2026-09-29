# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.attn_norm.test.1). The step is attn_norm [S, H] = w * x * rsqrt(mean(x^2) + 1e-5) (HF
HYV4RMSNorm, input_layernorm, plain w, eps = rms_norm_eps 1e-5; q_a / kv_a norms use 1e-6). The golden (s4096 chunk
1, 2048 x 6144) is bf16; w in [0.034, 0.208], mean 0.127; row rms of attn_x in [0.0045, 0.078] (median 0.0125), so
the smallest row's mean(x^2) is only 2.1x eps (layer 0: >= 70x). Measured on this golden (CPU; mutations of the
reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00234   [0.9999, 1.0001]     0.0025
    bf16 everywhere, bf16 rsqrt          1.000000   0.00381   [0.9953, 1.0046]     0.0058
    eps 1e-6 (the latent-norm eps)       0.998887   0.0683    [1.0008, 1.1896]     0.19
    eps 2e-5                             0.999262   0.0547    [0.8683, 0.9993]     0.13
    eps 0                                0.998544   0.0776    [1.0009, 1.2180]     0.22
    x 1.01                               1.000000   0.0103    [1.0099, 1.0101]     0.0104
    RMS over half the columns            1.000000   0.0107    [0.9835, 1.0289]     0.029
    RMS over a quarter of the columns    1.000000   0.0210    [0.9734, 1.0709]     0.071
    LayerNorm (centered) instead of RMS  1.000000   0.0101    [0.9998, 1.0002]     0.035
    last row zeroed                      0.999742   0.0236    [0.0000, 1.0001]     1.0
    sum instead of mean                  0.998486   0.986     [0.0128, 0.0155]     0.99
    eps 1e-2                             0.820      0.805     [0.055, 0.62]        0.94
    1 + w / no weight                    0.991 / 0.989  7.7 / 6.7  ~8x            7-8
    w halves / quarters permuted (TP)    0.981      0.19      [0.96, 1.02]         0.21

Everything from "eps 1e-6" to "sum instead of mean", and 1 + w, passes the 0.99 PCC gate. So the test also checks,
against the golden: output finite, shape [S, H], rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst
row rel L2 <= 0.015 (the same limits as layer 0; pessimistic bf16 estimate 0.0038 / [0.9953, 1.0046] / 0.0058).

Unlike layer 0, eps is visible on this golden. The module still runs a second time on the golden input scaled by 0.1
(rounded to bf16), compared with the CPU attn_norm on the same input, where every row is eps-sensitive:

    variant (x * 0.1)                    rel L2    per-row norm ratio   worst row rel L2
    bf16 everywhere, bf16 rsqrt          0.00291   [0.9953, 1.0041]     0.0052
    eps 1e-6                             0.613     [1.070, 2.908]       1.9
    eps 2e-5                             0.183     [0.711, 0.936]       0.29
    eps 0                                1.16      [1.08, 7.03]         6.0
    RMS over half the columns            0.0063    [0.9885, 1.0218]     0.022

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
LAYER = 1
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

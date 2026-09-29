# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: mlp of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.mlp.test.1). The step is mlp_out [S, H] = down(silu(gate(x)) * up(x)) (HF HYV4MLP, layer 0
only, intermediate 18432, unclamped: swiglu_limit applies to the routed experts only), on x = ffn_norm. The golden
(s4096 chunk 1, 2048 x 6144) is bf16; weights are bf16 as stored. On the golden the gate pre-activation lies in
[-1.11, 0.42] and up in [-1.25, 1.36], so the SwiGLU runs in its small-argument regime. Measured on this golden
(CPU; mutations of the fp32 reference):

    variant                                      PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                               1.000000   0.00238   [0.9996, 1.0004]     0.0025
    bf16 gate / up / h, fp32 accumulate          1.000000   0.00290   [0.9980, 1.0020]     0.0038
    bfp8 weights                                 0.999993   0.00609   [0.9994, 1.0005]     0.0065
    clamped SwiGLU at 10 (routed-expert act)     1.000000   0.00238   [0.9996, 1.0004]     0.0025
    x 1.01 / x 1.03                              1.000000   0.010 / 0.030  [1.0096, 1.0104] / [1.030, 1.030]  0.011 / 0.031
    HiFi2-like (weights truncated to 5 bits)     0.999992   0.0244    [0.9752, 0.9769]     0.025
    last row zeroed                              0.999427   0.0345    [0.0000, 1.0004]     1.0
    last 32 rows zeroed                          0.991234   0.133     [0.0000, 1.0004]     1.0
    gelu_tanh instead of silu                    0.993664   0.143     [0.8795, 0.9637]     0.18
    input x 0.5                                  0.996052   0.727     [0.258, 0.282]       0.74
    silu(up) * gate (gate / up swapped)          0.976542   0.356     [1.065, 1.376]       0.49
    one K half of down only (missing reduce)     0.915992   0.545     [0.526, 0.571]       0.56
    one of 4 intermediate quarters dropped       0.972573   0.274     [0.818, 0.848]       0.30
    no silu (gate * up)                          0.985583   1.44      [2.13, 2.57]         1.6
    down K halves swapped, gate/up shards mixed, output column halves swapped, rows shifted / SP halves
    swapped, sigmoid instead of silu              < 0.46     > 1.0

Everything down to "input x 0.5" passes the 0.99 PCC gate. So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015 (device
noise estimate: bf16 intermediates 0.0029 / [0.998, 1.002] / 0.0038).

The golden cannot see a clamp (the routed experts' ClampedSiluGlu reused by mistake). So the module runs a second
time on the golden input scaled by 30 (rounded to bf16; gate in [-33, 12.5], up beyond +-10 on 1.5e-4 of the
entries), compared with the CPU mlp on the same input:

    variant (x * 30)                             rel L2    per-row norm ratio   worst row rel L2
    bf16 everything (x, gate / up / h, out)      0.00265   [0.9996, 1.0001]     0.0029
    clamped SwiGLU at 10                         0.00245   [0.9979, 1.0000]     0.021
    x 1.01                                       0.0100    [1.0100, 1.0100]     0.010
    HiFi2-like                                   0.0267    [0.9738, 0.9753]     0.027
    gelu_tanh instead of silu                    0.083     [1.0205, 1.0698]     0.21
    golden-scale output (input not used)         0.999     [0.0010, 0.0019]     0.9993

Limits there: rel L2 <= 0.006, per-row norm ratio in [0.993, 1.007], worst row rel L2 <= 0.012.
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
STEP = "mlp"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.993, 1.007)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
SYN_SCALE = 30.0  # synthetic input = golden ffn_norm * SYN_SCALE (bf16): gate / up reach past +-10 (clamp visible)
SYN_MAX_REL_L2 = 0.006  # vs the CPU mlp on the same input
SYN_RATIO = (0.993, 1.007)
SYN_MAX_ROW_REL = 0.012


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
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; mlp is not on the device"
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
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2} (scale, fidelity or activation bug)"
    assert RATIO[0] <= rmin and rmax <= RATIO[1], f"row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.5f} > {MAX_ROW_REL} (dropped or misplaced rows?)"

    # Large-argument SwiGLU: the golden input scaled up so gate / up pass +-10, vs the CPU step on the same input.
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
    assert srel <= SYN_MAX_REL_L2, f"scaled input: relative L2 error {srel:.5f} > {SYN_MAX_REL_L2}"
    assert (
        SYN_RATIO[0] <= smin and smax <= SYN_RATIO[1]
    ), f"scaled input: row norm ratio [{smin:.5f}, {smax:.5f}] outside {list(SYN_RATIO)}"
    assert srow <= SYN_MAX_ROW_REL, f"scaled input: worst row rel L2 {srow:.5f} > {SYN_MAX_ROW_REL} (clamped SwiGLU?)"

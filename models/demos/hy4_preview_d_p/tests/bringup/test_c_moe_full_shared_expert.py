# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: shared_expert of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.shared_expert.test.1). The step is shared_out [S, H] = down(silu(gate(x)) * up(x)) (HF
HYV4MLP as mlp.shared_experts, one shared expert, intermediate 2048, unclamped: swiglu_limit applies to the routed
experts only), on x = ffn_norm. The golden (s4096 chunk 1, 2048 x 6144) is bf16; weights are bf16 as stored. On the
golden the gate pre-activation lies in [-4.04, 8.07] and up in [-8.17, 9.67], just below the routed clamp at 10.
Measured on this golden (CPU; mutations of the fp32 reference):

    variant                                      PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                               1.000000   0.00183   [0.9995, 1.0005]     0.0024
    bf16 gate / up / h, fp32 accumulate          1.000000   0.00336   [0.9946, 1.0032]     0.0073
    bfp8 weights                                 0.999980   0.00735   [0.9986, 1.0012]     0.0093
    clamped SwiGLU at 10 (routed-expert act)     1.000000   0.00183   [0.9995, 1.0005]     0.0024
    x 1.01 / x 1.03                              1.000000   0.010 / 0.030  [1.0095, 1.0105] / [1.0295, 1.0305]  0.011 / 0.031
    HiFi2-like (weights truncated to 5 bits)     0.999979   0.0267    [0.9726, 0.9794]     0.028
    last row zeroed                              0.999616   0.0296    [0.0000, 1.0005]     1.0
    last 32 rows zeroed                          0.991752   0.129     [0.0000, 1.0005]     1.0
    input x 0.5                                  0.990794   0.774     [0.213, 0.341]       0.79
    gelu_tanh instead of silu                    0.995204   0.115     [0.765, 1.113]       0.38
    no silu (gate * up)                          0.904408   0.717     [1.10, 3.88]         3.0
    one of 4 intermediate quarters dropped       0.903179   0.438     [0.392, 0.963]       0.86
    one K half of down only (missing reduce)     0.778865   0.643     [0.311, 0.880]       0.92
    silu(up) * gate (gate / up swapped)          0.786965   0.627     [0.354, 2.76]        2.1
    down K halves swapped, gate/up shards mixed, output column halves swapped, rows shifted / SP halves
    swapped, sigmoid instead of silu              < 0.64     > 0.77

Everything down to "gelu_tanh" passes the 0.99 PCC gate. So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.99, 1.01], worst row rel L2 <= 0.015 (device
noise estimate: bf16 intermediates 0.0034 / [0.9946, 1.0032] / 0.0073; fp32 intermediates as the reference).

The golden cannot see a clamp (the routed experts' ClampedSiluGlu reused by mistake). So the module runs a second
time on the golden input scaled by 2 (rounded to bf16; gate in [-8.1, 16.2], up beyond +-10 on 3.8e-4 of the
entries), compared with the CPU shared_expert on the same input:

    variant (x * 2)                              rel L2    per-row norm ratio   worst row rel L2
    bf16 everything (x, gate / up / h, out)      0.00316   [0.9968, 1.0029]     0.0056
    clamped SwiGLU at 10                         0.0848    [0.677, 1.0000]      0.42
    x 1.01                                       0.0100    [1.0100, 1.0100]     0.010
    HiFi2-like                                   0.0262    [0.9734, 0.9776]     0.028
    gelu_tanh instead of silu                    0.0689    [0.931, 1.060]       0.31
    golden-scale output (input not used)         0.766     [0.220, 0.342]       0.78

Limits there: rel L2 <= 0.006, per-row norm ratio in [0.99, 1.01], worst row rel L2 <= 0.012.
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
STEP = "shared_expert"
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
SYN_SCALE = 2.0  # synthetic input = golden ffn_norm * SYN_SCALE (bf16): gate / up reach past +-10 (clamp visible)
SYN_MAX_REL_L2 = 0.006  # vs the CPU shared_expert on the same input
SYN_RATIO = (0.99, 1.01)
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
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; shared_expert is not on the device"
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

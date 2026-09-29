# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: shared_expert of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.shared_expert.test.1). Same step and checks as test_c_moe_full_shared_expert.py (layer 1):
shared_out [S, H] = down(silu(gate(x)) * up(x)) (HF HYV4MLP as mlp.shared_experts, intermediate 2048, unclamped:
swiglu_limit applies to the routed experts only), on x = ffn_norm. The golden (s4096 chunk 1, 2048 x 6144) is bf16;
weights are bf16 as stored. The layer-2 input is narrower than layer 1's: gate pre-activation in [-2.91, 5.30], up in
[-5.65, 5.72]. Measured on this golden (CPU; mutations of the fp32 reference):

    variant                                      PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                               0.99998    0.00196   [0.9996, 1.0006]     0.0025
    bf16 gate / up / h, fp32 accumulate          1.000000   0.00358   [0.9957, 1.0031]     0.0062
    bfp8 weights                                 1.000000   0.00803   [0.9988, 1.0012]     0.0102
    clamped SwiGLU at 10 (routed-expert act)     0.99998    0.00196   [0.9996, 1.0006]     0.0025
    x 1.01 / x 1.03                              1.000000   0.0102 / 0.0301  [1.0096, 1.0106] / [1.0296, 1.0306]  0.011 / 0.031
    HiFi2-like (weights truncated to 5 bits)     1.000000   0.0272    [0.9726, 0.9772]     0.028
    last row zeroed                              0.999331   0.0361    [0.0000, 1.0006]     1.0
    last 32 rows zeroed                          0.992244   0.124     [0.0000, 1.0006]     1.0
    input x 0.5                                  0.989961   0.776     [0.214, 0.293]       0.79
    gelu_tanh instead of silu                    0.994289   0.138     [0.901, 1.108]       0.29
    no silu (gate * up)                          0.922626   0.770     [1.22, 3.02]         2.2
    one of 4 intermediate quarters dropped       0.892976   0.457     [0.475, 0.949]       0.79
    silu(up) * gate (gate / up swapped)          0.804332   0.599     [0.440, 1.85]        1.3
    one K half of down only (missing reduce)     0.751049   0.670     [0.289, 0.873]       0.91
    down K halves swapped, gate/up shards mixed, output column halves swapped, rows shifted / SP halves
    swapped, sigmoid instead of silu              < 0.61     > 0.95

Everything down to "gelu_tanh" passes the 0.99 PCC gate. So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.99, 1.01], worst row rel L2 <= 0.015 (bf16
intermediates score 0.0036 / [0.9957, 1.0031] / 0.0062; the layer-1 device module scored 0.0020 / 0.0025).

The golden cannot see a clamp (the routed experts' ClampedSiluGlu reused by mistake). So the module runs a second
time on the golden input scaled up (rounded to bf16), compared with the CPU shared_expert on the same input. At layer
2 a scale of 2 (layer 1's) barely reaches the clamp (gate max 10.59, 1e-6 of the entries past +-10; clamp rel
0.0069), so the scale is 3 (gate in [-8.7, 15.9], 6.6e-5 of gate and 5.0e-5 of up past +-10):

    variant (x * 3)                              rel L2    per-row norm ratio   worst row rel L2
    bf16 everything (x, gate / up / h, out)      0.00335   [0.9962, 1.0030]     0.0057
    clamped SwiGLU at 10                         0.0685    [0.689, 1.0000]      0.40
    x 1.01                                       0.0100    [1.0100, 1.0100]     0.010
    HiFi2-like                                   0.0266    [0.9723, 0.9756]     0.029
    gelu_tanh instead of silu                    0.0656    [1.0048, 1.0839]     0.19
    golden-scale output (input not scaled)       0.905     [0.093, 0.122]       0.91

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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
SYN_SCALE = 3.0  # synthetic input = golden ffn_norm * SYN_SCALE (bf16): gate / up reach past +-10 (clamp visible)
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

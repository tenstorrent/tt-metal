# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: q_a of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.q_a.test.1). Same step and checks as test_c_dense_full_q_a.py (layer 0): q_resid [S, 2048] =
q_a_layernorm(q_a_proj(attn_norm)), q_a_layernorm = w * y * rsqrt(mean(y^2) + 1e-6) (HF HYV4RMSNorm default eps,
not rms_norm_eps 1e-5). The layer-1 golden (s4096 chunk 1, 2048 x 6144 in, 2048 x 2048 out) is bf16; q_a_proj
[2048, 6144]; norm w in [0.057, 0.25], mean 0.166; pre-norm row rms in [0.358, 0.497] (mean square >= 1.3e5 x eps).
The K dim (6144) is split over the mesh columns, so the partials must be summed over axis 1 before the norm. Measured
on this golden (CPU; mutations of the reference):

    variant                                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                            1.000000   0.00180   [0.9997, 1.0003]     0.0020
    bf16 x / W, bf16 pre and out              0.999974   0.00291   [0.9995, 1.0004]     0.0035
    eps 1e-5 / eps 0                          1.000000   0.00180   [0.9997, 1.0003]     0.0020
    x 1.01                                    1.000000   0.0102    [1.0097, 1.0103]     0.0105
    RMS over half the columns                 0.999921   0.0179    [0.9706, 1.0543]     0.054
    norm per K partial, then sum              0.999804   0.826     [1.716, 1.871]       0.87
    last row zeroed                           0.999774   0.0220    [0.0000, 1.0003]     1.0
    last tile row (32 rows) zeroed            0.992180   0.125     [0.0000, 1.0003]     1.0
    no norm weight                            0.991435   4.58      [5.43, 5.73]         4.74
    1 + w                                     0.993858   5.57      [6.42, 6.72]         5.73
    only one K half (missing reduce)          0.918-0.926   0.38-0.40   [0.971, 1.004]  0.50-0.53
    norm w halves swapped                     0.986      0.177     [0.901, 0.956]       0.20
    row halves swapped (SP order)             0.640      0.848     [0.969, 1.032]       1.10

Unlike layer 0, a dropped norm weight and 1 + w pass the 0.99 PCC gate here too (narrower w range). Everything from
"eps 1e-5" to "1 + w" passes PCC, so the test also checks, against the golden: output finite, element count,
rel L2 <= 0.008, every row's norm ratio in [0.994, 1.006], worst row rel L2 <= 0.015 (bf16 estimate 0.0029 /
[0.9995, 1.0004] / 0.0035; the layer-0 device module measured rel 0.00198).

The golden cannot see eps (row mean squares are ~1e5 x eps). So the module runs a second time on the golden input
scaled by 0.01 (rounded to bf16), compared with the CPU q_a on the same input; the pre-norm mean square there is
1.3e-5 .. 2.5e-5, comparable to eps:

    variant (x * 0.01)                        rel L2    per-row norm ratio   worst row rel L2
    bf16 device estimate                      0.00235   [0.9996, 1.0004]     0.0027
    eps 1e-5 (rms_norm_eps)                   0.168     [0.778, 0.861]       0.22
    eps 0                                     0.0258    [1.020, 1.038]       0.038
    eps 2e-6                                  0.0239    [0.966, 0.981]       0.034

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
STEP = "q_a"
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want|| over [S, 2048], vs the golden
RATIO = (0.994, 1.006)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.015  # worst per-row rel L2, vs the golden
SYN_SCALE = 0.01  # synthetic input = golden attn_norm * SYN_SCALE (bf16), where the 1e-6 eps matters
SYN_MAX_REL_L2 = 0.01  # vs the CPU q_a on the same input
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
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; q_a is not on the device"
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
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2} (scale, reduce or norm bug)"
    assert RATIO[0] <= rmin and rmax <= RATIO[1], f"row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.5f} > {MAX_ROW_REL}"

    # eps: the golden input scaled down so eps is a visible share of the pre-norm mean square, vs the CPU step on
    # the same input.
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

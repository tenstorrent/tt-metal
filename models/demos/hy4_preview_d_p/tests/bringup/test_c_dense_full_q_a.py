# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: q_a of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.q_a.test.1). The step is q_resid [S, 2048] = q_a_layernorm(q_a_proj(attn_norm)), with
q_a_layernorm = w * y * rsqrt(mean(y^2) + 1e-6) (HF HYV4RMSNorm default eps 1e-6, not rms_norm_eps 1e-5). The golden
(s4096 chunk 1, 2048 x 6144 in, 2048 x 2048 out) is bf16; q_a_proj [2048, 6144] bf16; norm w in [0.017, 0.352],
mean 0.207; pre-norm row rms in [0.311, 0.531] (mean square >= 9.7e4 x eps). On the 2x2 mesh the K dim (6144) is
split over the columns, so the partial products must be summed over axis 1 before the norm. Measured on this golden
(CPU; mutations of the reference):

    variant                                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                            0.999998   0.00181   [0.9996, 1.0004]     0.0021
    bf16 x / W, fp32 acc, bf16 pre and out    0.999996   0.00293   [0.9994, 1.0006]     0.0038
    bfp8 W + bfp8 x (not the plan)            0.999986   0.00521   [0.9993, 1.0007]     0.0074
    eps 1e-5 / eps 0                          0.999998   0.00181   [0.9996, 1.0004]     0.0021
    x 1.01                                    0.999998   0.0102    [1.0096, 1.0104]     0.0105
    RMS over half the columns                 0.999472   0.0325    [0.9961, 1.0043]     0.066
    norm per K partial, then sum              0.999577   0.807     [1.647, 1.874]       0.87
    last row zeroed                           0.999763   0.0218    [0.0000, 1.0004]     1.0
    last tile row (32 rows) zeroed            0.992172   0.125     [0.0000, 1.0004]     1.0
    no norm weight / 1 + w                    0.972 / 0.982   3.4 / 4.3   ~4.5x / ~5.5x
    only one K half (missing reduce)          0.914-0.917   0.41   [0.954, 1.015]       0.58
    norm w halves swapped                     0.967      0.26      [0.855, 0.982]       0.38
    row halves swapped (SP order) / rolled    0.57       0.93      [0.92, 1.09]         1.24
    K halves of W swapped / out halves swap   0.05 / -0.01   1.3-1.4

Everything from "eps 1e-5" to "last tile row zeroed" passes the 0.99 PCC gate. So the test also checks, against the
golden: output finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.994, 1.006], worst row rel L2
<= 0.015 (pessimistic bf16 device estimate 0.0029 / [0.9994, 1.0006] / 0.0038).

The golden cannot see eps at all (row mean squares are ~1e5 x eps). So the module runs a second time on the golden
input scaled by 0.01 (rounded to bf16), compared with the CPU q_a on the same input; the pre-norm mean square there is
9.7e-6 .. 2e-5, comparable to eps:

    variant (x * 0.01)                        rel L2    per-row norm ratio   worst row rel L2
    bf16 device estimate                      0.00233   [0.9995, 1.0006]     0.0029
    eps 1e-5 (rms_norm_eps)                   0.173     [0.737, 0.874]       0.26
    eps 0                                     0.0272    [1.018, 1.050]       0.050
    eps 2e-6                                  0.0250    [0.956, 0.983]       0.044

Limits there: rel L2 <= 0.01, worst row rel L2 <= 0.02. (At x * 0.1, eps 1e-5 scores only rel 0.0025.)
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
LAYER = 0
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

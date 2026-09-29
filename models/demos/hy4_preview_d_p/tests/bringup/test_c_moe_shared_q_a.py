# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: q_a of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.q_a.test.1). Same step and checks as test_c_moe_full_q_a.py (layer 1) and
test_c_dense_full_q_a.py (layer 0): q_resid [S, 2048] = q_a_layernorm(q_a_proj(attn_norm)), q_a_layernorm =
w * y * rsqrt(mean(y^2) + 1e-6) (HF HYV4RMSNorm default eps, not rms_norm_eps 1e-5). A shared-index layer has its own
q_a (only the indexer is shared), so nothing here depends on layer 1's top-k. The layer-2 golden (s4096 chunk 1,
2048 x 6144 in, 2048 x 2048 out) is bf16; q_a_proj [2048, 6144]; norm w in [0.021, 0.21], mean 0.103; pre-norm row
rms in [0.309, 0.489] (mean square >= 9.6e4 x eps). The K dim (6144) is split over the mesh columns, so the partials
must be summed over axis 1 before the norm. Measured on this golden (CPU; mutations of the reference):

    variant                                   PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                            1.000000   0.00183   [0.9995, 1.0005]     0.0022
    bf16 x / W, bf16 pre and out              1.000000   0.00295   [0.9989, 1.0009]     0.0039
    eps 1e-5 / eps 0                          1.000000   0.00183   [0.9995, 1.0005]     0.0022
    x 1.01                                    1.000000   0.0102    [1.0095, 1.0105]     0.0107
    RMS over half the columns                 0.999846   0.0297    [0.9241, 1.0389]     0.076
    norm per K partial, then sum              0.999754   0.794     [1.662, 1.850]       0.85
    last row zeroed                           0.999784   0.0209    [0.0000, 1.0005]     1.0
    last tile row (32 rows) zeroed            0.992194   0.125     [0.0000, 1.0005]     1.0
    no norm weight                            0.951841   7.52      [7.66, 9.34]         8.41
    1 + w                                     0.961325   8.46      [8.63, 10.29]        9.34
    only one K half (missing reduce)          0.904-0.913   0.41-0.44   [0.944, 1.043]  0.55-0.58
    norm w halves swapped                     0.940      0.348     [0.798, 0.967]       0.41
    row halves swapped (SP order)             0.690      0.787     [0.856, 1.169]       1.17

Everything from "eps 1e-5" to "last tile row zeroed" passes the 0.99 PCC gate (at layer 1 a dropped norm weight and
1 + w did too; here w is smaller, mean 0.103, so they fail PCC). So the test also checks, against the golden: output
finite, element count, rel L2 <= 0.008, every row's norm ratio in [0.994, 1.006], worst row rel L2 <= 0.015 (bf16
estimate 0.0030 / [0.9989, 1.0009] / 0.0039; the layer-0 TtQa on the layer-1 golden measured rel 0.00198, ratio
[0.99885, 1.00041]).

The golden cannot see eps (row mean squares are ~1e5 x eps). So the module runs a second time on the golden input
scaled by 0.01 (rounded to bf16), compared with the CPU q_a on the same input; the pre-norm mean square there is
9.6e-6 .. 2.4e-5, comparable to eps:

    variant (x * 0.01)                        rel L2    per-row norm ratio   worst row rel L2
    bf16 device estimate                      0.00235   [0.9993, 1.0007]     0.0029
    eps 1e-5 (rms_norm_eps)                   0.181     [0.735, 0.857]       0.27
    eps 0                                     0.0288    [1.021, 1.051]       0.051
    eps 2e-6                                  0.0264    [0.956, 0.981]       0.044

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
LAYER = 2
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

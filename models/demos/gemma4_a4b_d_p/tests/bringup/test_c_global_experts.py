# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type global (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.global.experts.test.1): same checks as test_c_sliding_experts.py (C.sliding.experts.test.1). The global layers
share the experts step: experts_out = sum_e router[t, e] * down_e(gelu_tanh(gate_e(x)) * up_e(x)), x = moe_norm [2048, 2816],
router the dense [2048, 128] routing (8 nonzeros per row), 128 experts of width 704. Routing on the layer 5 golden is even
more skewed than layer 0: tokens per expert 0..1915, expert 93 takes 1915 of 2048 tokens, and some experts (e.g. 127) get
none. So dispatch must not cap per-expert capacity. The gated metric is PCC. The test also asserts a finite output,
relative L2 <= 0.03, per-token output-norm ratio within [0.97, 1.03], and per-token relative L2 <= 0.1 on every row.
Measured on this golden (CPU) as PCC / rel L2 / norm ratio / max per-row rel L2:
  CPU reference 0.999996 / 0.0027 / [0.996, 1.003] / 0.0043; bf16 out 0.0030 / 0.0058;
  emulated bfp8 activations and weights 0.99988 / 0.0153 / [0.992, 1.006] / 0.020; exact gelu 0.0028 / 0.0104 (harmless, passes);
  drop expert 0 0.99993 (passes) / 0.012 (passes) / min 0.93 / 0.24; drop the hottest expert (93) 0.73;
  drop experts 96..127 (one chip) 0.944; drop each token's smallest-weight pair 0.9970 (passes) / 0.078 / min 0.83 / 0.52;
  drop token 0's top-1 pair 0.99938 (passes) / 0.035 / min 0.28 / 0.98; last row zeroed 0.99987 (passes) / 0.016 (passes) / min 0 / 1.0;
  last 32 rows zeroed 0.9928 (passes) / 0.119; 2x 0.999996 (passes) / 1.0 / [1.99, 2.01]; capacity 512 tokens per expert 0.74;
  silu 0.91; uniform 1/8 routing weights 0.83.
  Known gap (as at layer 0): renormalizing the routing weights to sum 1 scores 0.99999 / 0.0071 / [0.990, 1.012] / 0.0125 and
  passes; that bug belongs to the router, whose test catches it.
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
STEP = "experts"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.1  # per-token ||got - want|| / ||want||, worst row (dropped (token, expert) pairs)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Dropped-expert, dropped-pair, scale and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = (got - w).norm(dim=-1) / wn
    worst = row_rel.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={worst:.5f} (<= {MAX_ROW_REL_L2}, worst row {row_rel.argmax().item()})"
    )
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (dropped expert, activation or scale bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed rows or dropped pairs?)"
    assert (
        worst <= MAX_ROW_REL_L2
    ), f"worst per-token rel L2 {worst:.4f} > {MAX_ROW_REL_L2} at row {row_rel.argmax().item()} (dropped (token, expert) pair?)"

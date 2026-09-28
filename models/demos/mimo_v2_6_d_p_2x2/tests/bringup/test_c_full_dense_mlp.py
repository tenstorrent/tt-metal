# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: mlp of block type full_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_dense.mlp.test.1, 2x2 run; body taken unchanged from the 1x4 prior's frozen test, same golden):
mlp_out = down(silu(gate(x)) * up(x)), dense SwiGLU 16384, input ffn_norm
[2048, 4096], fp8 block-scaled weights dequantized by the reference. The gated metric is PCC (float output). PCC
misses scale bugs (a 4x all_reduce, a 2x output) and bad rows, so the test also asserts a finite output, relative L2
<= 0.015, per-token output-norm ratio within [0.98, 1.02] and worst per-token relative L2 <= 0.05. Measured on this
golden (row norms 0.59-1.31, measured in the 1x4 run) as PCC / rel L2 / per-token norm ratio / worst row rel:
  CPU reference fp32 0.999998 / 0.0017 / [0.9986, 1.0015] / 0.0024; bf16 everything 0.999998 / 0.0019;
  emulated bfp8 weights 0.999996 / 0.0028; bfp8 weights and activations 0.999980 / 0.0064 / [0.9905, 1.0082] / 0.0126;
  weights truncated to 5 mantissa bits (HiFi2-like) 0.99994 / 0.0141 / [0.981, 1.004];
  1.02x output 0.999998 (passes) / 0.0201 / [1.019, 1.022]; 2x 0.999998 (passes) / 1.0; all_reduce x4 (passes) / 3.0;
  row 0 zeroed 0.99977 (passes) / 0.0214 / min 0; last row zeroed 0.99983 (passes) / 0.0187 / min 0;
  one row x1.1 0.999997 (passes) / 0.0025 (passes) / max 1.100 / worst row 0.100;
  last 32 rows zeroed 0.9922 (passes) / 0.125; one TP shard (4096 of 16384) missing 0.9958 (passes) / 0.26;
  one TP shard doubled 0.9985 (passes) / 0.26; gelu_tanh instead of silu 0.911; relu 0.883; gate/up swapped 0.745;
  down shard order rotated -0.12.
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
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.98, 1.02)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.05  # worst per-token ||got - want|| / ||want||


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

    # Scale and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    wn = w.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"worst_row_rel_l2_{STEP}_L{LAYER:02d}", row_rel)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"worst_row_rel_l2={row_rel:.4f} (<= {MAX_ROW_REL_L2})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale, shard or weight bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"
    assert row_rel <= MAX_ROW_REL_L2, f"worst per-token relative L2 {row_rel:.4f} > {MAX_ROW_REL_L2} (bad rows)"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: mlp of block type sliding (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding.mlp.test.1): mlp_out = down(gelu_tanh(gate(x)) * up(x)), dense 2112-wide MLP, input ffn_norm.
The gated metric is PCC (float output). PCC misses a wrong activation, scale bugs and bad rows, so the test also asserts
relative L2 <= 0.03, per-token output-norm ratio within [0.97, 1.03], and a finite output. Measured on this golden
([2048, 2816], row norms 385-4579) as PCC / rel L2 / per-token norm ratio:
  CPU reference 0.999998 / 0.0018 / [0.998, 1.002]; bf16 in, weights, out 0.999999 / 0.0016;
  emulated bfp8 weights 0.999985 / 0.0056 / [0.997, 1.003]; bfp8 weights and activations 0.999976 / 0.0069 / [0.992, 1.010];
  exact gelu instead of tanh 0.999998 / 0.0018 (numerically harmless);
  silu instead of gelu_tanh 0.9986 (passes) / 0.055; 2x output ~1.0 (passes) / 1.0;
  row 0 zeroed 0.99993 (passes) / 0.012 / min 0; last row zeroed 0.9996 (passes) / 0.027 / min 0;
  last 32 rows zeroed 0.9913 (passes) / 0.131; a quarter of the intermediate dim counted twice 0.986 / 0.37;
  a quarter missing 0.956 / 0.37; gate and up swapped 0.68; no activation 0.72.
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
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||


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

    # Scale, activation and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO})")
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (activation, scale or weight bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed or padded rows?)"

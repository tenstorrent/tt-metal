# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type sliding_moe (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding_moe.attn_norm.test.1, mimo_v2_6_d_p_2x2): adopted from the frozen 1x4 test
(models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attn_norm.py); same golden, same limits. Same checks as
test_c_full_dense_attn_norm.py (the step is the same RMSNorm, plain ``w``, eps 1e-6). PCC is scale-invariant, so the
test also asserts: output finite, relative L2 error <= 0.03, per-token output-norm ratio within [0.97, 1.03]. Measured
on this golden ([2048, 4096] bf16, layer 1 input_layernorm w in [-0.13, 2.47], mean 0.037): fp32 CPU reference rel
0.0024, ratio [0.9954, 1.0043]; bf16 HF math rel 0.0031, ratio [0.9938, 1.0063]. Mutations: sum instead of mean in the
RMS passes PCC (~1.0) but rel 0.98 / ratio 0.016; eps 1e-3 PCC 0.998 but rel 0.36; eps 1e-2 PCC 0.993 but rel 0.74; ``1
+ w`` PCC 0.78; no weight PCC 0.46; one zeroed row PCC ~1.0 but ratio min 0; last 32 rows zeroed PCC 0.993 but rel 0.12;
zero stub fails PCC.
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

    # Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO})")
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale or weight bug, e.g. 1 + w)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}"

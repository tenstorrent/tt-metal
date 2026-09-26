# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: post_ffn_norm of block type sliding (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding.post_ffn_norm.test.1): same checks as test_c_sliding_attn_norm.py. The gated metric is PCC, which is
blind to scale; Gemma-4 RMSNorm is ``rms(x) * w`` (no ``1 + w``), so the extra asserted checks are relative L2 error
<= 0.03 and per-token output-norm ratio within [0.97, 1.03]. Input is ffn_sum (mlp_post_norm + moe_post_norm), output
ffn_out (post_feedforward_layernorm; layer_scalar is applied later, in ffn_residual). Measured on this golden
([2048, 2816], w in about [0.02, 23]): CPU reference PCC 0.999997, rel 0.0023, ratio [0.9978, 1.0017]; bf16 in/out rel
0.0027; 1% noise 0.010; ``1 + w`` PCC 0.99988 (passes) but rel 0.047, ratio [1.042, 1.049]; no weight PCC 0.94; wrong
weight (post_ffn_norm_1 / _2) PCC 0.80 / 0.78; sum instead of mean PCC ~1.0 (passes) but rel 0.98; 2x PCC ~1.0
(passes) but rel 1.0; layer_scalar folded in PCC ~1.0 (passes) but rel 0.93; last row zeroed PCC 0.9997 (passes),
caught by the ratio (min 0); last 32 rows zeroed PCC 0.9922 (passes), rel 0.12.
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
STEP = "post_ffn_norm"
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

    # Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
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
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale or weight bug, e.g. 1 + w)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}"

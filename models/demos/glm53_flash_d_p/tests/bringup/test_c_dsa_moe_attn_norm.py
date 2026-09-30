# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.attn_norm.test.1): the gated metric is PCC (float output, [2048, 4096] bf16 golden), but PCC is
scale-invariant. GLM input_layernorm is plain ``w * rms(x)``, w in [0.0126, 0.0286], eps 1e-5. Input row RMS is
0.0015..0.0099 (mean square 2.1e-6..9.7e-5, at or below eps), so eps matters more than at layer 0. Measured on this
golden, rel L2 / per-token norm ratio / worst row rel L2 (PCC where it fails 0.99): fp32 CPU reference 0.0023 /
[0.9998, 1.0002] / 0.0025; bf16 input, weight and output 0.0028 / 0.0030; sum instead of mean PCC 0.980; eps 0 0.980;
eps 1e-6 0.988; eps 1e-4 0.983; no weight 46; ``1 + w`` 47; ffn_norm's weight 22; layer 0's weight 3.9; w reversed
0.115 / [1.0004, 1.0216]; eps 1.2e-5 0.040 / [0.927, 0.991]; eps 1.1e-5 0.021 / [0.961, 0.995] / 0.039; eps 9e-6
0.022 / [1.005, 1.044] / 0.044; LayerNorm-style mean subtraction 0.0141 / [0.9995, 1.0002] / 0.043; x1.01 0.0103 /
[1.0098, 1.0102]; x1.005 0.0055 / 0.0057 (passes); last row zero 0.018 / ratio 0; last 32 columns zero 0.086.
Device-like noise: squares accumulated in bf16 0.0048 / [0.989, 1.025] / 0.025 (fails the ratio and worst-row limits);
rsqrt with 5e-3 row error 0.0059 / [0.984, 1.021] / 0.021 (fails); 0.3% element noise 0.0042 / [0.9996, 1.0004] /
0.0045 (passes).
Extra checks: finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], worst per-token rel L2 <= 0.015 (the
ffn_norm limits; the worst-row limit catches mean subtraction). Limits are written
``not x <= lim`` so NaN fails.
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
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.015  # worst per-token ||got - want|| / ||want||


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
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_L{LAYER:02d}", row_rel)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={row_rel:.4f} (<= {MAX_ROW_REL_L2})"
    )
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale, eps or weight bug)")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if not row_rel <= MAX_ROW_REL_L2:
        fails.append(f"worst per-token rel L2 {row_rel:.4f} > {MAX_ROW_REL_L2}")
    assert not fails, "; ".join(fails)

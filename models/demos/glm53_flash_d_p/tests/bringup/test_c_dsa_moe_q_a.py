# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: q_a of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.q_a.test.1): q_resid = q_a_layernorm(q_a_proj(attn_norm)), [2048, 4096] -> [2048, 1536] bf16
golden. q_a_proj is fp8 e4m3 with a [12, 32] block weight_scale_inv (1.4e-4..2.4e-4); q_a_layernorm w in
[0.4375, 0.875], eps 1e-5. The projection's row mean square is 1.2e-4..8.1e-4, so eps shifts row scale by 1-4%. PCC
is scale-invariant and the norm removes most projection errors, so almost every bug passes PCC 0.99. Measured on this
golden, PCC / rel L2 / per-token norm ratio / worst row rel L2:
fp32 CPU reference 1.0000 / 0.0021 / [0.9997, 1.0003] / 0.0022; bf16 x, W and output 0.0026 / 0.0030; bf16 projection
before the norm 0.0031 / [0.9995, 1.0005] / 0.0035; bfp8 W 0.0055 / 0.0063; bfp8 x 0.0061 / 0.0071; 7-bit mantissa
truncation of x and W (HiFi2-like) 0.0025; 0.3% element noise 0.0040 / [0.9993, 1.0006] / 0.0044.
Bugs: eps 0 1.0000 / 0.0158 / [1.006, 1.040]; eps 1e-6 1.0000 / 0.0142 / [1.005, 1.035]; eps 1e-4 0.9991 / 0.110;
x1.005 0.0054 / [1.0047, 1.0053]; x1.01 0.0102 / [1.0097, 1.0103]; LayerNorm-style mean subtraction 0.99986 / 0.0229 /
[0.9994, 1.0003] / 0.086; norm over 4 or 2 column shards (a TP split) 0.9994 / 0.037 / 0.095 and 0.9999 / 0.018 /
0.069; fp8 weight_scale_inv ignored 0.9980 / 0.067 / [1.006, 1.040]; scale per row block only 0.9982 / 0.061; scale
columns reversed 0.9962 / 0.088; bfp4 W 0.9970 / 0.079; no norm weight 0.9988 / 0.35; ``1 + w`` 0.9996 / 1.34; norm
weight reversed 0.9963 / 0.089; attn_norm weight divided out of x 0.9978 / 0.070; last row zero 0.99983 / 0.022 /
ratio 0; last 32 columns zero 0.9901 / 0.14. Caught by PCC: no norm, eps 1e-3, 384 columns or 32 rows zeroed, row
shifts, attn_in instead of attn_norm as the input.
Extra checks: finite, rel L2 <= 0.01, per-token norm ratio in [0.995, 1.005] (x1.005 fails at 1.0053; device-like
noise stays within 0.0007), worst per-token rel L2 <= 0.015. Every bug above fails at least one. Limits are written
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
STEP = "q_a"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.995, 1.005)  # per-token ||got|| / ||want||
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
        fails.append(f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale, eps, fp8 dequant or weight bug)")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if not row_rel <= MAX_ROW_REL_L2:
        fails.append(f"worst per-token rel L2 {row_rel:.4f} > {MAX_ROW_REL_L2}")
    assert not fails, "; ".join(fails)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_norm of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.ffn_norm.test.1): the gated metric is PCC (float output, [2048, 4096] bf16 golden), but PCC is
scale-invariant. GLM post_attention_layernorm is plain ``w * rms(x)``, w in [0.028, 0.157], eps 1e-5. Input row RMS is
0.0034..0.0236 (mean square 1.2e-5..5.6e-4, near eps). Measured on this golden, PCC / rel L2 / per-token norm ratio /
worst row rel L2: fp32 CPU 0.999986 / 0.0024 / [0.9982, 1.0019] / 0.0038; bf16 output 0.0029 / 0.0052; sum instead of
mean 0.9979 / 0.98; no weight 0.821; ``1 + w`` 0.837; attn_norm's weight 0.844; w reversed 0.809; eps 1e-6 0.9984 /
0.086; eps 1e-4 0.983; eps 1.2e-5 0.99994 / 0.0149 / [0.957, 0.9995] / 0.043; eps 8e-6 0.0162 / [1.0001, 1.050] / 0.050;
x1.01 1.0 / 0.0103 / [1.008, 1.012]; x1.02 0.0202; LayerNorm-style mean subtraction 0.99996 / 0.0082 /
[0.9982, 1.0024] / 0.029; last row zero 0.9998 / 0.020 / ratio 0. Device-like noise: squares accumulated in bf16
0.0062 / [0.983, 1.023] / 0.023; rsqrt with 5e-3 row error 0.0059 / [0.985, 1.021] / 0.021.
Extra checks: finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], worst per-token rel L2 <= 0.015 (tighter
than attn_norm's 0.03: mean subtraction scores 0.029 here). They also fail bf16 accumulation of the squares.
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
STEP = "ffn_norm"
LAYER = 0
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
    if rel > MAX_REL_L2:
        fails.append(f"relative L2 error {rel:.4f} > {MAX_REL_L2} (scale, eps or weight bug)")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if row_rel > MAX_ROW_REL_L2:
        fails.append(f"worst per-token rel L2 {row_rel:.4f} > {MAX_ROW_REL_L2}")
    assert not fails, "; ".join(fails)

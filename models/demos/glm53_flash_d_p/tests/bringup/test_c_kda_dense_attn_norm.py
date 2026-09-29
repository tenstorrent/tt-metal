# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_norm of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.attn_norm.test.1): the gated metric is PCC (float output, [2048, 4096] bf16 golden), but PCC is
scale-invariant. GLM input_layernorm is plain ``w * rms(x)``, w in [0.070, 0.157]. The input rows are small (row RMS
0.0023..0.0155, so the mean square is 5e-6..2.4e-4, the same order as eps 1e-5), which makes eps matter. Measured on
this golden, PCC / rel L2 / per-token norm ratio / worst row rel L2: fp32 CPU reference 0.999985 / 0.0023 /
[0.9998, 1.0002] / 0.0029; bf16 math 0.0033 / [0.9994, 1.0014]; sum instead of mean 0.9877 (fails PCC); no weight
0.9985 / 8.7; ``1 + w`` 0.9987 / 9.7; eps 1e-6 0.9915 / 0.21; eps 1.2e-5 0.99986 / 0.027 / [0.940, 0.996] (passes the
usual rel 0.03); x1.02 1.0 / 0.020 / [1.0198, 1.0202]; LayerNorm-style mean subtraction 0.99992 / 0.0149 /
[0.9997, 1.0002] / 0.056; w reversed 0.9972 / 0.075; last 32 rows zero 0.9924 / 0.12. Device-like noise: squares
accumulated in bf16 0.0056 / [0.9865, 1.0149] / 0.015; rsqrt with 5e-3 row error 0.0054 / [0.984, 1.016] / 0.016.
Extra checks: finite, rel L2 <= 0.01, per-token norm ratio in [0.98, 1.02], worst per-token rel L2 <= 0.03.
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.98, 1.02)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.03  # worst per-token ||got - want|| / ||want||


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

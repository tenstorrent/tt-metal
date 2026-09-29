# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_norm of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.ffn_norm.test.1): the gated metric is PCC (float output, [2048, 4096] bf16 golden), but PCC is
scale-invariant. GLM post_attention_layernorm is plain ``w * rms(x)``, eps 1e-5. At layer 3, w is nearly constant
(0.441..0.531, mean 0.495), so a missing or wrong weight is almost a uniform scale: no weight scores PCC 0.99985 and
``1 + w`` 0.99997 (both pass 0.99, unlike layer 0). Input row RMS is 0.0016..0.0103 (mean square 2.6e-6..1.1e-4, at or
below eps; chunk 0 0.0014..0.0243), so eps dominates the small rows. The output feeds the router, the experts and the
shared expert, none of which re-normalizes it, so a small uniform scale matters downstream.
Measured on this golden (chunk 1), rel L2 / per-token norm ratio / worst row rel L2 / coefficient
``<got, want> / <want, want>`` (PCC where it fails 0.99):
fp32 CPU reference 0.0023 / [0.9998, 1.0002] / 0.0025 / 1.00000; bf16 input, weight and output 0.0028 / 0.0031;
0.3% element noise 0.0042 / 0.0044 / 0.99999; no weight 1.02 / ratio 2.02; ``1 + w`` 2.02; sum instead of mean PCC
0.982; eps 0 / 1e-6 / 1e-4 PCC 0.982 / 0.989 / 0.982; attn_norm's weight 0.997 / ratio 0.043; layer 0's weight 0.990 /
0.065; w reversed 0.0257 / [0.9973, 1.0024] / 0.027; eps 1.2e-5 0.038 / [0.929, ..]; eps 1.1e-5 0.0198 / 0.9625;
eps 1.05e-5 0.0103 / [0.9807, 0.9979] / 0.0195 / 0.991; eps 9.5e-6 0.0107 / [.., 1.0205] / 0.021; eps 9e-6 0.0215;
LayerNorm-style mean subtraction 0.0127 / [0.9995, 1.0002] / 0.042 / 0.99989; norm over 4 TP shards of 1024 0.0123 /
0.039; x1.01 0.0103 / 1.0102 / coef 1.010; x1.0075 0.0079 / 1.0077 / 1.0075; x1.005 0.0055 / 1.0052 / 1.0050 (passes
rel, ratio and worst row; the coefficient catches it); one row x1.02 0.0024 / 1.0199 / 0.020; one row replaced by
its neighbour's 0.022 / 1.39; last row zero 0.0175 / ratio 0; last 32 rows zero PCC 0.9925 / 0.12; last 32 columns
zero 0.089 / 0.12. Device-like noise: squares accumulated in bf16 0.0048 / [0.9896, 1.0189] / 0.019 / 1.00096 (fails
ratio and worst row); rsqrt with 5e-3 row error 0.0059 / [0.984, 1.021] / 0.021 (fails). Chunk 0 gives the same
picture (fp32 reference 0.0023 / 0.0025, mean subtraction worst row 0.047, x1.005 coef 1.0050).
Extra checks, on chunk 1 and on chunk 0: finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], worst per-token
rel L2 <= 0.015 (the ffn_norm / attn_norm limits), coefficient in [0.996, 1.004]. Limits are written
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
STEP = "ffn_norm"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.015  # worst per-token ||got - want|| / ||want||
COEF = (0.996, 1.004)  # <got, want> / <want, want>


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    if out.numel() != want.numel():
        return [f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"]
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    coef = ((got * w).sum() / (w * w).sum()).item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_{tag}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_{tag}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_{tag}", row_rel)
    metrics.record(f"coef_{STEP}_{tag}", coef)
    print(
        f"{tag}: rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={row_rel:.4f} (<= {MAX_ROW_REL_L2}) coef={coef:.5f} (in {COEF})"
    )
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"{tag}: relative L2 error {rel:.4f} > {MAX_REL_L2} (scale, eps or weight bug)")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"{tag}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if not row_rel <= MAX_ROW_REL_L2:
        fails.append(f"{tag}: worst per-token rel L2 {row_rel:.4f} > {MAX_ROW_REL_L2}")
    if not (COEF[0] <= coef <= COEF[1]):
        fails.append(f"{tag}: coefficient {coef:.5f} outside {COEF} (uniform scale)")
    return fails


def _run(fn, ref, g, chunk):
    gl = g.layer(chunk, LAYER)
    st = _step(ref, LAYER, STEP)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    return fn(reference_ctx(ref, LAYER, g, chunk), device_ctx(LAYER, g, chunk), *inputs), gl[st.output]


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out, want = _run(fn, ref, g, c)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale checks (PCC is scale-invariant). Informational metrics, not in the runner's threshold list.
    fails = _checks(f"L{LAYER:02d}", out, want)

    # Same module on the layer's other dumped chunk (start 0, wider row RMS range).
    if c != 0:
        out0, want0 = _run(fn, ref, g, 0)
        _, ok0 = compare(f"pcc_{STEP}_L{LAYER:02d}_c0", out0, want0, mode, thr)
        if not ok0:
            fails.append(f"chunk 0: PCC below {thr}")
        fails += _checks(f"L{LAYER:02d}_c0", out0, want0)
    assert not fails, "; ".join(fails)

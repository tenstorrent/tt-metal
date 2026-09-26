# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_moe.attention.test.1): golden is s4096 chunk 1 (start 2048, chunk 2048), ``attn_norm`` [2048, 4096]
bf16 -> ``attn_out`` [2048, 4096] bf16. Same attention as layer 0 (full causal GQA, 64 q heads / 4 KV heads, QK head
dim 192, V head dim 128, V x attention_value_scale 0.707, partial rotate-half RoPE on dims [0:64] (theta 1e7), scale
192**-0.5, no sink, fused fp8 qkv, bf16 o_proj), with layer 5's weights. The KV prefix [0, 2048) comes from the golden
state. Same checks and limits as test_c_full_dense_attention.py, plus a worst per-token rel L2 and asserts that layer 5
is a full layer without a sink. CPU measurements on this golden (PCC / rel whole / rel first 128 rows / norm ratio /
worst per-token rel):
fp32 reference 0.999999 / 0.0017 / 0.0017 / [0.9992, 1.0007] / 0.0021; bf16 act + q/k/v/P + KV rounding
0.999998 / 0.0017 / 0.0017 / [0.9988, 1.0012] / 0.0023; plus bfp8 qkv/o weights 0.999996 / 0.0027 / 0.0024 /
[0.9991, 1.0012] / 0.0037; 1% noise rel 0.0101; 2% noise rel 0.0201. (Layer-0 device attention scored rel 0.0051,
ratio [0.994, 1.007].)
Bugs that pass PCC 0.99 and what catches them: RoPE positions from 0 0.99888 / 0.047 / 0.070 (rel); non-causal
0.99947 / 0.0335 / 0.048 (rel); scale 128**-0.5 0.99891 / 0.096 (rel); no value scale 0.99827 / 0.175 (rel); RoPE on
all 192 dims 0.99453 / 0.167 (rel); theta 1e4 0.99245 / 0.131 (rel); a 1024 window 0.99732 / 0.076; a 128 window
0.99101 / 0.136; no KV prefix 0.99555 / 0.098 / 0.19 (rel); x1.02 0.999999 / 0.020 (rel, ratio); a zeroed last row
0.99981 / 0.019 / ratio 0.0 (ratio, worst row); a zeroed first row 0.99977 / 0.021 / 0.086 (rel first rows, ratio).
Layer 0's weights instead of layer 5's fail PCC (-0.25).
This test checks the returned attn_out only. The K/V this chunk writes to the state is checked by the state metrics.
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
STEP = "attention"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||, whole chunk (non-causal scores 0.0335, device noise est. 0.003)
HEAD_ROWS = 128  # first rows of the chunk: most of their attention goes to the KV prefix, most exposed to causality
MAX_REL_L2_HEAD_ROWS = 0.015  # non-causal scores 0.048 here
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_WORST_ROW_REL_L2 = (
    0.06  # max over tokens of ||got_t - want_t|| / ||want_t|| (noise estimate 0.004, RoPE from 0 0.20)
)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "component chunk must start after 0 so attention reads a real KV prefix"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    assert not ref.cfg.is_sliding(LAYER), f"layer {LAYER} must be a full-attention layer"
    assert ref.w[LAYER].sink is None, f"layer {LAYER} full attention has no sink"
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

    # Error-size checks (PCC barely sees RoPE-position, causality or scale bugs). Informational metrics, not in the
    # runner's list, but asserted.
    got = out.float().reshape(want.shape)
    w = want.float()
    rows = min(HEAD_ROWS, w.shape[0])
    rel = ((got - w).norm() / w.norm()).item()
    rel_head = ((got[:rows] - w[:rows]).norm() / w[:rows].norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    worst = ((got - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)).max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"rel_l2_head_rows_{STEP}_L{LAYER:02d}", rel_head)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"worst_row_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) rel_l2_first_{rows}_rows={rel_head:.6f} (<= {MAX_REL_L2_HEAD_ROWS}) "
        f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {list(ROW_NORM_RATIO)}) "
        f"worst_row_rel_l2={worst:.4f} (<= {MAX_WORST_ROW_REL_L2})"
    )
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2}"
    assert rel_head <= MAX_REL_L2_HEAD_ROWS, (
        f"relative L2 error on the first {rows} rows {rel_head:.4f} > {MAX_REL_L2_HEAD_ROWS} "
        "(RoPE positions not offset by the chunk start, causal mask, or KV prefix mishandled?)"
    )
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token output-norm ratio [{rmin:.4f}, {rmax:.4f}] outside {list(ROW_NORM_RATIO)}"
    assert worst <= MAX_WORST_ROW_REL_L2, f"worst per-token relative L2 error {worst:.4f} > {MAX_WORST_ROW_REL_L2}"

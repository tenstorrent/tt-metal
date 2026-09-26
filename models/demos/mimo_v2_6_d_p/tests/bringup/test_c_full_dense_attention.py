# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type full_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.full_dense.attention.test.1): golden is s4096 chunk 1 (start 2048, chunk 2048), ``attn_norm`` [2048, 4096]
bf16 -> ``attn_out`` [2048, 4096] bf16; full causal GQA, 64 q heads / 4 KV heads, QK head dim 192, V head dim 128,
V x attention_value_scale 0.707, partial rotate-half RoPE on dims [0:64] (theta 1e7), scale 192**-0.5, no sink,
fused qkv (fp8 dequantized per TP rank) and bf16 o_proj. The KV prefix [0, 2048) comes from the golden state.
The gated metric stays PCC >= 0.99, but on this golden PCC passes nearly every attention bug, so the test also asserts
relative L2 over the whole chunk <= 0.015, over the first 128 rows <= 0.015, and the per-token output-norm ratio in
[0.97, 1.03]. CPU measurements (PCC / rel whole / rel first 128 rows / ratio):
reference 0.999999 / 0.0017 / 0.0017 / [0.999, 1.001]; bf16 act + weights + q/k/v/P rounding 0.0017; bfp8 qkv/o
weights 0.0021; bfp8 weights + bfp8 KV 0.0022; 1% noise 0.0101; 2% noise 0.0201. (Gemma-4 device attention measured
rel 0.005-0.008, so 0.015 leaves about 2x headroom.)
Bugs that pass PCC 0.99 and what catches them: RoPE positions from 0 0.99968 / 0.025 / 0.027 (rel); non-causal
0.99960 / 0.0285 / 0.047 (rel); scale 128**-0.5 (V head dim) 0.99931 / 0.040 / 0.032 (rel); no value scale 0.99863 /
0.138 (rel); RoPE on all 192 dims 0.99631 / 0.088 (rel); a 1024 window 0.99506 / 0.099 (rel); no KV prefix 0.99324 /
0.116 / 0.208 (rel); a zeroed last row 0.99976 / 0.022 / ratio 0.0 (ratio).
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||, whole chunk (RoPE from 0 scores 0.025)
HEAD_ROWS = 128  # first rows of the chunk: most of their attention goes to the KV prefix, most exposed to causality
MAX_REL_L2_HEAD_ROWS = 0.015  # non-causal scores 0.047 here
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "component chunk must start after 0 so attention reads a real KV prefix"
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

    # Error-size checks (PCC barely sees RoPE-position, causality or scale bugs). Informational metrics, not in the
    # runner's list, but asserted.
    got = out.float().reshape(want.shape)
    w = want.float()
    rows = min(HEAD_ROWS, w.shape[0])
    rel = ((got - w).norm() / w.norm()).item()
    rel_head = ((got[:rows] - w[:rows]).norm() / w[:rows].norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"rel_l2_head_rows_{STEP}_L{LAYER:02d}", rel_head)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) rel_l2_first_{rows}_rows={rel_head:.6f} (<= {MAX_REL_L2_HEAD_ROWS}) "
        f"row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {list(ROW_NORM_RATIO)})"
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

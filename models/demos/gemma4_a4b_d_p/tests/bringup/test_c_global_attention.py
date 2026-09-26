# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type global (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.global.attention.test.1): golden is s4096 chunk 1 (start 2048, chunk 2048), ``attn_norm`` [2048, 2816] ->
``attn_out`` [2048, 2816]; full causal attention, 16 q heads x 512, 2 KV heads x 512, no v_proj (V = unscaled RMS of the
raw k_proj output, K = RoPE(k_norm(k_proj))), proportional RoPE theta 1e6 on dims [0:64] + [256:320], scale 1.0. The KV
prefix [0, 2048) comes from the golden state.
The gated metric stays PCC >= 0.99, but on this golden PCC misses RoPE positions counted from 0 instead of the chunk
start (0.9991): RoPE is relative and the rotated dims turn slowly (theta 1e6), so only query-to-prefix scores move, most
on the first rows of the chunk. Extra checks: relative L2 over the whole chunk <= 0.03, over the first 128 rows <= 0.03,
and the per-token output-norm ratio in [0.95, 1.05]. CPU measurements (PCC / rel whole / rel first 128 rows / ratio):
reference 0.999997 / 0.0028 / 0.0027 / [0.999, 1.004]; bf16 act + weights 0.0030; bf16 act + bfp8 q/k/o weights
0.999971 / 0.0076 / ratio [0.996, 1.005]; 1% noise 0.0104; RoPE from 0 0.9991 / 0.042 / 0.062 / [0.85, 1.22].
PCC already fails: no prefix 0.849, V = roped K 0.922, V with k_norm weight 0.929, a 1024 window 0.877, a 2048 window
0.912, keys older than 3072 dropped 0.965, scale 1/sqrt(512) 0.759, RoPE on all dims 0.929, interleaved RoPE 0.972,
no q norm 0.973, non-causal 0.943.
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
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||, whole chunk
HEAD_ROWS = 128  # first rows of the chunk: the largest share of their attention goes to the KV prefix
MAX_REL_L2_HEAD_ROWS = 0.03
ROW_NORM_RATIO = (0.95, 1.05)  # per-token ||got|| / ||want||


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

    # Error-size checks (PCC barely sees a RoPE-position bug). Informational metrics, not in the runner's list.
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
        "(RoPE positions not offset by the chunk start, or KV prefix mishandled?)"
    )
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token output-norm ratio [{rmin:.4f}, {rmax:.4f}] outside {list(ROW_NORM_RATIO)}"

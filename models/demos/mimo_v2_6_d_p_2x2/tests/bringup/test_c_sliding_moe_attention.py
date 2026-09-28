# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type sliding_moe (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding_moe.attention.test.1, 1x4 prior mimo_v2_6_d_p; carried unchanged to the 2x2 run, same golden
and CPU reference, so the mutation measurements below still apply): golden is s4096 chunk 1 (start 2048, chunk 2048), ``attn_norm`` [2048, 4096]
bf16 -> ``attn_out`` [2048, 4096] bf16. Sliding-window GQA: key j visible to query i iff i - 128 < j <= i, 64 q heads /
8 KV heads, QK head dim 192, V head dim 128, V x attention_value_scale 0.707, partial rotate-half RoPE on dims [0:64]
(theta 1e4), scale 192**-0.5, a per-head sink logit (bf16, 0.56..1.14) appended to the softmax and its probability
dropped. Fused fp8 qkv (dequantized per TP rank), bf16 o_proj. The KV prefix [0, 2048) comes from the golden state;
only the last 127 prefix rows are visible, and only to the first 127 query rows.
On this layer the sink takes nearly all the softmax mass (key scores median -26, max below the sink), so attn_out row
norms are small (0.007..0.057) and most bugs are loud. CPU measurements (PCC / rel whole / rel first 128 rows / norm
ratio / worst per-token rel):
fp32 reference vs golden 0.999989 / 0.0047 / 0.0042 / [0.983, 1.015] / 0.018; device noise estimate (bf16 q/k/v,
scores and P, bfp8 qkv/o weights, bfp8 KV) 0.999974 / 0.0072 / 0.0081 / [0.975, 1.022] / 0.025.
Bugs that pass PCC 0.99, and what catches them: sink zero 0.99995 / rel 1.08; sink negated 0.9998 / 3.3; no value
scale 0.9997 / 0.41; window 64 0.9988 / 0.25; no KV prefix 0.9964 / 0.085 / first rows 0.40; scale 128**-0.5
0.9926 / 0.90; x1.02 0.99999 / 0.020 (rel whole); a zeroed last row 0.99995 / 0.0096 (norm ratio, worst row);
a zeroed first row 0.99986 / 0.017 (ratio). Loud (fail PCC): RoPE from 0, theta 1e7, RoPE on all 192 dims, interleaved
RoPE, no window, window 256, non-causal within the window, no sink, wrong GQA head map.
Window off by one (127 or 129) scores PCC 0.99998 / rel 0.010 / worst row 0.047, inside device noise for the size
checks, so the test also asserts the output is closer to the CPU reference at window 128 than to the same reference
run at windows 127 and 129 (the two differ from it by rel 0.010, a direction random device noise does not follow).
Limits: rel L2 whole and first 128 rows <= 0.02 (smallest structural bug 0.085; the device noise floor is higher than
layer 0's because the golden itself is 0.0047 from the fp32 reference), per-token norm ratio in [0.95, 1.05] (noise
already reaches 0.975), worst per-token rel L2 <= 0.08.
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
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.02  # ||got - want|| / ||want||, whole chunk (x1.02 scores 0.020, no KV prefix 0.085)
HEAD_ROWS = 128  # first rows of the chunk: the only rows that attend into the KV prefix
MAX_REL_L2_HEAD_ROWS = 0.02  # no KV prefix scores 0.40 here
ROW_NORM_RATIO = (0.95, 1.05)  # per-token ||got|| / ||want|| (device noise estimate [0.975, 1.022])
MAX_WORST_ROW_REL_L2 = 0.08  # max over tokens of ||got_t - want_t|| / ||want_t|| (noise estimate 0.025)
WINDOW_ALTERNATIVES = (-1, 1)  # the output must be closer to the window-W reference than to W-1 and W+1


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _reference_at_window(ref, g, c, x: torch.Tensor, window: int) -> torch.Tensor:
    """The CPU reference attention on the same input with a different sliding window (restored afterwards)."""
    old = ref.cfg.sliding_window
    ref.cfg.sliding_window = window
    try:
        return ref.component(LAYER, STEP)(reference_ctx(ref, LAYER, g, c), x).float()
    finally:
        ref.cfg.sliding_window = old


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "component chunk must start after 0 so attention reads a real KV prefix"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    assert ref.cfg.is_sliding(LAYER), f"layer {LAYER} must be a sliding-window layer"
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

    # Error-size checks (PCC misses sink, value-scale, prefix and row bugs here). Informational metrics, not in the
    # runner's list, but asserted.
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    rows = min(HEAD_ROWS, w.shape[0])
    rel = _rel(got, w)
    rel_head = _rel(got[:rows], w[:rows])
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
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2}"
    assert rel_head <= MAX_REL_L2_HEAD_ROWS, (
        f"relative L2 error on the first {rows} rows {rel_head:.4f} > {MAX_REL_L2_HEAD_ROWS} "
        "(KV prefix ignored, RoPE positions not offset by the chunk start, or window mask wrong?)"
    )
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token output-norm ratio [{rmin:.4f}, {rmax:.4f}] outside {list(ROW_NORM_RATIO)}"
    assert worst <= MAX_WORST_ROW_REL_L2, f"worst per-token relative L2 error {worst:.4f} > {MAX_WORST_ROW_REL_L2}"

    # Window discrimination: an off-by-one window (e.g. `>=` instead of `>` in the mask) stays inside the size limits.
    window = int(ref.cfg.sliding_window)
    x = inputs[0]
    d_w = _rel(got, _reference_at_window(ref, g, c, x, window))
    metrics.record(f"rel_l2_vs_cpu_window_{STEP}_L{LAYER:02d}", d_w)
    for dw in WINDOW_ALTERNATIVES:
        alt = window + dw
        d_alt = _rel(got, _reference_at_window(ref, g, c, x, alt))
        print(f"rel_l2 vs CPU window {window}: {d_w:.6f}; vs CPU window {alt}: {d_alt:.6f}")
        assert d_w < d_alt, (
            f"output is closer to a window-{alt} attention (rel {d_alt:.5f}) than to window {window} (rel {d_w:.5f}): "
            f"key j must be visible to query i iff i - {window} < j <= i"
        )

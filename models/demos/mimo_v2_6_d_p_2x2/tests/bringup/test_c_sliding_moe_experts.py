# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type sliding_moe (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding_moe.experts.test.1, mimo_v2_6_d_p_2x2): adopted unchanged from the prior mimo_v2_6_d_p (1x4)
frozen test; the golden is shared and every check is on the gathered [S, 4096] output, so none depends on the mesh
shape (on 2x2 the per-chip pair counts below still hold for 64 contiguous experts per chip, whatever the chip order).
Prior review: experts_out = sum_e router[t, e] * down_e(silu(gate_e(x)) * up_e(x)), with
x = ffn_norm [2048, 4096] and router the dense [2048, 256] routing matrix (8 nonzeros per row, bf16, rows sum to 1),
256 experts of width 2048 (mxfp4 weights, exact in fp32). No shared expert. Golden experts_out bf16, row norms
0.071..1.11 (median 0.25). Routing on this golden: tokens per expert 0..804 (expert 64 takes 804 of 2048), pairs per
chip of 64 experts 3381 / 5292 / 3782 / 3929, so a dispatch capacity below 8 x S rows per chip drops real work.
The gated metric is PCC (float output). PCC misses dropped experts, dropped (token, expert) pairs, zeroed rows and
scale errors, so the test also asserts a finite output, rel L2 <= 0.03, per-token norm ratio within [0.97, 1.03] and
per-token rel L2 <= 0.1 on every row (the Gemma-4 limits; MiMo's device-noise estimate is below a third of each).
Measured on this golden as PCC / rel L2 / norm ratio / worst per-row rel L2:
  CPU reference 0.999997 / 0.0023 / [0.995, 1.004] / 0.0050; bf16 out 0.0027 / 0.0053;
  bfp8 weights in device layout (blocks along the output dim) 0.999993 / 0.0037 / [0.993, 1.005] / 0.014;
  that plus bfp8 activations (x and the SwiGLU product) 0.99996 / 0.0091 / [0.986, 1.014] / 0.028 (device-noise estimate);
  that with a 5-bit-truncated operand (a crude LoFi model) 0.9988 / 0.048 (fails) / [0.925, 1.064] / 0.18: the fused
  kernel's hard-coded LoFi may not pass, see known issues;
  drop expert 0 0.9983 (passes) / 0.059 / min 0.47 / 0.77; drop the hottest expert (64) 0.9960 (passes) / 0.090;
  drop expert 255 (3 tokens) 0.999996 / 0.0028 / 0.992 / 0.068 (passes every check: known gap);
  drop one chip's 64 experts 0.85..0.95; drop the smallest-weight pair of every token 0.9956 (passes) / 0.094;
  drop token 0's top-1 pair 0.99997 (passes) / 0.0077 (passes) / min 0.78 / 0.33;
  drop one token's smallest pair 0.044..0.090 worst row (passes: known gap, a sub-1% contribution to one row);
  last row zeroed 0.99977 (passes) / 0.022 (passes) / min 0 / 1.0; last 32 rows zeroed 0.9937 (passes) / 0.112;
  x1.02 0.999997 / 0.020 / [1.015, 1.024] (passes: known gap); x1.05 / 0.050 / [1.045, 1.054]; 2x 0.999997 (passes) / 1.0;
  capacity 512 tokens per expert 0.9982 (passes) / 0.060; 256 0.9935 (passes) / 0.114; 128 0.950;
  routing weight 1 0.976 / 5.5; uniform 1/8 0.976 / 0.27; gelu_tanh instead of silu 0.749; gate and up swapped 0.44;
  expert ids shifted by one 0.11. Renormalizing the routing weights again changes nothing (they already sum to 1).
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
STEP = "experts"
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.1  # per-token ||got - want|| / ||want||, worst row (dropped (token, expert) pairs)


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

    # Dropped-expert, dropped-pair, scale and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, golden {want.numel()}"
    got = out.float().reshape(want.shape)
    w = want.float()
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = (got - w).norm(dim=-1) / wn
    worst = row_rel.max().item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={worst:.5f} (<= {MAX_ROW_REL_L2}, worst row {row_rel.argmax().item()})"
    )
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (dropped expert, activation or scale bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed rows or dropped pairs?)"
    assert (
        worst <= MAX_ROW_REL_L2
    ), f"worst per-token rel L2 {worst:.4f} > {MAX_ROW_REL_L2} at row {row_rel.argmax().item()} (dropped (token, expert) pair?)"

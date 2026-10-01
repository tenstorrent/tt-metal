# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type full_moe (layer 5) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 5, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (mimo_v2_6_d_p_cp4 C.full_moe.experts.test.1): copied from the prior bring-up mimo_v2_6_d_p's frozen test (same
golden, rung s4096 chunk 2048, same CPU reference) with the cp4 additions of test_c_sliding_moe_experts.py: the deferred /
CPU-bridge guards and a rel L2 per CP slice (rows [r S/4, (r+1) S/4), <= MAX_SLICE_REL_L2). Under CP=4 (EP=4) the device
output must be the full [S, 4096] (each chip's combined slice, gathered in order).
Measured (cp4) as rel L2 / per-slice rel L2: CPU reference 0.0050 / 0.0047..0.0057; golden slices 1 and 2 swapped 0.97 /
1.48, 1.30; slice 3 zeroed 0.55 / 1.0; slice 3 x1.05 0.0275 (passes whole-chunk rel) / 0.050 (caught by slice and ratio).
Prior review (C.full_moe.experts.test.1): same step as layer 1 (test_c_sliding_moe_experts.py): experts_out =
sum_e router[t, e] * down_e(silu(gate_e(x)) * up_e(x)), x = ffn_norm [2048, 4096], router the dense [2048, 256]
routing matrix (8 nonzeros per row, rows sum to 1), 256 mxfp4 experts of width 2048, no shared expert.
Layer 5 differs from layer 1:
  - golden experts_out row norms 1.3e-4..0.98 (median 0.008), so per-row checks see very small rows;
  - x has large outlier channels in every row (|x| up to 131 on channel 3891, median |x| 0.009, row norm ~178). A bfp8
    x (shared exponent per 16) flushes the small channels next to an outlier: bfp8 x alone gives rel 0.045 / ratio
    [0.905, 1.128] / worst row 0.135, which FAILS. Keep the expert input in bf16 (the fused unified kernel packs
    activations to bfp8 internally, see known issues);
  - routing weights go down to 1e-10; expert 235 outputs norm ~690 at weight ~5e-9 (contribution 3e-6), so a routing
    weight bug blows up (uniform 1/8: ratio up to 843);
  - tokens per expert 0..377 (hottest 163; 11 experts get no token), pairs per chip 3966 / 3813 / 4848 / 3757.
Limits kept from layer 1 (the Gemma-4 limits): PCC >= 0.99 (gated), finite, rel L2 <= 0.03, per-token norm ratio in
[0.97, 1.03], worst per-token rel L2 <= 0.1.
Measured on this golden (CPU) as PCC / rel L2 / norm ratio / worst per-row rel L2:
  CPU reference 0.999976 / 0.0050 / [0.987, 1.016] / 0.016; bf16 out 0.0053 / 0.0165;
  bfp8 weights in device layout (blocks along the output dim), x bf16: 0.999975 / 0.0051 / [0.987, 1.016] / 0.016;
  bfp8 x (h bf16 or bfp8): 0.99897 / 0.045 (fails) / [0.905, 1.128] / 0.13;
  drop hottest expert (163, 377 tokens) 0.9936 / 0.114; drop the coldest used expert (58, 1 token) 0.99997 / 0.0055 /
  min 0.83 / 0.43 (caught by ratio and worst row); drop expert 255 min 0.84 / 0.35; drop expert 0 (no tokens) unchanged;
  drop one chip's 64 experts 0.76..0.98; drop the smallest-weight pair of every token 0.9933 (passes) / 0.117;
  drop one token's smallest pair: worst row up to 1.03 over sampled tokens (the smallest pair can dominate a tiny row);
  drop token 0's top-1 pair 0.999976 / 0.0050 / worst 0.025 (passes every check: known gap, that pair's output is small);
  last row zeroed 0.999976 (passes) / 0.0050 (passes) / min 0 / 1.0; last 32 rows zeroed 0.9943 (passes) / 0.107;
  x1.02 0.999975 / 0.021 / [1.006, 1.037] (caught by ratio only); x1.05 / 0.050; 2x 0.999976 (passes) / 1.0;
  capacity 512 tokens per expert unchanged (max 377); 256 0.9967 (passes) / 0.082; 128 0.945;
  routing weight 1 or uniform 1/8 0.186; gelu_tanh instead of silu 0.742.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    impl_mode,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "experts"
LAYER = 5
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.97, 1.03)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.1  # per-token ||got - want|| / ||want||, worst row (dropped (token, expert) pairs)
CP = 4  # context-parallel slices of the chunk (mesh 1x4)
MAX_SLICE_REL_L2 = 0.03  # rel L2 over each CP slice's rows


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    from models.demos.common.bringup.plan.op_request import deferred_steps

    assert not (
        impl_mode() == "device" and (S.block_type_of(LAYER), STEP) in deferred_steps(S)
    ), f"{STEP}: deferred to op-gen; it runs on the CPU until the op is delivered"
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), f"{STEP}: device_component returned a CPU bridge, not on the device"
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
    n = w.shape[-2] // CP
    srel = [
        (
            (got[..., r * n : (r + 1) * n, :] - w[..., r * n : (r + 1) * n, :]).norm()
            / w[..., r * n : (r + 1) * n, :].norm()
        ).item()
        for r in range(CP)
    ]
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"max_slice_rel_l2_{STEP}_L{LAYER:02d}", max(srel))
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_L{LAYER:02d}", worst)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={worst:.5f} (<= {MAX_ROW_REL_L2}, worst row {row_rel.argmax().item()}) "
        f"slice_rel_l2={[round(x, 5) for x in srel]} (<= {MAX_SLICE_REL_L2})"
    )
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2} (dropped expert, activation or scale bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO} (zeroed rows or dropped pairs?)"
    assert (
        worst <= MAX_ROW_REL_L2
    ), f"worst per-token rel L2 {worst:.4f} > {MAX_ROW_REL_L2} at row {row_rel.argmax().item()} (dropped (token, expert) pair?)"
    assert (
        max(srel) <= MAX_SLICE_REL_L2
    ), f"rel L2 per CP slice {[round(x, 4) for x in srel]}, max > {MAX_SLICE_REL_L2} (a slice's dispatch / combine lost work?)"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.experts.test.1). HF HYV4Experts: experts_out = sum_e router[t, e] * down_e(silu(min(g, 10)) *
clamp(u, -10, 10)), (g, u) = gate_up_proj[e] @ x split in halves (gate rows 0-2047, up 2048-4095), on x = ffn_norm
[2048, 6144] bf16 and router = the dense [2048, 256] routing matrix (golden bf16, 8 nonzeros per row, weights
0.088..1.65, rows sum to 2.827 = routed_scaling_factor). 256 experts of width 2048, bf16 weights as stored.
This golden (s4096 chunk 1): x row norms 8.8..11.4, max |x| 0.85 (no outlier channels, unlike MiMo layer 5);
experts_out row norms 16.8..447 (median 109); tokens per expert 2..505 (hottest 187), every expert used; pairs per chip
of the plan's EP layout (experts 128c + 64r ..+63) 3986 / 4298 / 4429 / 3671. The clamp fires on a few entries (gate
max 10.49, |up| up to 11.2). Measured on this golden (CPU study /tmp/hy4_exp1, study.py / study2.py) as PCC / rel L2 /
per-token norm ratio / worst per-token rel L2 / global coefficient <got, want> / <want, want>:
  CPU reference fp32 0.999997 / 0.0023 / [0.9962, 1.0037] / 0.0043 / 0.99992; bf16 out 0.0027 / 0.0048;
  bf16 h 0.0031 / [0.995, 1.004] / 0.0056; device estimate bfp8 weights (blocks along the output dim) + bf16 h / out
  0.0072 / [0.995, 1.005] / 0.0093 / 0.99988; with bfp8 x and bfp8 h as well 0.0104 / [0.993, 1.008] / 0.0153;
  HiFi2-like (weights cut to 5 mantissa bits) 0.0259 / [0.969, 0.984] / 0.031;
  no clamp 0.9999 / 0.0095 / max 1.0696 / 0.078; clamp gate only 0.0084 / 1.0696 / 0.078; clamp up only 0.0050 /
  1.0305 / 0.035; limit 7 0.988 / 0.19; gelu_tanh 0.99999 (passes PCC) / 0.062 / [0.79, 1.11]; SwiGluOai 0.30;
  gate / up swapped 0.65; gate / up interleaved split 1.01; no silu 0.35;
  x 1.02 0.99999 / 0.020 / coef 1.0200; x 1.01 0.010 / [1.006, 1.014] / 0.014 / coef 1.0099; x 1.005 coef 1.0049;
  weights renormalized to 1 (route scale dropped) 0.99999 / 0.646 / 0.354; routing weight 1 or uniform: PCC 0.9475;
  drop the hottest expert (187) 0.982 / 0.199; drop expert 235 (2 tokens) 0.99999 / 0.0035 / min 0.974 / 0.185;
  drop expert 0 (46 tokens) / 0.0071 / 0.938 / 0.218; dropping any single expert gives worst row >= 0.046 (expert 98);
  drop one chip's 64 experts PCC 0.85..0.93; experts 128..255 missing (no axis-1 reduce) 0.78;
  drop the smallest pair of every token 0.99999 / 0.034 / min 0.57 / 0.70; token 0's top-1 pair 0.0094 / 0.65 / 0.55;
  last row zeroed 0.99999 (passes PCC) / 0.031 / 0 / 1.0; last 32 rows zeroed 0.9935 (passes PCC) / 0.127;
  rows 1023 / 1024 swapped 0.9993 / 0.041 / [0.90, 1.11] / 1.44; capacity 256 per expert 0.992 (passes PCC) / 0.139;
  SP row halves or output column halves swapped, second SP half zeroed: PCC < 0.7.
The gated metric is PCC. It misses scale errors, gelu, dropped experts / pairs / rows and a capacity limit, so the test
also asserts, vs the golden: finite, element count, rel L2 <= 0.015, every token's norm ratio in [0.98, 1.02], worst
token rel L2 <= 0.03, global coefficient within 0.004 of 1. These fail every mutation above except x 1.003 (coef
1.0029), and three numerically harmless ones (min(silu(g), 10), g clamped on both sides: bit-close to the reference).
Known gap: one token's smallest pair: half of the tokens' smallest pair moves its row by < 0.03 (median 0.029); that
needs an exact pair list, which the dense boundary does not give.
The golden barely exercises the clamp (6e-7 of gate entries > 10), so the module runs again on x * 2 (exact in bf16;
gate up to 21) with the golden routing, vs the CPU experts on the same input. There: device estimate 0.0068 / [0.998,
1.002] / 0.0090 / 0.99988 (bfp8 x and h too: 0.0105 / [0.9965, 1.0043] / 0.0151); no clamp 0.47; clamp gate only 0.23
/ max 1.69; clamp up only 0.158; up clamped at max only 0.167; limit 9 0.078 / min 0.845; gelu_tanh 0.044 / 0.32;
SwiGluOai 0.19; golden-scale output (input ignored) 0.72. Limits there: the same as vs the golden.
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
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.98, 1.02)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.03  # worst per-token ||got - want|| / ||want|| (dropped experts / pairs, zeroed or swapped rows)
MAX_COEF_ERR = 0.004  # |<got, want> / <want, want> - 1| (uniform scale errors), float64
SYN_SCALE = 2.0  # probe input = golden ffn_norm * 2 (exact in bf16): gate reaches 21, so the clamp at 10 matters


def _errors(got: torch.Tensor, want: torch.Tensor):
    """(rel L2, row norm ratio min, max, worst row rel L2, worst row, global coefficient)."""
    got, want = got.float().reshape(want.shape), want.float()
    wn = want.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - want).norm() / want.norm()).item()
    ratio = got.norm(dim=-1) / wn
    row = (got - want).norm(dim=-1) / wn
    g64, w64 = got.double(), want.double()
    coef = ((g64 * w64).sum() / (w64 * w64).sum()).item()
    return rel, ratio.min().item(), ratio.max().item(), row.max().item(), row.argmax().item(), coef


def _check(tag: str, got: torch.Tensor, want: torch.Tensor):
    assert got.numel() == want.numel(), f"{tag}: output has {got.numel()} elements, golden {tuple(want.shape)}"
    assert torch.isfinite(got.float()).all(), f"{tag}: non-finite output"
    rel, rmin, rmax, row, argrow, coef = _errors(got, want)
    p = "" if tag == "golden" else "syn_"
    metrics.record(f"{p}rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"{p}row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"{p}row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"{p}max_row_rel_l2_{STEP}_L{LAYER:02d}", row)
    metrics.record(f"{p}coef_{STEP}_L{LAYER:02d}", coef)
    print(
        f"{tag}: rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(ROW_NORM_RATIO)}) "
        f"max_row_rel_l2={row:.5f} (<= {MAX_ROW_REL_L2}, row {argrow}) coef={coef:.5f} (1 +- {MAX_COEF_ERR})"
    )
    assert rel <= MAX_REL_L2, f"{tag}: relative L2 error {rel:.5f} > {MAX_REL_L2} (fidelity, activation or scale bug)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"{tag}: per-token norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ROW_NORM_RATIO)} (clamp, dropped expert or zeroed rows?)"
    assert (
        row <= MAX_ROW_REL_L2
    ), f"{tag}: worst per-token rel L2 {row:.5f} > {MAX_ROW_REL_L2} at row {argrow} (dropped expert / pair, capacity?)"
    assert abs(coef - 1) <= MAX_COEF_ERR, f"{tag}: global scale {coef:.5f} not within {MAX_COEF_ERR} of 1"


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(
        fn, "cpu_bridge", False
    ), "device_component returned a CPU bridge; the experts are not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale, activation, dropped-expert and row checks (PCC misses them). Informational metrics, not in the runner's
    # threshold list.
    _check("golden", out, want)

    # Large-argument probe: the clamp at 10 barely fires on the golden. Same routing, x * 2, vs the CPU step.
    x, routing = inputs
    xs = (x * SYN_SCALE).bfloat16().float()
    syn_want = ref.component(LAYER, STEP)(rctx, xs, routing).float()
    syn_out = fn(rctx, dctx, xs, routing)
    _check(f"x*{SYN_SCALE:g} vs CPU", syn_out, syn_want)

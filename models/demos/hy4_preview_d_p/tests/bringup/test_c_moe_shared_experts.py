# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.experts.test.1), ported from the layer-1 test (test_c_moe_full_experts.py) and re-measured
on layer 2. HF HYV4Experts: experts_out = sum_e router[t, e] * down_e(silu(min(g, 10)) * clamp(u, -10, 10)),
(g, u) = gate_up_proj[e] @ x split in halves (gate rows 0-2047, up 2048-4095), on x = ffn_norm [2048, 6144] bf16 and
router = the dense [2048, 256] routing matrix (golden bf16, 8 nonzeros per row, weights 0.178..1.38, rows sum to
2.827 = routed_scaling_factor). 256 experts of width 2048, bf16 weights as stored. The step is the same as at layer 1;
a moe_shared layer differs only in its attention (shared top-k).
This golden (s4096 chunk 1): x row norms 5.2..8.4, max |x| 0.77 (no outlier channels); experts_out row norms
5.5..150 (median 21); tokens per expert 1..390 (hottest 118), every expert used, 4 experts have 1 token; pairs per
chip of the plan's EP layout (experts 128c + 64r ..+63) 4514 / 4079 / 3894 / 3897. The clamp never fires on this
golden (gate max 9.74, |up| max 9.91). Measured on this golden (CPU study /tmp/hy4_exp2, study.py / study2.py /
study3.py) as PCC / rel L2 / per-token norm ratio / worst per-token rel L2 / global coefficient <got, want> /
<want, want>:
  CPU reference fp32 0.999997 / 0.0023 / [0.9964, 1.0035] / 0.0041 / 0.99994; bf16 out 0.0028 / 0.0045;
  bf16 h 0.0031 / [0.995, 1.005] / 0.0057; device estimate bfp8 weights (blocks along the output dim) + bf16 h / out
  0.0075 / [0.994, 1.005] / 0.0105; with bfp8 x and bfp8 h as well 0.0110 / [0.992, 1.008] / 0.0163;
  HiFi2-like (weights cut to 5 mantissa bits) 0.0269 / [0.970, 0.981] / 0.031;
  no clamp, clamp gate only, clamp up only: identical to the reference (the clamp never fires); limit 7 0.99974 /
  0.071 / min 0.72; gelu_tanh 0.99962 (passes PCC) / 0.087 / [0.72, 1.12]; SwiGluOai 0.87; gate / up swapped 0.81;
  gate / up interleaved split 1.03; no silu 0.43;
  x 1.01 0.99999 / 0.010 / [1.006, 1.014] / 0.014 / coef 1.0099; x 1.005 coef 1.0049; x 1.003 coef 1.0029;
  weights renormalized to 1 (route scale dropped) 0.99999 / 0.646 / 0.354; routing weight 1 or uniform: PCC 0.938;
  drop the hottest expert (118) 0.990 (passes PCC) / 0.154; drop expert 0 (384 tokens) 0.981; drop expert 147
  (1 token) 0.99999 / 0.0029 / min 0.974 / 0.20; drop expert 155 (24 tokens) 0.0042 / 0.987 / 0.106;
  drop expert 180 (1 token) 0.999996 / 0.0027 / [0.9964, 1.0035] / 0.0197 / 0.99993: the only single-expert drop
  whose worst row is < 0.10 (next: 155 at 0.106), and it passes the layer-1 worst-row limit 0.03;
  drop one chip's 64 experts PCC 0.85..0.90; experts 128..255 missing (no axis-1 reduce) 0.75;
  drop the smallest pair of every token 0.9995 / 0.055 / min 0.49 / 0.79; token 0's top-1 pair 0.0072 / 0.71 / 0.58;
  token 0's smallest pair 0.0027 / 0.988 / 0.124;
  last row zeroed 0.99998 (passes PCC) / 0.0060 / 0 / 1.0; last 32 rows zeroed 0.9946 (passes PCC) / 0.121;
  rows 1023 / 1024 swapped 0.99974 (passes PCC) / 0.023 / [0.23, 4.4] / 4.36; capacity 256 per expert 0.9897 / 0.157;
  SP row halves or output column halves swapped, second SP half zeroed: PCC < 0.72.
The gated metric is PCC. It misses scale errors, gelu, dropped experts / pairs / rows and a capacity limit, so the test
also asserts, vs the golden: finite, element count, rel L2 <= 0.015, every token's norm ratio in [0.98, 1.02], worst
token rel L2 <= 0.018, global coefficient within 0.004 of 1. The worst-row limit is 0.018 here (layer 1: 0.03) so that
dropping expert 180 (0.0197 on the golden, 0.0299 on the probe) fails; the device module scores 0.0111 / 0.0114 and
the all-bfp8 estimate 0.0163 / 0.0167. These fail every mutation above except x 1.003 (coef 1.0029), the three clamp
bugs on the golden (the probe catches them), and two numerically harmless ones (min(silu(g), 10), g clamped on both
sides: bit-close to the reference).
Known gap: one token's smallest pair: 7% of the tokens' smallest pair moves its row by < 0.02 (median 0.065); that
needs an exact pair list, which the dense boundary does not give.
The golden never exercises the clamp, so the module runs again on x * 2 (exact in bf16; gate up to 19.5, 3.8e-6 of
gate entries > 10) with the golden routing, vs the CPU experts on the same input. There: device estimate 0.0072 /
[0.997, 1.003] / 0.0109 / 0.99993 (bfp8 x and h too: 0.0109 / [0.993, 1.006] / 0.0167); no clamp 0.205; clamp gate
only 0.141 / max 1.76; clamp up only 0.097; up clamped at max only 0.097; limit 9 0.054 / min 0.825; gelu_tanh 0.065
/ 0.61; SwiGluOai 0.31; golden-scale output (input ignored) 0.75; drop expert 180 0.0023 / 0.9970 / 0.0299. Limits
there: the same as vs the golden.
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
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.98, 1.02)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.018  # worst per-token ||got - want|| / ||want|| (dropped experts / pairs, zeroed or swapped
# rows); 0.018, not layer 1's 0.03: dropping 1-token expert 180 scores 0.0197 (probe 0.0299), the device 0.0111
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

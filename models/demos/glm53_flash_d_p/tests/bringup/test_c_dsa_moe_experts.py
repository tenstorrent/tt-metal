# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.experts.test.1): experts_out = sum_e router[t, e] * down_e(silu(min(gate_e(x), 10)) *
clamp(up_e(x), -10, 10)), x = ffn_norm [2048, 4096] (|x| <= 2.36, no outlier channels), router the golden dense
[2048, 288] bf16 routing matrix (8 nonzeros per row, rows sum to 2.5, weights 0.029..2.14), 288 fp8 experts (128x128
block scales) of width 2048. Golden bf16, row norms 2.7..177 (median 48). Tokens per expert 4..214 (hottest 198,
coldest 206; every expert used); pairs per chip (72 experts each) 3856 / 4003 / 4179 / 4346.
The clamps engage on this golden only 39 times in 33.5M entries (4 gate > 10, 35 |up| > 10).
Measured on chunk 1 (host script, fp32 weights unless noted) as PCC / rel L2 / per-token norm ratio / worst per-token
rel L2 / scale coefficient <got, want> / <want, want>:
  CPU reference on the bf16 golden inputs 0.999997 / 0.0023 / [0.9965, 1.0038] / 0.0042 / 1.00003 (the golden came
  from the fp32 router); all bf16 (W, x, h, y) 0.0043 / [0.9939, 1.0056] / 0.0076 / 0.99955;
  device-like (bfp8 weights, blocks along the output dim, bf16 x / h / y / out) 0.999976 / 0.0070 / [0.9942, 1.0066] /
  0.0112 / 1.00006 (chunk 0: 0.0070 / [0.9933, 1.0080] / 0.0116); 128-row-block coefficients [0.99966, 1.00071];
  bfp8 x 0.0075 / 0.0145; bfp8 x and h (the fused kernel's default path) 0.0135 / [0.9831, 1.0150] / 0.0238 (fails rel
  and ratio: use high_precision); weights truncated to bf16 0.0071 (passes rel) / [0.9899, 0.9971] / coef 0.99335
  (fails coef); 6-bit mantissa truncation (HiFi2-like) 0.0134 / coef 0.987;
  x1.005 0.0055 / coef 1.00503 (fails coef); x1.01 0.0103 / [1.0064, 1.0139] / 1.01003; x1.02 ratio max 1.0239;
  second sequence half (mesh row 1) x1.005 0.0042 / coef 1.0025 (passes the global coef) / 128-row block max 1.0052;
  rows 0..127 x1.01 block coef 1.0104; no routed scale 2.5 PCC 0.999997 (passes) / rel 0.60;
  drop hottest expert 0.9987 / 0.0515 / worst row 0.998; drop the coldest expert (4 tokens) 0.99974 (passes) / 0.023
  (passes rel) / min 0.49 / 0.70; drop expert 0 or 287 0.9983..0.9986 / 0.052..0.058; drop one chip's 72 experts
  0.85..0.91; drop every token's smallest pair 0.99987 / 0.0164 / min 0.943 / 0.29; drop token 0's top-1 pair 0.99948 /
  0.032 / worst row 0.998; drop one token's smallest pair: worst row 0.0085..0.024 on 6 sampled tokens (known gap, not
  separable from the device noise 0.011); last row zeroed 0.999977 (passes) / 0.0068 (passes) / min 0 / 1.0; last 32
  rows 0.9939 / 0.11; second half zeroed 0.71; capacity 256 tokens per expert unchanged (max 214), 128 0.9986 / 0.053;
  uniform routing 0.89; gelu_tanh 0.9983 / 0.075; gate and up swapped 0.81; chip 0's local expert index off by one 0.85;
  no gate clamp 0.999996 / 0.0027 / max 1.033 / 0.036; no up clamp 0.999993 / 0.0038 / max 1.047 / 0.060; up clamped
  above only 0.051, below only 0.060; limit 9.5 min 0.965 / 0.039; limit 7 0.996 / 0.099; gate clamped to +-10 and
  clamping silu(g) * u equal the reference here.
Clamp probe: the golden barely reaches the limit, so the module also runs on 8 * x (gate > 10 on 1.1% of entries, |up|
> 10 on 3.2%) against the CPU reference of the same input: device-like 0.0083 / [0.9988, 1.0014] / 0.0117 / 0.99992;
no gate clamp 0.58 / max 2.16; no up clamp 0.60 / max 2.18.
The gated metric is PCC (pcc_experts_L03, chunk 1). It misses scale errors, a missing routed scale, dropped rows and
small experts, and the clamps. So the test also asserts, on chunk 1 and on chunk 0 (start 0): shape, finite, rel L2
<= 0.012, per-token norm ratio in [0.985, 1.015], worst per-token rel L2 <= 0.025, scale coefficient in [0.997, 1.003]
and every 128-row block's coefficient in [0.995, 1.005] (the sequence is split over the mesh rows). On the clamp probe
(chunk 1): rel L2 <= 0.015, ratio [0.985, 1.015], worst row <= 0.025, coefficient [0.997, 1.003]. Limits are written
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
STEP = "experts"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.012  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.985, 1.015)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.025  # worst per-token ||got - want|| / ||want|| (dropped pairs, small experts, clamps)
SCALE_COEF = (0.997, 1.003)  # <got, want> / <want, want>
BLOCK_ROWS = 128
BLOCK_COEF = (0.995, 1.005)  # the same coefficient on every 128-row block
CLAMP_PROBE_SCALE = 8.0  # input multiplier that makes the swiglu clamps (limit 10) engage
PROBE_MAX_REL_L2 = 0.015
PROBE_ROW_NORM_RATIO = (0.985, 1.015)
PROBE_MAX_ROW_REL_L2 = 0.025
PROBE_SCALE_COEF = (0.997, 1.003)


def _stats(got: torch.Tensor, want: torch.Tensor) -> dict:
    w = want.float()
    g = got.float().reshape(w.shape)
    wn = w.norm(dim=-1).clamp_min(1e-12)
    ratio = g.norm(dim=-1) / wn
    row_rel = (g - w).norm(dim=-1) / wn
    nb = w.shape[0] // BLOCK_ROWS
    gb, wb = g[: nb * BLOCK_ROWS].reshape(nb, -1), w[: nb * BLOCK_ROWS].reshape(nb, -1)
    bcoef = (gb * wb).sum(-1) / (wb * wb).sum(-1)
    return {
        "finite": bool(torch.isfinite(g).all()),
        "rel": ((g - w).norm() / w.norm()).item(),
        "rmin": ratio.min().item(),
        "rmax": ratio.max().item(),
        "row_rel": row_rel.max().item(),
        "worst_row": row_rel.argmax().item(),
        "coef": ((g * w).sum() / (w * w).sum()).item(),
        "bmin": bcoef.min().item(),
        "bmax": bcoef.max().item(),
    }


def _check(tag: str, got, want, max_rel, ratio_lim, max_row, coef_lim, block_lim) -> list[str]:
    if got.numel() != want.numel():
        return [f"{tag}: output has {got.numel()} elements, golden {tuple(want.shape)}"]
    st = _stats(got, want)
    for k in ("rel", "rmin", "rmax", "row_rel", "coef", "bmin", "bmax"):
        metrics.record(f"{k}_{tag}_{STEP}_L{LAYER:02d}", st[k])
    print(
        f"{tag}: rel_l2={st['rel']:.6f} (<= {max_rel}) row_norm_ratio=[{st['rmin']:.4f}, {st['rmax']:.4f}] "
        f"(in {ratio_lim}) max_row_rel_l2={st['row_rel']:.4f} (<= {max_row}, row {st['worst_row']}) "
        f"coef={st['coef']:.5f} (in {coef_lim}) block_coef=[{st['bmin']:.5f}, {st['bmax']:.5f}]"
        + (f" (in {block_lim})" if block_lim else "")
    )
    fails = []
    if not st["finite"]:
        fails.append(f"{tag}: non-finite output")
    if not st["rel"] <= max_rel:
        fails.append(f"{tag}: relative L2 error {st['rel']:.4f} > {max_rel} (precision, scale or dropped expert)")
    if not (ratio_lim[0] <= st["rmin"] and st["rmax"] <= ratio_lim[1]):
        fails.append(
            f"{tag}: per-token norm ratio [{st['rmin']:.4f}, {st['rmax']:.4f}] outside {ratio_lim} "
            "(zeroed rows, dropped pairs, clamps, bfp8 activations?)"
        )
    if not st["row_rel"] <= max_row:
        fails.append(
            f"{tag}: worst per-token rel L2 {st['row_rel']:.4f} > {max_row} at row {st['worst_row']} "
            "(dropped (token, expert) pair, small expert or clamp?)"
        )
    if not (coef_lim[0] <= st["coef"] <= coef_lim[1]):
        fails.append(f"{tag}: scale coefficient {st['coef']:.5f} outside {coef_lim} (fidelity or scale bug)")
    if block_lim and not (block_lim[0] <= st["bmin"] and st["bmax"] <= block_lim[1]):
        fails.append(
            f"{tag}: {BLOCK_ROWS}-row block coefficients [{st['bmin']:.5f}, {st['bmax']:.5f}] outside {block_lim} "
            "(one sequence shard or mesh row scaled?)"
        )
    return fails


def _run(fn, ref, g, chunk, scale: float = 1.0):
    gl = g.layer(chunk, LAYER)
    st = _step(ref, LAYER, STEP)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    inputs[0] = inputs[0] * scale
    out = fn(reference_ctx(ref, LAYER, g, chunk), device_ctx(LAYER, g, chunk), *inputs)
    return out, gl[st.output], inputs


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out, want, inputs = _run(fn, ref, g, c)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale, row, block and clamp checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    lims = (MAX_REL_L2, ROW_NORM_RATIO, MAX_ROW_REL_L2, SCALE_COEF, BLOCK_COEF)
    fails = _check("golden", out, want, *lims)

    # Same module on the layer's other dumped chunk (start 0).
    if c != 0:
        out0, want0, _ = _run(fn, ref, g, 0)
        _, ok0 = compare(f"pcc_{STEP}_L{LAYER:02d}_c0", out0, want0, mode, thr)
        if not ok0:
            fails.append(f"chunk 0: PCC below {thr}")
        fails += _check("golden_c0", out0, want0, *lims)

    # Clamp probe: the golden barely reaches the swiglu limit, so run a scaled input against the CPU reference.
    out_p, _, inputs_p = _run(fn, ref, g, c, CLAMP_PROBE_SCALE)
    want_p = ref.component(LAYER, STEP)(reference_ctx(ref, LAYER, g, c), *inputs_p)
    fails += _check(
        "clamp_probe",
        out_p,
        want_p,
        PROBE_MAX_REL_L2,
        PROBE_ROW_NORM_RATIO,
        PROBE_MAX_ROW_REL_L2,
        PROBE_SCALE_COEF,
        None,
    )
    assert not fails, "; ".join(fails)

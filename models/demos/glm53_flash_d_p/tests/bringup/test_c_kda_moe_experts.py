# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.experts.test.1), built from the frozen layer-3 test (test_c_dsa_moe_experts.py) with every limit
re-measured on layer 4. experts_out = sum_e router[t, e] * down_e(silu(min(gate_e(x), 10)) * clamp(up_e(x), -10, 10)),
x = ffn_norm [2048, 4096] (|x| <= 1.52, median 0.15), router the golden dense [2048, 288] bf16 routing matrix (8 nonzeros
per row, rows sum to 2.5, weights 0.032..1.77), 288 fp8 experts (128x128 block scales) of width 2048. Golden bf16, row
norms 0.65..12.7 (median 3.9). Tokens per expert 3..974 (hottest 271, coldest 254; every expert used; chunk 0 5..967);
pairs per chip (72 experts each) 3461 / 4055 / 4009 / 4859.
The clamps never engage on this golden: max gate 3.45, max |up| 3.41, so no clamp, a wrong limit or a clamp on
silu(g) * u all score exactly the reference there.
Measured on chunk 1 (host script, fp32 weights unless noted) as PCC / rel L2 / per-token norm ratio / worst per-token
rel L2 / scale coefficient <got, want> / <want, want>:
  CPU reference on the bf16 golden inputs 0.999996 / 0.0029 / [0.9966, 1.0034] / 0.0041 / 0.99993; all bf16 (W, x, h,
  y) 0.0053 / [0.9939, 1.0067] / 0.0087 / 0.99966;
  device-like (bfp8 weights, blocks along the output dim, bf16 x / h / y / out) 0.999947 / 0.0103 / [0.9940, 1.0048] /
  0.0129 / 0.99978 (chunk 0: 0.0103 / [0.9943, 1.0051] / 0.0129); 128-row-block coefficients [0.99949, 1.00006];
  bfp8 x 0.0128 / [0.9947, 1.0058] / 0.0170; bfp8 x and h (the fused kernel's default path) 0.0187 / [0.9842, 1.0079] /
  0.0244 (fails rel, ratio and worst row: use high_precision); weights truncated to bf16 0.0077 / [0.9896, 0.9967] /
  coef 0.99308 (fails coef); 6-bit mantissa truncation (HiFi2-like) 0.0141 / [0.9829, 0.9907] / coef 0.98683;
  x1.005 0.0057 / coef 1.00493 (fails coef); x1.01 0.0103 / [1.0066, 1.0134] / 1.00993;
  second sequence half (mesh row 1) x1.005 0.0045 / coef 1.00241 (passes the global coef) / 128-row block max 1.00511;
  rows 0..127 x1.005 block max 1.00516 / ratio max 1.0083 (x1.01: 1.0133); no routed scale 2.5 PCC 0.999996 (passes) /
  rel 0.60; drop hottest expert (974 tokens) 0.85; drop the coldest expert (3 tokens) 0.99963 (passes) / 0.027 / min
  0.63 / worst row 0.68; drop expert 0 0.99985 (passes) / 0.0175 / worst row 0.41; drop expert 287 0.9993 / 0.038;
  drop one chip's 72 experts 0.76..0.93; drop every token's smallest pair 0.9964 / 0.085; drop one token's smallest pair
  0.999993 (passes) / 0.0031..0.0038 (passes rel) / worst row 0.035..0.128 on 6 sampled tokens (layer 3: 0.0085..0.024,
  a gap there; the routed outputs are smaller here, so the smallest pair is a larger share of a row); drop token 0's
  top-1 pair 0.99986 / 0.0167 / worst row 0.67; last row zeroed 0.99990 (passes) / 0.0140 / min 0; last 32 rows 0.9942;
  second half zeroed 0.71; per-expert capacity 512 0.94 (the hottest expert takes 974 of 2048 tokens, so the capacity
  must be the whole chunk); uniform routing 0.84; gelu_tanh 0.9929; gate and up swapped 0.89; chip 0's local expert
  index off by one 0.90.
Clamp probe: the golden never reaches the limit, so the module also runs on 16 * x (exact in bf16; gate > 10 on 1.8% of
entries, |up| > 10 on 4.2%, like layer 3's 8 * x) against the CPU reference of the same input: device-like 0.0109 /
[0.9985, 1.0014] / 0.0131 / 0.99971; bfp8 x and h 0.0196 / 0.0239; truncated bf16 weights coef 0.99485; no gate clamp
0.28 / max 1.48; no up clamp 0.25 / 1.44; up clamped above only 0.17, below only 0.17; limit 7 0.33; limit 9.5 0.050;
limit 9.9 0.0098 / [0.9899, 1.0] / coef 0.99306; limit 10.1 0.0097 / coef 1.00677; limit 10.5 0.047; silu(g) * u
clamped instead 0.79. Clamping gate below at -10 too equals the reference (silu(-10) is 0 to bf16 precision). At 8 * x
the clamps engage on only 0.03% and limit 9.9 / 10.1 give coef 0.99815 / 1.00178 (they would pass), so the probe uses 16.
The gated metric is PCC (pcc_experts_L04, chunk 1). It misses scale errors, a missing routed scale, dropped rows, pairs
and small experts, and the clamps. So the test also asserts, on chunk 1 and on chunk 0 (start 0): shape, finite, rel L2
<= 0.015, per-token norm ratio in [0.988, 1.012], worst per-token rel L2 <= 0.02, scale coefficient in [0.997, 1.003]
and every 128-row block's coefficient in [0.996, 1.004] (the sequence is split over the mesh rows). On the clamp probe
(chunk 1): rel L2 <= 0.015, ratio [0.985, 1.015], worst row <= 0.02, coefficient [0.997, 1.003]. Layer 3's limits
(rel 0.012, worst row 0.025) do not carry over: layer 4's device-like floor is rel 0.0103 (layer 3: 0.0070), and a
dropped pair is visible here. Limits are written ``not x <= lim`` so NaN fails.
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
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.988, 1.012)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.02  # worst per-token ||got - want|| / ||want|| (dropped pairs, small experts, clamps)
SCALE_COEF = (0.997, 1.003)  # <got, want> / <want, want>
BLOCK_ROWS = 128
BLOCK_COEF = (0.996, 1.004)  # the same coefficient on every 128-row block
CLAMP_PROBE_SCALE = 16.0  # input multiplier that makes the swiglu clamps (limit 10) engage
PROBE_MAX_REL_L2 = 0.015
PROBE_ROW_NORM_RATIO = (0.985, 1.015)
PROBE_MAX_ROW_REL_L2 = 0.02
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

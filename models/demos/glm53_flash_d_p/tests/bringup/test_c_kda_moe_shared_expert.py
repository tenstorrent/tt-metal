# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: shared_expert of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.shared_expert.test.1): shared_out = down(silu(min(gate(x), 10)) * clamp(up(x), -10, 10)), one
shared expert of width 2048, fp8 weights with 128x128 block scales, x = ffn_norm [2048, 4096] bf16 (|x| <= 1.52, row
RMS 0.13..0.31). Golden bf16, row norms 0.38..5.3 (chunk 1; median 1.24), much smaller than layer 3 (2.8..62).
Unlike layer 3, the clamps never engage on this golden: max gate 2.16, max |up| 4.72 (chunk 0: 2.41 / 4.61), so every
clamp bug (none, one-sided, limit 7 / 9.5 / 9.9 / 10.1, clamping silu(g) * u) scores exactly the correct output.
Measured on chunk 1 (host script, fp32 weights unless noted) as PCC / rel L2 / per-token norm ratio / worst per-token
rel L2 / scale coefficient <got, want> / <want, want> / 128-row block coefficients:
  CPU reference on the bf16 golden input 0.999998 / 0.0019 / [0.9993, 1.0008] / 0.0026 / 0.99997; planned device path
  (bf16 weights, fp32 intermediates, bf16 out) 0.999997 / 0.0026 / [0.9989, 1.0006] / 0.0035 / 0.99983 /
  [0.99978, 0.99988] (the bf16 copy of the fp8 weights shifts the scale down here, up at layer 3); all bf16 0.0029 /
  [0.9986, 1.0009] / 0.0042; bfp8 weights 0.999984 / 0.0057 / [0.9988, 1.0023] / 0.0082 / 1.00054 (passes);
  bfp8 h 0.0073 (fails rel); bfp8 x and h 0.0085 / [0.9960, 1.0037] / 0.0120 (fails rel: keep activations bf16);
  bfp8 weights, x and h 0.0098; weights truncated to 7 mantissa bits (a HiFi2-like bias) 0.0059 / [0.9935, 0.9965] /
  coef 0.99492 (fails ratio and coef); 6 bits 0.0111 / coef 0.9898; x1.002 coef 1.0018 (passes; the known bf16
  all_reduce bias is about this size); x1.005 0.0055 / [1.0039, 1.0056] / coef 1.0048 (fails ratio and coef);
  x1.01 0.0102; x0.99 min 0.9889; second sequence half x1.005 coef 1.0025 / block max 1.0049 (fails blocks);
  rows 0..127 x1.01 block max 1.0098; row 0 zeroed 0.0270 / min 0 / worst row 1.0; last row zeroed 0.0115 / min 0;
  last 32 rows 0.9948 / 0.109; gelu or gelu_tanh 0.942 / 0.35; gate and up swapped 0.74; no activation 0.62;
  one of four TP shards (512 columns) dropped 0.93..0.94 / 0.35..0.38, doubled 0.972.
  Chunk 0 matches chunk 1 to within 0.0003 on every device-like row.
Clamp probe: the module also runs on 16 * x (exact in bf16; gate > 10 on 0.27% of entries, |up| > 10 on 1.08%) against
the CPU reference of the same input. Layer 3's 4 * x engages almost nothing here (gate 0%, |up| 0.002%) and at 8 * x
limit 9.9 / 10.1 pass the coefficient (0.99890 / 1.00103). At 16 * x: planned path 0.0025 / [0.9989, 1.0012] / 0.0041 /
0.99989; bfp8 weights 0.0071 / [0.9963, 1.0039] / 0.0158 / 1.00048 (passes); bfp8 x and h 0.0103 (fails); limit 9.9
0.0086 / min 0.9873 / coef 0.99412, 10.1 0.0085 / max 1.0128 / coef 1.00575 (fail); limit 9.5 0.044; no gate clamp
0.23; no up clamp 0.21; up clamped above only 0.14, below only 0.14; clamping silu(g) * u 0.74. Numerically equal
variants pass: gate clamped to [-10, 10] (rel 0.0001), min(silu(g), 10) * u. Chunk 0 at 16 * x within 0.0005.
The gated metric is PCC (pcc_shared_expert_L04, chunk 1). It misses scale, fidelity, row and clamp bugs, so the
test also asserts, on chunk 1 and on chunk 0 (start 0): shape, finite, rel L2 <= 0.007, per-token norm ratio in
[0.995, 1.005], worst per-token rel L2 <= 0.012, scale coefficient in [0.997, 1.003], every 128-row block's
coefficient in [0.997, 1.003]; and on the clamp probe (chunk 1): rel L2 <= 0.008, ratio [0.993, 1.007], worst row
<= 0.02, coefficient [0.997, 1.003]. Limits are written ``not x <= lim`` so NaN fails.
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
STEP = "shared_expert"
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.007  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.995, 1.005)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.012  # worst per-token ||got - want|| / ||want|| (zeroed rows, clamps)
SCALE_COEF = (0.997, 1.003)  # <got, want> / <want, want>
BLOCK_ROWS = 128
BLOCK_COEF = (0.997, 1.003)  # the same coefficient on every 128-row block
CLAMP_PROBE_SCALE = 16.0  # input multiplier (exact in bf16) that makes the swiglu clamps (limit 10) engage more
PROBE_MAX_REL_L2 = 0.008
PROBE_ROW_NORM_RATIO = (0.993, 1.007)
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
        fails.append(f"{tag}: relative L2 error {st['rel']:.4f} > {max_rel} (precision, bfp8 activations or scale?)")
    if not (ratio_lim[0] <= st["rmin"] and st["rmax"] <= ratio_lim[1]):
        fails.append(
            f"{tag}: per-token norm ratio [{st['rmin']:.4f}, {st['rmax']:.4f}] outside {ratio_lim} "
            "(zeroed rows, clamp limit, scale?)"
        )
    if not st["row_rel"] <= max_row:
        fails.append(
            f"{tag}: worst per-token rel L2 {st['row_rel']:.4f} > {max_row} at row {st['worst_row']} "
            "(zeroed row or missing clamp?)"
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

    # Clamp probe: a scaled input, against the CPU reference of the same input, engages the swiglu limit more often.
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

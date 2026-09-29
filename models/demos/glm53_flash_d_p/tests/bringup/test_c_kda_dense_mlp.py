# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: mlp of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.mlp.test.1): mlp_out = down(silu(min(gate(x), 10)) * clamp(up(x), -10, 10)), dense 12288-wide,
fp8 weights with 128x128 block scales, input ffn_norm [2048, 4096] (row RMS 0.032..0.057, |x| up to 1.53), output row
norms 0.22..0.69. The gated metric is PCC, which is scale-invariant. Measured on this golden (host script, fp32 weights
unless noted) as PCC / rel L2 / per-token norm ratio / worst per-token rel L2 / global scale coefficient
<got, want> / <want, want>:
  fp32 CPU 0.999997 / 0.0026 / [0.9986, 1.0013] / 0.0031 / 1.0000; bf16 weights, intermediates and output
  0.999993 / 0.0039 / [0.9983, 1.0011] / 0.0044 / 0.9998; bfp8 weights 0.0068..0.0089 / worst row 0.0078..0.0105;
  bfp8 x and h (bf16 weights) 0.99968 / 0.025 / [0.9965, 1.0036] / 0.034 (fails: outlier channels, see known issues);
  weights truncated to 7 mantissa bits (a HiFi2-like bias) 0.999996 / 0.0068 / [0.9925, 0.9952] / 0.0080 / 0.9938;
  gelu instead of silu 0.99953 / 0.031 / worst row 0.046; gate and up swapped 0.9983 / 0.059; no activation 0.9987 /
  1.03; x1.01 1.0 / 0.0103 / [1.0086, 1.0113]; x0.99 [0.9886, 0.9912]; row 0 zeroed 0.99968 / 0.025; last row zeroed
  0.99987 / 0.016; one TP shard of four counted twice 0.978 / 0.36; one dropped 0.943.
The clamps never engage on this golden (|gate| <= 0.70, |up| <= 0.66), so no golden metric can see them. The clamp
probe feeds 48 * x (gate > 10 on 0.007% of entries, |up| > 10 on 0.008%) and compares with the CPU reference of the
same input: bf16 path 0.0030 / [0.9986, 1.0011] / worst row 0.0037; no gate clamp worst row 0.27 / ratio max 1.108;
no up clamp 0.19; up clamped above only 0.070, below only 0.19; limit 7 0.19; clamping silu(g) * u instead 0.45.
Numerically equivalent variants pass (gate clamped to [-10, 10], min(silu(g), 10); limit 9.5 scores 0.029).
Extra checks: finite; vs golden rel L2 <= 0.015, per-token ratio [0.99, 1.01], worst row <= 0.02, scale coefficient in
[0.995, 1.005]; clamp probe rel L2 <= 0.01, ratio [0.99, 1.01], worst row <= 0.03.
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
STEP = "mlp"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.015  # ||got - want|| / ||want||
ROW_NORM_RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.02  # worst per-token ||got - want|| / ||want||
SCALE_COEF = (0.995, 1.005)  # <got, want> / <want, want>
CLAMP_PROBE_SCALE = 48.0  # input multiplier that makes the swiglu clamps (limit 10) engage
PROBE_MAX_REL_L2 = 0.01
PROBE_ROW_NORM_RATIO = (0.99, 1.01)
PROBE_MAX_ROW_REL_L2 = 0.03


def _stats(got: torch.Tensor, want: torch.Tensor) -> dict:
    w = want.float()
    g = got.float().reshape(w.shape)
    wn = w.norm(dim=-1).clamp_min(1e-12)
    ratio = g.norm(dim=-1) / wn
    return {
        "finite": bool(torch.isfinite(g).all()),
        "rel": ((g - w).norm() / w.norm()).item(),
        "rmin": ratio.min().item(),
        "rmax": ratio.max().item(),
        "row_rel": ((g - w).norm(dim=-1) / wn).max().item(),
        "coef": ((g * w).sum() / (w * w).sum()).item(),
    }


def _check(tag: str, st: dict, max_rel: float, ratio_lim: tuple, max_row: float, coef_lim: tuple | None) -> list[str]:
    for k in ("rel", "rmin", "rmax", "row_rel", "coef"):
        metrics.record(f"{k}_{tag}_{STEP}_L{LAYER:02d}", st[k])
    print(
        f"{tag}: rel_l2={st['rel']:.6f} (<= {max_rel}) row_norm_ratio=[{st['rmin']:.4f}, {st['rmax']:.4f}] "
        f"(in {ratio_lim}) max_row_rel_l2={st['row_rel']:.4f} (<= {max_row}) coef={st['coef']:.5f}"
        + (f" (in {coef_lim})" if coef_lim else "")
    )
    fails = []
    if not st["finite"]:
        fails.append(f"{tag}: non-finite output")
    if st["rel"] > max_rel:
        fails.append(f"{tag}: relative L2 error {st['rel']:.4f} > {max_rel}")
    if not (ratio_lim[0] <= st["rmin"] and st["rmax"] <= ratio_lim[1]):
        fails.append(f"{tag}: per-token norm ratio [{st['rmin']:.4f}, {st['rmax']:.4f}] outside {ratio_lim}")
    if st["row_rel"] > max_row:
        fails.append(f"{tag}: worst per-token rel L2 {st['row_rel']:.4f} > {max_row}")
    if coef_lim and not (coef_lim[0] <= st["coef"] <= coef_lim[1]):
        fails.append(f"{tag}: scale coefficient {st['coef']:.5f} outside {coef_lim} (fidelity or scale bug)")
    return fails


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

    # Scale, activation and row checks (PCC misses them). Informational metrics, not in the runner's threshold list.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    fails = _check("golden", _stats(out, want), MAX_REL_L2, ROW_NORM_RATIO, MAX_ROW_REL_L2, SCALE_COEF)

    # Clamp probe: the golden never reaches the swiglu limit, so run a scaled input against the CPU reference.
    x_probe = inputs[0] * CLAMP_PROBE_SCALE
    cpu = ref.component(LAYER, STEP)
    want_probe = cpu(reference_ctx(ref, LAYER, g, c), x_probe)
    out_probe = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), x_probe)
    assert out_probe.numel() == want_probe.numel(), f"probe output has {out_probe.numel()} elements"
    fails += _check(
        "clamp_probe",
        _stats(out_probe, want_probe),
        PROBE_MAX_REL_L2,
        PROBE_ROW_NORM_RATIO,
        PROBE_MAX_ROW_REL_L2,
        None,
    )
    assert not fails, "; ".join(fails)

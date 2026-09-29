# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc_pre of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.ffn_hc_pre.test.1). The step is ffn_x [S, H] = sum_j pre_j * stream_j from h_mid [S, 4H] and
the iHC gates ffn_hc [S, 8] (pre = columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16: h_mid, ffn_hc and
ffn_x. Unlike attn_hc_pre, the four h_mid streams are distinct, but the pre gates are small and very unequal (column
means 0.094, 4e-6, 0.0009, 0.018): ffn_x is mostly 0.094 x stream 0 + 0.018 x stream 3, stream 1 contributes 8e-5 of
the norm. The golden was computed from fp32 upstream values, so even the fp32 CPU step on the bf16 golden inputs is
rel 0.0024 off it (bf16 rounding of the gates). Measured on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference (CPU step)            0.999997   0.00241   [0.9966, 1.0032]     0.0041
    bf16 output                          0.999996   0.00282   [0.9965, 1.0032]     0.0045
    bf16 accumulation (4 terms)          0.999995   0.00322   [0.9966, 1.0035]     0.0050
    pre x 1.005                          0.999997   0.00550   [1.0016, 1.0083]     0.0086
    pre x 1.01                           0.999997   0.0102    [1.0065, 1.0133]     0.0135
    pre + 3e-4                           0.999994   0.0163    [1.0042, 1.0215]     0.0216
    stream 1 dropped                     0.999997   0.00241   [0.9965, 1.0032]     0.0041
    stream 2 dropped                     0.999934   0.0186    [0.9318, 0.9997]     0.072
    streams 1 / 0 swapped                0.995804   0.110     [1.0245, 1.0970]     0.22
    streams 2 / 0 swapped                0.997917   0.0653    [0.9151, 1.0084]     0.30
    last row zeroed                      0.999567   0.0294    [0.0000, 1.0032]     1.0
    last tile row zeroed                 0.991586   0.130     [0.0000, 1.0032]     1.0
    last 32 columns zeroed               0.997754   0.0670    [0.9945, 1.0017]     0.087
    gate j on stream j + 1               0.987648   0.167     [0.8962, 1.0069]     0.24
    stream blocks chip-major             0.680231   0.735     [0.6900, 0.9240]     0.92

Every row from "pre x 1.005" to "last 32 columns zeroed" passes the 0.99 PCC gate. So the test also checks, against
the golden: output finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.994, 1.006], worst row rel L2
<= 0.01. Against the CPU step on the same (golden) inputs, which removes the golden's own rounding: rel L2 <= 0.003,
worst row <= 0.006 (bf16 output 0.0017 / 0.0017, bf16 accumulation 0.0022 / 0.0030; pre x 1.005 0.0050, pre + 1e-4
0.0054 / 0.0069, stream 2 dropped 0.018 / 0.072).

"Stream 1 dropped" is invisible on the golden (its gate is ~4e-6), so the test runs the module a second time with each
row's pre gates rotated by (row mod 4): every stream meets the large gate 0 on a quarter of the rows. That output is
compared with the CPU step on the same inputs:

    variant (rotated gates)              PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00166   0.0018
    bf16 accumulation                    0.999998   0.00198   0.0030
    stream 1 dropped                     0.945425   0.340     0.92
    streams 1 / 0 swapped                0.997923   0.0649    0.22
    streams 1 / 2 swapped                0.995260   0.0981    0.48
    gate rotation ignored                0.945786   0.325     0.82

Limits there: rel L2 <= 0.004, worst row rel L2 <= 0.01.
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
STEP = "ffn_hc_pre"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.005  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.994, 1.006)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.01  # worst per-row rel L2, vs the golden
CPU_MAX_REL_L2 = 0.003  # vs the CPU step on the same (golden) inputs
CPU_MAX_ROW_REL = 0.006
ROT_MAX_REL_L2 = 0.004  # rotated pre gates, vs the CPU step on the same inputs
ROT_MAX_ROW_REL = 0.01
HC_MULT = 4


def _errors(got: torch.Tensor, want: torch.Tensor):
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm()).item()
    wn = want.norm(dim=-1).clamp_min(1e-30)
    ratio = got.norm(dim=-1) / wn
    row = ((got - want).norm(dim=-1) / wn).max().item()
    return rel, ratio.min().item(), ratio.max().item(), row


def _rotated(gates: torch.Tensor):
    """Each row's pre gates rotated by (row mod 4), so every stream meets the large gate on a quarter of the rows."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return gs


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
    ), "device_component returned a CPU bridge; ffn_hc_pre is not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # PCC cannot see a per-row scale or a small dropped stream; check the error itself.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    assert torch.isfinite(out.float()).all(), "non-finite output"
    rel, rmin, rmax, row = _errors(out, want)
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"worst_row_rel_l2_{STEP}_L{LAYER:02d}", row)
    print(
        f"golden: rel_l2={rel:.6f} (<= {MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(RATIO)}) "
        f"worst_row_rel_l2={row:.5f} (<= {MAX_ROW_REL})"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2}"
    assert RATIO[0] <= rmin and rmax <= RATIO[1], f"row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(RATIO)}"
    assert row <= MAX_ROW_REL, f"worst row rel L2 {row:.5f} > {MAX_ROW_REL}"

    # Same inputs through the CPU step: removes the golden's own bf16-gate rounding (rel 0.0024).
    cpu_step = ref.component(LAYER, STEP)
    cpu = cpu_step(rctx, *inputs).float()
    crel, cmin, cmax, crow = _errors(out, cpu)
    metrics.record(f"cpu_rel_l2_{STEP}_L{LAYER:02d}", crel)
    print(
        f"vs CPU step: rel_l2={crel:.6f} (<= {CPU_MAX_REL_L2}) row norm ratio=[{cmin:.5f}, {cmax:.5f}] "
        f"worst_row_rel_l2={crow:.5f} (<= {CPU_MAX_ROW_REL})"
    )
    assert crel <= CPU_MAX_REL_L2, f"vs CPU step: relative L2 error {crel:.5f} > {CPU_MAX_REL_L2}"
    assert crow <= CPU_MAX_ROW_REL, f"vs CPU step: worst row rel L2 {crow:.5f} > {CPU_MAX_ROW_REL}"

    # Stream 1's gate is ~4e-6 at layer 0: rotate the pre gates per row so every stream carries weight somewhere.
    streams, gates = inputs
    rot = (streams, _rotated(gates))
    rot_want = cpu_step(rctx, *rot).float()
    rot_out = fn(rctx, dctx, *rot)
    assert rot_out.numel() == rot_want.numel(), f"rotated-gates output has {rot_out.numel()} elements"
    assert torch.isfinite(rot_out.float()).all(), "non-finite output (rotated gates)"
    srel, smin, smax, srow = _errors(rot_out, rot_want)
    metrics.record(f"rot_rel_l2_{STEP}_L{LAYER:02d}", srel)
    print(
        f"rotated pre gates: pcc={metrics.pcc(rot_out.float().reshape(rot_want.shape), rot_want):.6f} "
        f"rel_l2={srel:.6f} (<= {ROT_MAX_REL_L2}) row norm ratio=[{smin:.5f}, {smax:.5f}] "
        f"worst_row_rel_l2={srow:.5f} (<= {ROT_MAX_ROW_REL})"
    )
    assert srel <= ROT_MAX_REL_L2, f"rotated gates: relative L2 error {srel:.5f} > {ROT_MAX_REL_L2} (stream order?)"
    assert srow <= ROT_MAX_ROW_REL, f"rotated gates: worst row rel L2 {srow:.5f} > {ROT_MAX_ROW_REL} (stream order?)"

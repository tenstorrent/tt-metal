# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc_pre of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.ffn_hc_pre.test.1). The step is ffn_x [S, H] = sum_j pre_j * stream_j from h_mid [S, 4H] and
the iHC gates ffn_hc [S, 8] (pre = columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16: h_mid, ffn_hc and
ffn_x. The four h_mid streams are distinct (mean row norms 1.14, 1.05, 1.20, 3.02), but pre gates 0 and 1 sit at
hc_eps (column means 1.0e-5, 1.5e-6, 0.93, 0.33): ffn_x is streams 2 and 3 only; streams 0 and 1 contribute 1e-5
and 2e-6 of the norm. The golden came from fp32 upstream values, so the fp32 CPU step on the bf16 golden inputs is
already rel 0.0026 off it. Measured on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference (CPU step)            0.999997   0.00260   [0.9965, 1.0038]     0.0054
    bf16 output                          0.999995   0.00301   [0.9965, 1.0038]     0.0057
    bf16 accumulation (4 terms)          0.999995   0.00327   [0.9965, 1.0039]     0.0057
    stream 0 dropped                     0.999997   0.00260   [0.9965, 1.0038]     0.0054
    stream 1 dropped                     0.999997   0.00260   [0.9965, 1.0038]     0.0054
    streams 0 / 1 swapped                0.999997   0.00260   [0.9965, 1.0038]     0.0054
    pre + 3e-4                           0.999997   0.00275   [0.9971, 1.0045]     0.0059
    pre x 1.005                          0.999997   0.00555   [1.0014, 1.0088]     0.0096
    pre x 1.01                           0.999997   0.0102    [1.0064, 1.0139]     0.0144
    last row zeroed                      0.999584   0.0289    [0.0000, 1.0038]     1.0
    SP rows 1023 / 1024 swapped          0.999877   0.0157    [0.7608, 1.3126]     1.21
    last tile row zeroed                 0.992139   0.125     [0.0000, 1.0038]     1.0
    last 32 columns zeroed               0.997522   0.0704    [0.9939, 1.0019]     0.096
    streams 0 / 2 swapped                0.971409   0.240     [0.7851, 1.1285]     0.48
    rows of the gates shifted by 1       0.941953   0.386     [0.3713, 2.9136]     2.2
    gate j on stream j + 1               0.821695   0.570     [0.5341, 0.9468]     1.2
    stream 3 dropped                     0.849969   0.542     [0.2509, 0.9795]     1.2
    stream blocks chip-major             0.491022   0.972     [0.7703, 1.0609]     1.06

Every row from "stream 0 dropped" to "last 32 columns zeroed" passes the 0.99 PCC gate. So the test also checks,
against the golden: output finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.994, 1.006], worst
row rel L2 <= 0.01. Against the CPU step on the same (golden) inputs, which removes the golden's own rounding: rel L2
<= 0.003, worst row <= 0.006 (bf16 output 0.0017 / 0.0017, bf16 accumulation 0.0021 / 0.0026; pre x 1.005 0.0050 /
0.0050).

Streams 0 and 1 are invisible on the golden: dropping either, or swapping them, is bit-identical to the reference.
The test runs the module a second time with each row's pre gates rotated by (row mod 4), so every stream meets the
large gate 2 on a quarter of the rows, and compares with the CPU step on the same inputs:

    variant (rotated gates)              PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00166   0.0017
    bf16 accumulation                    0.999998   0.00188   0.0026
    pre x 1.005                          1.000000   0.00500   0.0050
    streams 0 / 1 swapped                0.993150   0.118     0.36
    streams 0 / 2 swapped                0.987403   0.159     0.56
    stream 1 dropped                     0.954589   0.302     0.96
    stream 0 dropped                     0.944666   0.330     0.94
    gate rotation ignored                0.795154   0.613     1.9

Limits there: rel L2 <= 0.004, worst row rel L2 <= 0.01. Not caught: pre + 3e-4 on every gate (golden rel 0.0028);
the gates are an input, so the module cannot make that error by itself.
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
LAYER = 1
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

    # PCC cannot see a per-row scale or a row swap; check the error itself.
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

    # Same inputs through the CPU step: removes the golden's own rounding (rel 0.0026).
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

    # Pre gates 0 / 1 sit at hc_eps here: rotate the pre gates per row so every stream meets the large gate.
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

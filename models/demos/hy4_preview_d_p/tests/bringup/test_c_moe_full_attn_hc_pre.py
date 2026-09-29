# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc_pre of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.attn_hc_pre.test.1). The step is attn_x [S, H] = sum_j pre_j * stream_j from the block input
streams [S, 4H] and the iHC gates attn_hc [S, 8] (pre = columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16:
streams, gates and attn_x. Unlike layer 0 the four streams are distinct (mean row norms 1.14, 1.05, 1.19, 2.25), but
the pre gates are very unequal (column means 0.026, 0.013, 0.83, 0.49): attn_x is mostly streams 2 and 3; streams 0
and 1 contribute about 2 % and 1 % of the norm. The golden came from fp32 upstream values, so the fp32 CPU step on
the bf16 golden inputs is already rel 0.0026 off it. Measured on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference (CPU step)            0.999997   0.00260   [0.9969, 1.0031]     0.0050
    bf16 output                          0.999996   0.00299   [0.9968, 1.0032]     0.0053
    bf16 accumulation (4 terms)          0.999995   0.00320   [0.9968, 1.0031]     0.0053
    pre x 1.005                          0.999997   0.00566   [1.0019, 1.0082]     0.0089
    pre x 1.01                           0.999997   0.0104    [1.0068, 1.0132]     0.0136
    stream 0 dropped                     0.999888   0.0218    [0.9115, 1.0018]     0.093
    stream 1 dropped                     0.999967   0.0126    [0.9561, 1.0023]     0.052
    streams 0 / 1 swapped                0.999987   0.00504   [0.9920, 1.0051]     0.027
    streams 0 / 2 swapped                0.998467   0.0581    [0.9883, 1.1935]     0.27
    last row zeroed                      0.999448   0.0333    [0.0000, 1.0031]     1.0
    last tile row zeroed                 0.991239   0.132     [0.0000, 1.0031]     1.0
    last 32 columns zeroed               0.997723   0.0675    [0.9938, 1.0014]     0.095
    SP rows 1023 / 1024 swapped          0.999888   0.0149    [0.6916, 1.4480]     1.35
    rows of the gates shifted by 1       0.971888   0.244     [0.4903, 2.1146]     1.6
    streams 1 / 2 swapped                0.981616   0.202     [0.9868, 1.3234]     0.47
    gate j on stream j + 1               0.893652   0.452     [0.5663, 1.0435]     1.2
    stream blocks chip-major             0.518882   0.986     [0.8948, 1.1108]     1.1

Every row from "pre x 1.005" to "SP rows swapped" passes the 0.99 PCC gate. So the test also checks, against the
golden: output finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.994, 1.006], worst row rel L2
<= 0.01. Against the CPU step on the same (golden) inputs, which removes the golden's own rounding: rel L2 <= 0.003,
worst row <= 0.006 (bf16 output 0.0017 / 0.0017, bf16 accumulation 0.0020 / 0.0023; pre x 1.005 0.0050 / 0.0050,
streams 0 / 1 swapped 0.0043 / 0.027, stream 1 dropped 0.012 / 0.050).

Streams 0 and 1 carry small gates, so a mix-up between them is only just visible on the golden (0 / 1 swapped: rel
0.00504). The test runs the module a second time with each row's pre gates rotated by (row mod 4), so every stream
meets the large gate 2 on a quarter of the rows, and compares with the CPU step on the same inputs:

    variant (rotated gates)              PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00166   0.0017
    bf16 accumulation                    0.999997   0.00235   0.0033
    pre x 1.005                          1.000000   0.00500   0.0050
    streams 0 / 2 swapped                0.998829   0.0484    0.36
    streams 0 / 1 swapped                0.994762   0.105     0.29
    stream 1 dropped                     0.956842   0.316     0.81
    gate rotation ignored                0.912447   0.413     1.9

Limits there: rel L2 <= 0.004, worst row rel L2 <= 0.01. The device module runs in fp32 (at layer 0 it matched the
fp32 reference to rel 0.00095, identical to the CPU step), so every limit leaves room over bf16 accumulation.
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
STEP = "attn_hc_pre"
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
    ), "device_component returned a CPU bridge; attn_hc_pre is not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # PCC cannot see a per-row scale, a dropped small stream or a row swap; check the error itself.
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

    # Streams 0 / 1 carry small gates at layer 1: rotate the pre gates per row so every stream meets the large gate.
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

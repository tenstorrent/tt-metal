# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_hc_pre of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.ffn_hc_pre.test.1). The step is ffn_x [S, H] = sum_j pre_j * stream_j from h_mid [S, 4H] and
the iHC gates ffn_hc [S, 8] (pre = columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16: h_mid, ffn_hc and
ffn_x. Same checks and limits as the layer-1 test (test_c_moe_full_ffn_hc_pre.py); the layer-2 golden differs in
shape: pre column means 1.1e-3 / 1.4e-6 / 1.0 / 0.10 (gate 2 saturated at exactly 1.0, gate 1 at hc_eps), stream
mean row norms 0.84 / 0.70 / 0.94 / 6.07, contribution norms 1.5e-3 / 1.4e-6 / 0.94 / 0.40. Gate 0 reaches 0.06 on
some rows, so a dropped stream 0 now shows in the worst row; stream 1 is invisible. The fp32 CPU step on the bf16
golden inputs is already rel 0.0024 off the golden. Measured on this golden (CPU; mutations of the reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row   vs CPU rel / row
    fp32 reference (CPU step)            0.999997   0.00243   [0.9971, 1.0029]     0.0057      0 / 0
    bf16 output                          0.999996   0.00287   [0.9970, 1.0030]     0.0060      0.00166 / 0.0018
    bf16 accumulation (4 terms)          0.999995   0.00313   [0.9971, 1.0030]     0.0060      0.00206 / 0.0028
    stream 1 dropped                     0.999997   0.00243   [0.9971, 1.0029]     0.0057      0 / 0
    pre + 3e-4                           0.999996   0.00304   [0.9982, 1.0040]     0.0067      0.00183 / 0.0048
    streams 0 / 1 swapped                0.999996   0.00281   [0.9892, 1.0029]     0.0238      0.00141 / 0.0236
    stream 0 dropped                     0.999993   0.00394   [0.9582, 1.0029]     0.0495      0.00310 / 0.0493
    pre x 1.005                          0.999997   0.00557   [1.0021, 1.0079]     0.0094      0.00500 / 0.0050
    pre x 1.01                           0.999997   0.0103    [1.0071, 1.0130]     0.0139      0.0100 / 0.0100
    SP rows 1023 / 1024 swapped          0.999902   0.0140    [0.7952, 1.2574]     1.22        0.0138 / 1.22
    last row zeroed                      0.999591   0.0286    [0.0000, 1.0029]     1.0
    last 32 columns zeroed               0.997336   0.0730    [0.9939, 1.0010]     0.101
    last tile row zeroed                 0.991914   0.127     [0.0000, 1.0029]     1.0
    drop stream 3                        0.921295   0.389     [0.3512, 1.2892]     1.46
    streams 0 / 2 swapped                0.914375   0.405     [0.7112, 1.1634]     0.63
    rows of the gates shifted by 1       0.865534   0.583     [0.3758, 3.6055]     3.4
    gate j on stream j + 1               0.792465   0.612     [0.3573, 1.0564]     1.1
    stream blocks chip-major             0.404439   1.02      [0.6875, 1.3940]     1.36
    streams 2 / 3 swapped                0.313266   5.31      [1.8740, 16.55]      16.4

Every row from "stream 1 dropped" to "last tile row zeroed" passes the 0.99 PCC gate. So the test also checks,
against the golden: output finite, element count, rel L2 <= 0.005, every row's norm ratio in [0.994, 1.006], worst
row rel L2 <= 0.01. Against the CPU step on the same (golden) inputs, which removes the golden's own rounding: rel L2
<= 0.003, worst row <= 0.006.

Stream 1 is invisible on the golden: dropping it is bit-identical to the reference. The test runs the module a second
time with each row's pre gates rotated by (row mod 4), so every stream meets the large gate 2 on a quarter of the
rows, and compares with the CPU step on the same inputs:

    variant (rotated gates)              PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00156   0.0018
    bf16 accumulation                    0.999999   0.00160   0.0028
    pre x 1.005                          1.000000   0.00500   0.0050
    streams 0 / 1 swapped                0.995825   0.0915    0.73
    streams 0 / 2 swapped                0.994886   0.101     0.82
    stream 1 dropped                     0.992207   0.125     0.98
    stream 0 dropped                     0.988892   0.149     0.99
    gate rotation ignored                0.355282   0.935     2.4

Limits there: rel L2 <= 0.004, worst row rel L2 <= 0.01. Not caught: pre + 3e-4 on every gate (golden rel 0.0030,
row 0.0067; vs CPU 0.0018 / 0.0048); the gates are an input, so the module cannot make that error by itself.
Study script: /tmp/hy4_ssh_ffnhcpre2/study.py (outside the repo).
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
LAYER = 2
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

    # Same inputs through the CPU step: removes the golden's own rounding (rel 0.0024).
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

    # Pre gate 1 sits at hc_eps here: rotate the pre gates per row so every stream meets the large gate.
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

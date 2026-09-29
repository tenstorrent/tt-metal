# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc_pre of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.attn_hc_pre.test.1). Same step and checks as the layer-1 test (test_c_moe_full_attn_hc_pre):
attn_x [S, H] = sum_j pre_j * stream_j from the block input streams [S, 4H] and the iHC gates attn_hc [S, 8] (pre =
columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16: streams, gates and attn_x. At layer 2 the streams are
distinct (mean row norms 1.14, 1.05, 1.20, 6.07) and the pre gates are even more unequal than at layer 1 (column
means 1.7e-6, 0.96, 0.20, 0.032): pre gate 0 sits at hc_eps, so stream 0 contributes 3e-6 of the norm and is
invisible on the golden (dropping it changes nothing); attn_x is mostly stream 1 (norm 0.97 of 1.10), with streams 2
and 3 at about 0.13 each. Measured on this golden (CPU; mutations of the reference; "CPU" = vs the fp32 CPU step on
the same bf16 golden inputs):

    variant                              PCC        rel L2    per-row norm ratio   worst row   CPU rel / row
    fp32 reference (CPU step)            0.999997   0.00253   [0.9977, 1.0023]     0.0035      0 / 0
    bf16 output                          0.999996   0.00298   [0.9977, 1.0022]     0.0039      0.00166 / 0.0017
    bf16 accumulation (4 terms)          0.999993   0.00376   [0.9964, 1.0023]     0.0050      0.00278 / 0.0032
    pre x 1.005                          0.999997   0.00558   [1.0027, 1.0073]     0.0076      0.00500 / 0.0050
    pre x 1.01                           0.999997   0.0103    [1.0076, 1.0123]     0.0125      0.0100 / 0.0100
    stream 0 dropped                     0.999997   0.00253   [0.9977, 1.0023]     0.0035      0 / 0
    streams 0 / 2 swapped                0.999281   0.0380    [0.9905, 1.0601]     0.27        0.038 / 0.26
    last row zeroed                      0.999522   0.0309    [0.0000, 1.0023]     1.0         0.031 / 1.0
    last tile row zeroed                 0.991857   0.127     [0.0000, 1.0023]     1.0         0.128 / 1.0
    last 32 columns zeroed               0.997653   0.0685    [0.9937, 1.0007]     0.099       0.069 / 0.099
    SP rows 1023 / 1024 swapped          0.999812   0.0194    [0.7417, 1.3506]     1.35        0.019 / 1.35
    stream 3 dropped                     0.993763   0.114     [0.9078, 1.0611]     0.56        0.114 / 0.56
    streams 0 / 3 swapped                0.993424   0.115     [0.9403, 1.1724]     0.61        0.114 / 0.61
    stream 2 dropped                     0.996670   0.102     [0.7215, 0.9929]     0.62        0.102 / 0.62
    rows of the gates shifted by 1       0.969581   0.390     [0.4982, 2.2086]     1.28        0.390 / 1.27
    streams 0 / 1 swapped                0.964930   0.283     [1.0035, 1.1256]     0.42        0.283 / 0.42
    streams 1 / 2 swapped                0.923974   0.421     [0.9967, 1.2575]     0.74        0.421 / 0.74
    streams 2 / 3 swapped                0.897240   0.507     [0.9760, 4.0261]     3.6         0.507 / 3.6
    gate j on stream j + 1               0.863952   0.611     [1.0420, 3.9881]     3.5         0.611 / 3.5
    post gates instead of pre            0.898880   0.924     [0.0939, 2.8544]     1.9         0.924 / 1.9

Every row from "pre x 1.005" to "streams 0 / 3 swapped" passes the 0.99 PCC gate (and "stream 0 dropped" passes
everything on the golden). So the test also checks, against the golden: output finite, element count, rel L2 <=
0.005, every row's norm ratio in [0.994, 1.006], worst row rel L2 <= 0.01; against the CPU step on the same inputs
(removes the golden's own rounding, rel 0.0025): rel L2 <= 0.003, worst row <= 0.006. bf16 accumulation (0.00278 /
0.0032) fits; the device module runs in fp32 (identical to the CPU step at layer 0).

Stream 0 is invisible on the golden, so the test runs the module a second time with each row's pre gates rotated by
(row mod 4), so every stream meets the large gate 1 on a quarter of the rows, and compares with the CPU step on the
same inputs:

    variant (rotated gates)              PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00166   0.0017
    bf16 accumulation                    0.999997   0.00240   0.0031
    pre x 1.005                          1.000000   0.00500   0.0050
    SP rows 1023 / 1024 swapped          0.999960   0.00892   1.6
    streams 0 / 1 swapped                0.997323   0.0732    0.41
    streams 0 / 2 swapped                0.995971   0.0898    0.58
    stream 1 dropped                     0.985390   0.171     0.96
    stream 0 dropped                     0.981436   0.192     0.96
    stream 2 dropped                     0.980121   0.199     0.97
    gate rotation ignored                0.453344   0.894     1.0
    streams 0 / 3 swapped                0.215881   1.24      13

Limits there: rel L2 <= 0.004, worst row rel L2 <= 0.01. Every mutation in both tables fails at least one check.
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

    # Same inputs through the CPU step: removes the golden's own rounding (rel 0.0025).
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

    # Stream 0 carries a gate at hc_eps at layer 2: rotate the pre gates per row so every stream meets the large gate.
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

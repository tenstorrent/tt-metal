# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc_pre of block type dense_full (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense_full.attn_hc_pre.test.1). The step is attn_x [S, H] = sum_j pre_j * stream_j from the streams
[S, 4H] and the iHC gates [S, 8] (pre = columns 0-3). The golden (s4096 chunk 1, 2048 rows) is bf16: streams, gates
and attn_x. At layer 0 the four streams are identical (each is the embedding), so attn_x = (sum_j pre_j) * embedding,
and PCC is blind to anything that only changes a row's scale. Measured on this golden (CPU; mutations of the
reference):

    variant                              PCC        rel L2    per-row norm ratio   worst row rel L2
    fp32 reference                       1.000000   0.00095   [0.9980, 1.0017]     0.0026
    bf16 output                          1.000000   0.00073   [0.9974, 1.0024]     0.0043
    bf16 accumulation (4 terms)          0.999999   0.00130   [0.9981, 1.0040]     0.0050
    pre x 1.01                           1.000000   0.00987   [1.0080, 1.0117]     0.0117
    pre x 1.02                           1.000000   0.0198    [1.0180, 1.0217]     0.0217
    no gating (pre = 1)                  0.999815   0.0196    [1.0000, 1.3321]     0.33
    pre gate 0 used for all 4 streams    0.999818   0.0194    [0.9991, 1.3321]     0.33
    gate rows shifted by 1               0.999654   0.0264    [0.7506, 1.3321]     0.33
    last row zeroed                      0.999491   0.0319    [0.0000, 1.0017]     1.0
    one stream dropped                   0.999818   0.249     [0.7435, 0.9991]     0.26
    post gates instead of pre            0.969236   0.693     [0.0577, 0.4809]     0.94

Every row from "pre x 1.01" to "one stream dropped" passes the 0.99 PCC gate. So the test also checks, against the
golden: output finite, shape [S, H], rel L2 <= 0.004, every row's norm ratio in [0.995, 1.005], and worst row rel L2
<= 0.01. pre x 1.005 scores rel 0.0049 / ratio [1.0030, 1.0067] and is caught too.

Layer 0 cannot see stream order (which gate multiplies which stream, which column block is which stream), because
the streams are equal. So the test runs the module a second time on synthetic distinct streams: stream j = the
golden stream with its rows rolled by 7 j, and each row's pre gates rotated by (row mod 4), so every stream meets the
varying gate 3 on a quarter of the rows. That output is compared with the CPU hc_pre on the same inputs:

    variant (synthetic)                  PCC        rel L2    worst row rel L2
    bf16 output                          0.999999   0.00167   0.0018
    bf16 accumulation                    0.999997   0.00258   0.0035
    streams 0 / 1 swapped                0.999695   0.0247    0.89
    streams 2 / 3 swapped                0.999675   0.0255    0.76
    gate j on stream j + 1               0.999333   0.0365    0.89
    streams reversed                     0.999166   0.0409    0.78
    stream blocks chip-major             0.540041   0.930     1.1

Limits there: rel L2 <= 0.008, worst row rel L2 <= 0.02 (device noise about 0.003 / 0.004, smallest bug 0.025 / 0.76).
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
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.004  # ||got - want|| / ||want|| over [S, H], vs the golden
RATIO = (0.995, 1.005)  # per-row ||got|| / ||want||, vs the golden
MAX_ROW_REL = 0.01  # worst per-row rel L2, vs the golden
SYN_MAX_REL_L2 = 0.008  # synthetic distinct streams, vs the CPU hc_pre on the same inputs
SYN_MAX_ROW_REL = 0.02
HC_MULT = 4


def _errors(got: torch.Tensor, want: torch.Tensor):
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm()).item()
    wn = want.norm(dim=-1).clamp_min(1e-30)
    ratio = got.norm(dim=-1) / wn
    row = ((got - want).norm(dim=-1) / wn).max().item()
    return rel, ratio.min().item(), ratio.max().item(), row


def _synthetic(streams: torch.Tensor, gates: torch.Tensor):
    """Distinct streams (golden stream 0 with rows rolled by 7 j) and per-row rotated pre gates."""
    n = streams.shape[0]
    base = streams.view(n, HC_MULT, -1)[:, 0]
    st = torch.stack([torch.roll(base, 7 * j, 0) for j in range(HC_MULT)], 1).reshape(n, -1).contiguous()
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return st, gs


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

    # PCC cannot see a per-row scale at layer 0 (identical streams); check the error itself.
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

    # Stream order: distinct streams, vs the CPU step on the same inputs.
    streams, gates = inputs
    syn = _synthetic(streams, gates)
    syn_want = ref.component(LAYER, STEP)(rctx, *syn).float()
    syn_out = fn(rctx, dctx, *syn)
    assert syn_out.numel() == syn_want.numel(), f"synthetic output has {syn_out.numel()} elements"
    assert torch.isfinite(syn_out.float()).all(), "non-finite output (synthetic streams)"
    srel, smin, smax, srow = _errors(syn_out, syn_want)
    metrics.record(f"syn_rel_l2_{STEP}_L{LAYER:02d}", srel)
    print(
        f"synthetic distinct streams: pcc={metrics.pcc(syn_out.float().reshape(syn_want.shape), syn_want):.6f} "
        f"rel_l2={srel:.6f} (<= {SYN_MAX_REL_L2}) row norm ratio=[{smin:.5f}, {smax:.5f}] "
        f"worst_row_rel_l2={srow:.5f} (<= {SYN_MAX_ROW_REL})"
    )
    assert srel <= SYN_MAX_REL_L2, f"synthetic streams: relative L2 error {srel:.5f} > {SYN_MAX_REL_L2} (stream order?)"
    assert (
        srow <= SYN_MAX_ROW_REL
    ), f"synthetic streams: worst row rel L2 {srow:.5f} > {SYN_MAX_ROW_REL} (stream order?)"

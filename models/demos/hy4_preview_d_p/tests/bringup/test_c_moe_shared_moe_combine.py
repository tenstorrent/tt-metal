# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: moe_combine of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.moe_combine.test.1). The step is mlp_out = experts_out + shared_out (all [S, 6144]; on the
device [S/2, 3072] per chip, both inputs already reduce-scattered, no collective). The golden stores every tensor in
bf16 (s4096 chunk 1, 2048 rows), and golden mlp_out is the bf16 of the fp32 sum, so even the exact sum of the bf16
inputs sits at rel 0.0022 from it. ||experts_out|| 6479, ||shared_out|| 3630, ||mlp_out|| 8216; row norms 17-447
(experts) and 27-250 (shared), shared / experts per row 0.13-8.3. Both addends are large, so (unlike attn_residual)
no bf16 rounding budget is needed. Measured on this golden (CPU, on the golden inputs; "coef" / "add rel" / "add row"
per addend: delta = out - other addend vs this addend, coef = <delta, t> / ||t||^2 in float64; e = experts, s = shared):

    variant                  PCC       rel L2   row norm ratio    worst row  coef s / e       add rel s / e    add row s / e
    fp32 a + b (reference)   0.999997  0.00223  [0.9998, 1.0002]  0.0024     1.0000 / 1.0000  0 / 0            0 / 0
    bf16 output              0.999996  0.00275  [0.9998, 1.0002]  0.0030     1.0000 / 1.0000  0.0040 / 0.0023  0.014 / 0.015
    golden itself            1.0       0        [1, 1]            0          1.0000 / 1.0000  0.0051 / 0.0028  0.019 / 0.020
    1.01 x shared            0.999992  0.00495  [1.0004, 1.0096]  0.0100     1.0100 / 1.0015  0.0100 / 0.0056  0.010 / 0.083
    1.01 x experts           0.999992  0.00820  [1.0004, 1.0096]  0.0099     1.0047 / 1.0100  0.0178 / 0.0100  0.079 / 0.010
    0.99 x shared            0.999992  0.00495  [0.9904, 0.9996]  0.0100     0.9900 / 0.9985  0.0100 / 0.0056  0.010 / 0.083
    last row zeroed          0.999525  0.0308   [0.0, 1.0002]     1.0        0.9986 / 0.9989  0.070 / 0.039    2.36 / 1.27
    last 32 columns zeroed   0.997471  0.0711   [0.9948, 0.9991]  0.101      0.9929 / 0.9941  0.161 / 0.090    0.62 / 0.64
    last 32 rows zeroed      0.991607  0.129    [0.0, 1.0002]     1.0        -                -                -
    2 (a + b)                0.999997  0.9999   [1.9997, 2.0003]  1.0        -                -                -
    (a + b) / 2              0.999997  0.4999   [0.4999, 0.5001]  0.5        -                -                -
    a + 0.5 b                0.981268  0.221    -                 -          -                -                -
    shared dropped           0.904507  0.442    -                 -          -                -                -
    shared shifted 1 row     0.858872  0.518    -                 -          -                -                -
    a - b                    0.535752  0.884    -                 -          -                -                -
    zero stub                nan       1.0      0                 -          -                -                -

The gated PCC (0.99) passes every bug above down to 32 zeroed rows and a 2x scale. So the test also checks, against
the golden: device output (not a CPU bridge), size, finite, rel L2 <= 0.004, per-token norm ratio in [0.995, 1.005],
worst row rel <= 0.01; per addend: |coef - 1| <= 0.002, add rel <= 0.008 (shared) / 0.005 (experts), add row <= 0.03.
It then runs the module again on (experts_out, -shared_out) and on (experts_out, 0) and compares with the exact sums
of those inputs (rel <= 0.004, worst row <= 0.005; a bf16 output scores 0.0017 / 0.0018 on the first, 0 on the
second): this catches a module that does not add its own inputs (a cached result or the golden shared_out: rel 1.11).

Checked on the CPU by patching each variant in as the module: fp32 and bf16 outputs pass; 1.01 x shared, 1.005 and
1.01 x experts, 2 (a + b), last row zeroed and last 32 columns zeroed fail rel L2; 0.995 x shared fails the experts
worst-row addend check (0.041); a cached output and a + golden shared fail the probe; the zero stub fails PCC.
Blind spot: a scale error on shared_out below about 0.5 % passes.

Layer 2 (C.moe_shared.moe_combine.test.1). This file is the layer-1 test above with LAYER = 2; every check and limit
is unchanged. The layer-1 study (tables above) was re-run on the layer-2 golden (s4096 chunk 1, 2048 rows, bf16):
||experts_out|| 1769, ||shared_out|| 1669, ||mlp_out|| 2646 (the two addends are balanced here), row norms 5.5-150
(experts) and 9.9-90 (shared), shared / experts per row 0.12-10.9. Measured on the CPU:

    variant                  PCC       rel L2   row norm ratio    worst row  add rel s / e    add row s / e   verdict
    fp32 a + b (reference)   0.999997  0.00225  [0.9998, 1.0002]  0.0024     0 / 0            0 / 0           PASS
    bf16 output              0.999996  0.00276  [0.9998, 1.0002]  0.0030     0.0028 / 0.0026  0.013 / 0.019   PASS
    golden itself            1.0       0        [1, 1]            0          0.0036 / 0.0034  0.019 / 0.026   -
    1.01 x shared            0.999989  0.00670  [1.0002, 1.0097]  0.0101     -                -               FAIL rel L2
    1.005 x shared           -         0.00388  [1.0001, 1.0049]  0.0055     -                e row 0.054     FAIL add row
    0.997 x shared           -         0.00294  [0.9970, 1.0000]  0.0038     -                e row 0.033     FAIL add row
    1.005 / 0.995 x experts  -         0.00403  -                 0.0056     -                -               FAIL rel L2
    last row zeroed          0.999708  0.0242   [0.0, 1.0002]     1.0        -                -               FAIL rel L2
    last 32 columns zeroed   0.997446  0.0714   [0.9951, 0.9990]  0.099      -                -               FAIL rel L2
    last 32 rows zeroed      0.992423  0.123    -                 1.0        -                -               FAIL rel L2
    2 (a + b)                0.999997  0.9999   [1.9996, 2.0003]  1.0        -                -               FAIL rel L2
    a + 0.5 b                0.964978  0.315    -                 -          -                -               FAIL PCC
    shared shifted 1 row     0.694456  0.766    -                 -          -                -               FAIL PCC
    cached output, a + golden shared                                                                          FAIL probe
    zero stub                nan       1.0                                                                    FAIL PCC

The tightest margin for a correct module is the experts addend worst row of a bf16 output (0.019 of 0.03). The
blind spot shrinks to a shared_out scale error below about 0.3 %, because shared_out is as large as experts_out here.
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
STEP = "moe_combine"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.004  # vs golden (fp32 reference 0.00223, bf16 output 0.00275; 1.01 x shared 0.00495)
ROW_NORM_RATIO = (0.995, 1.005)  # per token ||got|| / ||want|| (bf16 output [0.9998, 1.0002])
MAX_ROW_REL = 0.01  # per token ||got - want|| / ||want|| (bf16 output 0.0030; last 32 columns zeroed 0.101)
MAX_COEF_DEV = 0.002  # per addend |coef - 1| (bf16 output 1e-5; 1.01 x experts moves the shared coef by 0.0047)
MAX_ADD_REL = {"shared_out": 0.008, "experts_out": 0.005}  # ||delta - t|| / ||t|| (bf16 0.0040 / 0.0023)
MAX_ADD_ROW_REL = 0.03  # per token, per addend (bf16 0.014 / 0.015, golden 0.019 / 0.020; last row zeroed 2.4 / 1.3)
PROBE_MAX_REL = 0.004  # probe calls vs the exact sum of their own inputs (bf16 output 0.0017)
PROBE_MAX_ROW_REL = 0.005  # bf16 output 0.0018


def _rows(t: torch.Tensor, n: int) -> torch.Tensor:
    return t.float().reshape(n, -1)


def _row_rel(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)


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
    ), "device_component returned a CPU bridge; moe_combine is not on the device"
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Scale and per-row checks vs golden (PCC is scale-invariant and barely sees a few bad rows).
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    n = want.shape[0]
    got = _rows(out, n)
    w = _rows(want, n)
    assert torch.isfinite(got).all(), "non-finite output"
    rel = ((got - w).norm() / w.norm()).item()
    ratio = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row = _row_rel(got, w)
    worst, worst_at = row.max().item(), int(row.argmax().item())
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"worst_row_rel_{STEP}_L{LAYER:02d}", worst)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.5f}, {rmax:.5f}] (in {ROW_NORM_RATIO}) "
        f"worst_row_rel={worst:.5f} (<= {MAX_ROW_REL}) at row {worst_at}"
    )
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.5f} > {MAX_REL_L2} (scale bug, dropped or zeroed rows/columns)"
    assert (
        ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]
    ), f"per-token norm ratio [{rmin:.5f}, {rmax:.5f}] outside {ROW_NORM_RATIO} (zeroed or scaled rows?)"
    assert worst <= MAX_ROW_REL, f"worst row rel {worst:.5f} > {MAX_ROW_REL} at row {worst_at} (a bad row?)"

    # Each addend on its own: delta = out - other vs t (float64; fp32 sums over 12.6M elements drift).
    named = dict(zip(st.inputs, (_rows(x, n).double() for x in inputs)))
    g64 = got.double()
    for name, t in named.items():
        other = sum(v for k, v in named.items() if k != name)
        err = (g64 - other) - t
        tn = t.norm()
        coef = (((g64 - other) * t).sum() / (tn * tn)).item()
        arel = (err.norm() / tn).item()
        arow = (err.norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-30)).max().item()
        tag = name.replace("_out", "")
        metrics.record(f"add_coef_{tag}_{STEP}_L{LAYER:02d}", coef)
        metrics.record(f"add_rel_l2_{tag}_{STEP}_L{LAYER:02d}", arel)
        metrics.record(f"add_worst_row_{tag}_{STEP}_L{LAYER:02d}", arow)
        print(
            f"addend {name}: coef={coef:.6f} (|coef-1| <= {MAX_COEF_DEV}) rel={arel:.5f} (<= {MAX_ADD_REL[name]}) "
            f"worst row={arow:.4f} (<= {MAX_ADD_ROW_REL})"
        )
        assert abs(coef - 1) <= MAX_COEF_DEV, f"{name} enters mlp_out with coefficient {coef:.5f} (scaled addend?)"
        assert arel <= MAX_ADD_REL[name], f"{name} addend error {arel:.5f} > {MAX_ADD_REL[name]} (misaligned?)"
        assert arow <= MAX_ADD_ROW_REL, f"{name} worst row addend error {arow:.4f} > {MAX_ADD_ROW_REL} (a bad row?)"

    # Probes: the module must add its own inputs (compared with the exact fp32 sum of the probe inputs).
    experts, shared = inputs
    for label, a, b in (("experts - shared", experts, -shared), ("experts + 0", experts, torch.zeros_like(shared))):
        exact = _rows(a, n) + _rows(b, n)
        p = fn(rctx, dctx, a, b)
        assert p.numel() == exact.numel(), f"probe {label}: output has {p.numel()} elements"
        pg = _rows(p, n)
        assert torch.isfinite(pg).all(), f"probe {label}: non-finite output"
        prel = ((pg - exact).norm() / exact.norm()).item()
        prow = _row_rel(pg, exact).max().item()
        tag = label.replace(" ", "").replace("-", "minus").replace("+", "plus")
        metrics.record(f"probe_rel_l2_{tag}_{STEP}_L{LAYER:02d}", prel)
        print(f"probe {label}: rel={prel:.6f} (<= {PROBE_MAX_REL}) worst row={prow:.5f} (<= {PROBE_MAX_ROW_REL})")
        assert prel <= PROBE_MAX_REL, f"probe {label}: rel {prel:.5f} > {PROBE_MAX_REL} (module ignores its inputs?)"
        assert prow <= PROBE_MAX_ROW_REL, f"probe {label}: worst row {prow:.5f} > {PROBE_MAX_ROW_REL}"

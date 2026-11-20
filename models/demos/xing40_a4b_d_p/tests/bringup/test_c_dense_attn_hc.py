# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_hc of block type dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dense.attn_hc.test.1). The output is [S, 24] fp32 = [pre 4 | post 4 | comb 16, row-major 4x4]. Its three
parts have very different scales on this golden (s4096 chunk 1, 2048 rows): pre 0.04..0.50, post 3e-8..0.18 (mean
0.013), comb 1e-8..1.0. A whole-output check is dominated by comb and misses a precision loss in post. The built-in
checks (checks="auto") let 1 % output noise (mutate noise1e-2) through: their calibrated rel limit is 0.0148, set by the
bf16-everywhere precision model (comb logits up to ~30 rounded to bf16 give comb rel 0.0077), and that noise is only
rel 0.007 on the whole output, but post rel 0.092 and negative post / comb values.

Measured, rel L2 per part vs the fp32 CPU step (pre / post / comb), and the worst |column sum - 1| of comb:
    fp32 step vs the bf16 golden     0.0014 / 0.0017 / 0.0014   colsum 0.0029
    tf32 or bf16 projection operands 0.0002 / 0.0009 / 0.0001   (a device on the fp32 plan)
    projection error x (1 + 3e-3 N)  0.0018 / 0.0077 / 0.0061
    bf16 everywhere (precision model) 0.0024 / 0.0118 / 0.0077
    noise1e-2                        0.0114 / 0.0916 / 0.0070   colsum 0.022, post min -0.011
    RMS eps 0 / 1e-5 (vs 1e-6)       0.0124 / 0.0200 / 0.0039;  0.0925 / 0.2108 / 0.0241
    post = sigmoid (no x 2)          post 0.50;  pre / post scales swapped: pre 0.44
    19 Sinkhorn iterations           comb 0.0064 (not caught; PCC 0.99997); 10 iterations comb 0.17
    comb transposed / row step last  comb 0.13 / 0.048, colsum 0.11 / 0.097 (20 iterations do not converge: rows
                                     sum to 1 +- 0.11, columns to 1 exactly, since the column step is last)
Not visible on real inputs: the [-30, 30] clamp and the row-max subtraction (logits stay inside), hc_eps 0 or 1e-5, and
stream order at layer 0 (the 4 streams are equal; the auto "mixed" and "layer39" inputs cover stream order).
Extra checks on the golden input: rel L2 per part vs the CPU step (PART_REL) and vs the golden (PART_REL + the fp32
step's own error there), comb column sums within COL_SUM_TOL of 1, and the ranges pre in (0, 1], post in [0, 2],
comb in [0, 1] (RANGE_TOL slack). BRINGUP_IMPL=mutations runs the auto sweep (printed) and then this test's own sweep:
every standard mistake must fail the auto checks or the extra checks, the controls must pass both.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.component import _step, module_under_test, run_component_test
from models.demos.common.bringup.testing.harness import (
    component_golden,
    device_ctx,
    impl_mode,
    mesh_parametrize,
    reference_ctx,
    spec,
)

S = spec()
STEP = "attn_hc"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
N = 4  # hc_mult
PARTS = {"pre": slice(0, N), "post": slice(N, 2 * N), "comb": slice(2 * N, 2 * N + N * N)}
PART_REL = {"pre": 0.006, "post": 0.03, "comb": 0.012}  # rel L2 per part vs the CPU step on the same input
COL_SUM_TOL = 0.01  # |sum_i comb[i, j] - 1| (the last Sinkhorn step normalizes columns)
RANGE_TOL = 1e-3  # slack on the value ranges


def _rel(a, b) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def part_fails(got, cpu, want, record=False) -> list[str]:
    """The extra checks of one output on the golden input: vs the CPU step ``cpu`` and vs the golden ``want``."""
    if got.numel() != want.numel():
        return [f"{got.numel()} elements, want {tuple(want.shape)}"]
    got = got.float().reshape(want.shape)
    if not bool(torch.isfinite(got).all()):
        return ["non-finite output"]
    fails = []
    for name, sl in PARTS.items():
        a, c, w = got[:, sl], cpu[:, sl], want[:, sl]
        r_cpu, r_gold, own = _rel(a, c), _rel(a, w), _rel(c, w)
        lim_gold = PART_REL[name] + own
        if record:
            metrics.record(f"rel_{STEP}_{name}_vs_cpu_L{LAYER:02d}", r_cpu)
            metrics.record(f"rel_{STEP}_{name}_vs_golden_L{LAYER:02d}", r_gold)
            print(
                f"{name}: rel vs cpu {r_cpu:.5f} (<= {PART_REL[name]}) vs golden {r_gold:.5f} (<= {lim_gold:.5f}) "
                f"max abs vs cpu {(a - c).abs().max().item():.2e}"
            )
        if not r_cpu <= PART_REL[name]:
            fails.append(f"{name} rel vs cpu {r_cpu:.5f} > {PART_REL[name]}")
        if not r_gold <= lim_gold:
            fails.append(f"{name} rel vs golden {r_gold:.5f} > {lim_gold:.5f}")
    pre, post = got[:, PARTS["pre"]], got[:, PARTS["post"]]
    comb = got[:, PARTS["comb"]].reshape(-1, N, N)
    col_err = (comb.sum(dim=-2) - 1).abs().max().item()
    if record:
        metrics.record(f"comb_col_sum_err_{STEP}_L{LAYER:02d}", col_err)
        print(f"comb column sums: worst |. - 1| {col_err:.5f} (<= {COL_SUM_TOL})")
    if not col_err <= COL_SUM_TOL:
        fails.append(f"comb column sums off 1 by {col_err:.4f} (transposed comb, row step last, or precision)")
    for name, t, lo, hi in (("pre", pre, 0.0, 1.0), ("post", post, 0.0, 2.0), ("comb", comb, 0.0, 1.0)):
        if not (t.min().item() >= lo - RANGE_TOL and t.max().item() <= hi + RANGE_TOL):
            fails.append(f"{name} outside [{lo}, {hi}]: [{t.min().item():.3e}, {t.max().item():.4f}]")
    if not pre.min().item() > 0:
        fails.append(f"pre not positive (sigmoid): min {pre.min().item():.3e}")
    return fails


def _golden():
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    cpu_step = ref.component(LAYER, STEP)
    cpu = cpu_step(reference_ctx(ref, LAYER, g, c), *[t.clone() for t in inputs]).float()
    return g, c, ref, st, inputs, cpu_step, cpu, gl[st.output].float()


def _sweep() -> bool:
    """This test's mistake sweep (CPU): every float mutation fails the auto checks or the extra checks; the reference
    and the bf16 controls pass both."""
    from models.demos.common.bringup.testing.component import _EXPECT

    run_component_test(S, STEP, LAYER, None, COMPARE, THRESHOLD, checks=CHECKS)  # the auto sweep's table, printed
    g, c, ref, st, inputs, cpu_step, cpu, want = _golden()
    ex = next(e for k, e in _EXPECT.items() if k[:3] == (LAYER, STEP, c))
    rows, ok = [], True
    for k in MU.FLOAT_KINDS:
        outs = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        auto = ex.evaluate(outs, quiet=True) + (
            [] if CC.golden_gate_quiet(outs["golden"], want, ex.kind, ex.thr) else ["gate"]
        )
        extra = part_fails(outs["golden"], cpu, want)
        caught = bool(auto or extra)
        ok &= caught
        rows.append((k, "caught" if caught else "SLIPPED", f"auto: {len(auto)} checks; extra: {'; '.join(extra)}"))
    controls = {"reference": ex.cpu, "bf16 everywhere": ex.model}
    controls["bf16 inputs + output"] = {
        x.name: MU.bf16(CC.run_cpu(ex.cpu_step, x, [MU.bf16(t) for t in x.inputs])) for x in ex.cases
    }
    for name, outs in controls.items():
        f = ex.evaluate(outs, quiet=True) + part_fails(outs["golden"], cpu, want)
        ok &= not f
        rows.append((name, "pass" if not f else "FAIL", "; ".join(f)))
    print(f"{'ok  ' if ok else 'FAIL'} test sweep {STEP} L{LAYER} (auto + extra checks):")
    for r in rows:
        print(f"    {r[0]:<22} {r[1]:<8} {r[2][:200]}")
    return ok


@mesh_parametrize
def test_component(mesh_device):
    if impl_mode() == "mutations":
        assert _sweep(), "a standard mistake slipped through the auto and extra checks"
        return
    ok = run_component_test(S, STEP, LAYER, mesh_device, COMPARE, THRESHOLD, checks=CHECKS)
    g, c, ref, st, inputs, cpu_step, cpu, want = _golden()
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)
    fails = part_fails(out, cpu, want, record=True)
    for f in fails:
        print(f"FAIL extra {STEP} L{LAYER}: {f}")
    assert ok, "built-in checks or the PCC gate failed (see the log above)"
    assert not fails, "; ".join(fails)

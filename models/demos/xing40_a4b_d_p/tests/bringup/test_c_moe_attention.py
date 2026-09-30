# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type moe (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe.attention.test.1). The output is attn_out [2048, 3584] fp32 (s4096 chunk 1, 2048-row latent prefix).
The built-in checks (checks="auto") let 1 % output noise (mutate noise1e-2, rel 0.0100) through: their calibrated rel
limit on the golden input is 0.0135 (2 x the bf16-everywhere precision model's rel 0.0068). The limit cannot be
tightened, because a correct device is already above the noise: the current module (ring_mla fork, HiFi4, fp32 dest)
measured rel 0.0108 vs the CPU step on this golden (layer 0: 0.0031). No size limit separates them.

What does separate them is where the error lies. A correct device's error goes through the same values (latent cache
x W_uv) and o_proj as the output, so most of it lies in the output's dominant row subspace. White noise is spread over
all 3584 columns. Measured, rel L2 of the error outside the top K = 1024 right singular vectors of the CPU output
(which hold 0.9993 of its energy), vs the CPU output norm (golden / chunk0 / layer39 inputs):
    device (ring_mla fork, fp32 dest)  0.0027 / 0.0027 / 0.0024     (whole rel 0.0108 / 0.0104 / 0.0085)
    bf16 everywhere (precision model)  0.0021 / 0.0023 / 0.0020     (whole rel 0.0068 / 0.0067 / 0.0075)
    noise1e-2                          0.0085 / 0.0085 / 0.0085     (whole rel 0.0100)
Extra check (OFF_REL): that off-subspace rel L2 <= 0.006 vs the CPU step on the golden, chunk0 and layer39 inputs.
BRINGUP_IMPL=mutations runs the auto sweep (printed; it reports noise1e-2 SLIPPED) and then this test's own sweep:
every standard mistake must fail the auto checks or the extra check, the controls must pass both.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.component import _EXPECT, module_under_test, run_component_test
from models.demos.common.bringup.testing.harness import impl_mode, mesh_parametrize, spec

S = spec()
STEP = "attention"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
OFF_K = 1024  # dominant right singular vectors of the CPU output kept as its subspace
OFF_REL = 0.006  # error outside that subspace, rel to the CPU output norm (device 0.0027, noise 0.0085)
OFF_CASES = ("golden", "chunk0", "layer39")


def _expect():
    return next(e for k, e in _EXPECT.items() if k[:2] == (LAYER, STEP))


def _basis(ex, name):
    """The top-OFF_K right singular vectors of the CPU output on input ``name``, cached on the Expect object (the
    precompile collect pass builds its own)."""
    cache = ex.__dict__.setdefault("_off_basis", {})
    if name not in cache:
        y = ex.cpu[name].float().reshape(-1, ex.cpu[name].shape[-1])
        cache[name] = torch.linalg.svd(y, full_matrices=False).Vh[:OFF_K].T.contiguous()
    return cache[name]


def off_fails(ex, outs, record=False) -> list[str]:
    """The extra check: the error vs the CPU step outside the CPU output's top-OFF_K row subspace."""
    fails = []
    for name in OFF_CASES:
        if name not in ex.cpu or name not in outs:
            continue
        got, y = outs[name], ex.cpu[name].float()
        if isinstance(got, Exception):
            fails.append(f"{name}: the module raised {type(got).__name__}")
            continue
        if got.numel() != y.numel():
            fails.append(f"{name}: {got.numel()} elements, want {tuple(y.shape)}")
            continue
        y = y.reshape(-1, y.shape[-1])
        e = got.float().reshape(y.shape) - y
        if not bool(torch.isfinite(e).all()):
            fails.append(f"{name}: non-finite output")
            continue
        V = _basis(ex, name)
        off = ((e - (e @ V) @ V.T).norm() / y.norm().clamp_min(1e-30)).item()
        if record:
            metrics.record(f"off_subspace_rel_{STEP}_{name}_L{LAYER:02d}", off)
            print(
                f"{'ok  ' if off <= OFF_REL else 'FAIL'} off-subspace (K {OFF_K}) rel {name}: {off:.5f} (<= {OFF_REL})"
            )
        if not off <= OFF_REL:
            fails.append(f"{name}: error off the output subspace rel {off:.5f} > {OFF_REL} (white precision loss)")
    return fails


def _sweep() -> bool:
    """This test's mistake sweep (CPU): every float mutation fails the auto checks or the extra check; the reference
    and the bf16 controls pass both."""
    run_component_test(S, STEP, LAYER, None, COMPARE, THRESHOLD, checks=CHECKS)  # the auto sweep's table, printed
    ex = _expect()
    rows, ok = [], True
    for k in MU.FLOAT_KINDS:
        outs = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        auto = ex.evaluate(outs, quiet=True) + (
            [] if CC.golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr) else ["gate"]
        )
        extra = off_fails(ex, outs)
        caught = bool(auto or extra)
        ok &= caught
        rows.append((k, "caught" if caught else "SLIPPED", f"auto: {len(auto)} checks; extra: {'; '.join(extra)}"))
    controls = {"reference": ex.cpu, "bf16 everywhere": ex.model}
    controls["bf16 inputs + output"] = {
        x.name: MU.bf16(CC.run_cpu(ex.cpu_step, x, [MU.bf16(t) for t in x.inputs])) for x in ex.cases
    }
    for name, outs in controls.items():
        f = ex.evaluate(outs, quiet=True) + off_fails(ex, outs)
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
    ex = _expect()
    fn = module_under_test(S, ex.ref, mesh_device, LAYER, STEP)
    outs = {}
    for case in ex.cases:
        if case.name in OFF_CASES:
            outs[case.name] = fn(case.rctx(), case.dctx(), *CC._clone(case.inputs))
    fails = off_fails(ex, outs, record=True)
    for f in fails:
        print(f"FAIL extra {STEP} L{LAYER}: {f}")
    assert ok, "built-in checks or the PCC gate failed (see the log above)"
    assert not fails, "; ".join(fails)

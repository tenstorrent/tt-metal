# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: shared_expert of block type swa (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.swa.shared_expert.test.1): shared_out = w2 q(silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10)) [S, 5120],
q = fp8 act quant-dequant (blocks of 32, ue8m0), w1 / w3 [2304, 5120], w2 [5120, 2304] (fp8 x ue8m0 blocks, bf16-exact).
The built-in checks (CHECKS = "auto") calibrate rel L2 on the bf16 precision model, whose fp8 re-roundings of x and h
put it near 1 % (DS41-14, DS41-22): 1 % output noise (noise1e-2) slipped them (K-36). One structural check is added,
on every input of the built-in checks (golden, layer24, mixed, small, big):
    w2 span     the last op is the w2 GEMM, so every output row is a combination of w2's 2304 columns, a subspace of
                R^5120. The part of the output outside it, ||out - P out|| / ||out|| (P the orthogonal projector on
                w2's column space, float64), is only the output's own rounding: a bf16 output gives ~2^-9 x
                sqrt(2816 / 5120); 1 % noise of the output's std puts ~0.74 x 1 % outside the span. Limit SPAN_REL.
                The measured values are in the BREADCRUMBS entry of this task.
Built-in checks plus this one catch every standard mistake of testing/mutate.py. BRINGUP_IMPL=mutations runs the
sweep with this check added (the built-in sweep alone reports noise1e-2 as slipped).
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component as C
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import model_precision
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import component_golden, impl_mode, mesh_parametrize, spec, threshold

S = spec()
STEP = "shared_expert"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
W2 = "ffn.shared_experts.w2.weight"  # [5120, 2304]: the output lies in its column space
SPAN_REL = 0.003  # ||out - P out|| / ||out||, P the projector on w2's columns (bf16 output ~0.001; 1 % noise ~0.007)

_BASIS = {}


def _expect():
    """The built-in checks' Expect (CPU outputs per input), reused from run_component_test when it built one."""
    g, c = component_golden(S)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    ex = C._EXPECT.get((LAYER, STEP, c, COMPARE, thr))
    if ex is None:
        ref = S.hooks().reference(S, layers=[*S.state_sources(LAYER), LAYER], dtype=torch.float32)
        st = C._step(ref, LAYER, STEP)
        want = g.layer(c, LAYER)[st.output].float()
        ex = C._EXPECT[(LAYER, STEP, c, COMPARE, thr)] = CC.Expect(S, ref, LAYER, st, g, c, want, thr, COMPARE)
    return ex


def _basis(ex) -> torch.Tensor:
    """An orthonormal basis [5120, 2304] (float64) of the column space of layer LAYER's shared w2."""
    if LAYER not in _BASIS:
        w2 = ex.ref.w[LAYER][W2].double()
        q, r = torch.linalg.qr(w2)
        assert bool((r.diagonal().abs() > 1e-9 * r.diagonal().abs().max()).all()), "w2 is rank deficient"
        _BASIS[LAYER] = q
    return _BASIS[LAYER]


def _span_error(got: torch.Tensor, q: torch.Tensor) -> float | None:
    if got.shape[-1] != q.shape[0]:
        return None
    t = got.double().reshape(-1, q.shape[0])
    if not bool(torch.isfinite(t).all()):
        return None
    r = t - (t @ q) @ q.T
    return (r.norm() / t.norm().clamp_min(1e-300)).item()


def _span_fails(outs: dict, ex, quiet: bool) -> list[str]:
    """The w2-span check on every input; the names of the failed checks."""
    q = _basis(ex)
    failed = []
    for case in ex.cases:
        got, tag = outs[case.name], f"span_{STEP}_L{LAYER:02d}_{case.name}"
        if isinstance(got, Exception):
            failed.append(tag)
            continue
        e = _span_error(got, q)
        bad = None
        if e is None:
            bad = f"shape {tuple(got.shape)} or non-finite values, want {tuple(ex.cpu[case.name].shape)}"
        elif not e <= SPAN_REL:
            bad = f"{e:.5f} of the output outside w2's column space (> {SPAN_REL})"
        if bad:
            failed.append(tag)
        if not quiet:
            if e is not None:
                metrics.record(tag, e)
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: rel_outside={e}" + (f" ({bad})" if bad else ""))
    return failed


def _sweep(mesh_device) -> bool:
    """BRINGUP_IMPL=mutations: the built-in sweep's mistakes, each caught by the built-in checks or the span check."""
    model_precision.apply(S)
    ex = _expect()
    gate = f"pcc_{STEP}_L{LAYER:02d}"
    q = _basis(ex)

    def verdict(outs, stop_early):
        f = ex.evaluate(outs, quiet=True, stop_early=stop_early)
        f += [] if CC.golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr, ex.mode) else [gate]
        return f + _span_fails(outs, ex, quiet=True)

    def spans(outs):
        return " ".join(f"{n}={_span_error(o, q):.5f}" for n, o in outs.items())

    rows, ok = [], True
    for name, outs in (
        ("reference", ex.cpu),
        (
            "bf16 inputs + output",
            {x.name: MU.bf16(CC.run_cpu(ex.cpu_step, x, [MU.bf16(t) for t in x.inputs])) for x in ex.cases},
        ),
        ("bf16 everywhere", ex.model),
    ):
        f = verdict(outs, False)
        rows.append((name, "pass" if not f else "FAIL", ", ".join(f) or "span " + spans(outs)))
        ok &= not f
    for k in [k for k in MU.KINDS if MU.applies(k, ex.want)]:
        mut = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        f = verdict(mut, False)
        rows.append((k, "caught" if f else "SLIPPED", ", ".join(f) + " | span " + spans(mut)))
        ok &= bool(f)
    print(f"golden output (bf16) outside w2's column space: {_span_error(ex.want, q):.5f}")
    print(f"{'ok  ' if ok else 'FAIL'} sweep {STEP} L{LAYER} (built-in + span checks):")
    for r in rows:
        print(f"    {r[0]:<24} {r[1]:<8} {r[2][:400]}")
    return ok


@mesh_parametrize
def test_component(mesh_device):
    if impl_mode() == "mutations":
        assert _sweep(mesh_device)
        return
    ok = run_component_test(S, STEP, LAYER, mesh_device, COMPARE, THRESHOLD, checks=CHECKS)
    ex = _expect()
    fn = C.module_under_test(S, ex.ref, mesh_device, LAYER, STEP)
    outs = {}
    for case in ex.cases:
        try:
            outs[case.name] = fn(case.rctx(), case.dctx(), *CC._clone(case.inputs))
        except Exception as e:  # noqa: BLE001 - a module that fails on a second input fails that check
            outs[case.name] = e
    fails = _span_fails(outs, ex, quiet=False)
    assert ok and not fails, f"w2 span checks failed: {fails}" if fails else "built-in checks failed"

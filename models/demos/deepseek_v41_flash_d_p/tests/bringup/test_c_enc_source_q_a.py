# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: q_a of block type enc_source (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.enc_source.q_a.test.1): q_a = q_norm(fp8_linear(attn_norm, wq_a)) [S, 1280] (golden chunk, bf16), q_norm
the official RMSNorm (fp32 rsqrt(mean(x^2) + eps), times weight, out bf16; hf/inference/model.py RMSNorm). The
built-in checks (CHECKS = "auto") calibrate rel L2 on the input nudge, whose fp8 re-quantization of attn_norm flips
whole e4m3 steps (DS41-1): rel limits 0.0158 (golden, small, big), 0.0292 (layer24), 0.0325 (mixed). 1 % output noise
(rel 0.010) slipped them (K-36, DS41-14). One structural check of the normalized output is added, on every input of the
built-in checks (golden, layer24, mixed, small, big):
    norm energy  every output row divided by q_norm.weight has RMS exactly 1 (eps 1e-20): the norm re-normalizes any
                 error made before it (the fp8 flips, the GEMM's summation order), so only errors made after the norm
                 move a row's energy. Statistic: bias = mean over rows of (rms(out_row / w) - 1), |bias| <=
                 NORM_BIAS. Measured (layer 2, scaled checkpoint): the CPU step and the input nudge 2e-9; the bf16
                 output (the official output dtype) |bias| <= 3.6e-6 (row rms 0.8e-4 .. 1.2e-4, unbiased);
                 noise1e-2 +5.5e-5 .. +7.7e-5 on every input (noise adds energy: mean((e / w)^2) / 2 > 0);
                 noise1e-3 <= 2.1e-6 (below the bf16 floor, not a mistake); scale 1.02: 0.02. NORM_BIAS 1.5e-5 is
                 ~4x the bf16 control and ~3.7x below 1 % noise. A zero output is -1.
Built-in checks plus this one catch every standard mistake of testing/mutate.py. BRINGUP_IMPL=mutations runs the
built-in sweep's controls and mistakes with this check added (the built-in sweep alone reports noise1e-2 as slipped).
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component as C
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import model_precision
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing import noise_floor as NF
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import component_golden, impl_mode, mesh_parametrize, spec, threshold

S = spec()
STEP = "q_a"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
NORM_W = "attn.q_norm.weight"  # [1280]: the output is n * w with rms(n) = 1 per row
NORM_BIAS = 1.5e-5  # |mean over rows of (rms(out / w) - 1)|: bf16 output <= 3.6e-6, 1 % noise >= 5.5e-5


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


def _norm_bias(got: torch.Tensor, cpu: torch.Tensor, w: torch.Tensor) -> float | None:
    """mean over the rows of (rms(got_row / w) - 1), on the rows whose CPU output is nonzero; None if the shape or
    the values are wrong."""
    if got.numel() != cpu.numel() or cpu.shape[-1] != w.numel():
        return None
    t = got.double().reshape(-1, w.numel())
    if not bool(torch.isfinite(t).all()):
        return None
    live = cpu.double().reshape(t.shape).abs().amax(-1) > 0
    if not bool(live.any()):
        return None
    r = (t[live] / w.double()).pow(2).mean(-1).sqrt() - 1
    return r.mean().item()


def _norm_fails(outs: dict, ex, quiet: bool) -> list[str]:
    """The norm-energy check on every input; the names of the failed checks."""
    w = ex.ref.w[LAYER][NORM_W].float()
    failed = []
    for case in ex.cases:
        got, tag = outs[case.name], f"norm_bias_{STEP}_L{LAYER:02d}_{case.name}"
        if isinstance(got, Exception):
            failed.append(tag)
            continue
        b = _norm_bias(got, ex.cpu[case.name], w)
        bad = None
        if b is None:
            bad = f"shape {tuple(got.shape)} or non-finite values, want {tuple(ex.cpu[case.name].shape)}"
        elif not abs(b) <= NORM_BIAS:
            bad = f"rows of out / {NORM_W} have mean rms 1 {b:+.3g} (|bias| > {NORM_BIAS})"
        if bad:
            failed.append(tag)
        if not quiet:
            if b is not None:
                metrics.record(tag, b)
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: bias={b}" + (f" ({bad})" if bad else ""))
    return failed


def _sweep(mesh_device) -> bool:
    """BRINGUP_IMPL=mutations: the built-in sweep (CC.sweep: controls, mistakes) with the norm-energy check added."""
    model_precision.apply(S)
    ex = _expect()
    err_name = f"err_{STEP}_L{LAYER:02d}"
    w = ex.ref.w[LAYER][NORM_W].float()

    def verdict(outs, stop_early):
        f = ex.evaluate(outs, quiet=True, stop_early=stop_early)
        ok = CC.error_gate(err_name, outs["golden"], ex.want, ex, quiet=True)
        if ex.thr_literal:
            ok = CC.golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr, ex.mode) and ok
        return f + ([] if ok else [err_name]) + _norm_fails(outs, ex, quiet=True)

    def biases(outs):
        return " ".join(f"{n}={_norm_bias(o, ex.cpu[n], w):+.2e}" for n, o in outs.items())

    nudged = {
        x.name: CC.run_cpu(
            ex.cpu_step, x, [NF.nudge(t, 99991 + j, d) for j, (t, d) in enumerate(zip(x.inputs, ex.dtypes))]
        )
        for x in ex.cases
    }
    rows, ok = [], True
    for name, outs in (
        ("reference", ex.cpu),
        ("input nudge (new seed)", nudged),
        ("bf16 output", {n: MU.bf16(o) for n, o in ex.cpu.items()}),
    ):
        f = verdict(outs, False)
        rows.append((name, "pass" if not f else "FAIL", ", ".join(f) + " | bias " + biases(outs)))
        ok &= not f
    for k in [k for k in MU.KINDS if MU.applies(k, ex.want)]:
        mut = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        f = verdict(mut, False)
        rows.append((k, "caught" if f else "SLIPPED", ", ".join(f) + " | bias " + biases(mut)))
        ok &= bool(f)
    print(f"{'ok  ' if ok else 'FAIL'} sweep {STEP} L{LAYER} (built-in + norm-energy checks):")
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
    fails = _norm_fails(outs, ex, quiet=False)
    assert ok and not fails, f"norm-energy checks failed: {fails}" if fails else "built-in checks failed"

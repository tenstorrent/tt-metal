# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: kv of block type enc_source (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.enc_source.kv.test.1): the same step as swa layer 0 (ds41_ref.py:_window_kv; layer 2 only rotates with the
ratio-layer RoPE table), so the same checks as test_c_swa_kv.py (DS41-14).
kv = act_quant_dequant(rope(kv_norm(fp8_linear(attn_norm, wkv)), start..), 32) [S, 512] (golden chunk, bf16). The
output is fp8 e4m3 x a power-of-two (ue8m0) scale per block of 32 columns, so it lies on a grid that bf16 holds
exactly. The built-in checks (CHECKS = "auto") calibrate rel L2 on the input nudge, whose fp8 re-quantization of
attn_norm flips whole e4m3 steps (DS41-1): rel limits 0.0464 (golden, small, big), 0.0589 (layer24), 0.0619 (mixed).
The freeze sweep let scale 1.02, scale 0.98 and 1 % output noise (rel 0.010-0.020) through them (attempt 1, K-36).
Two structural checks of the quantized output are added, on every input of the built-in checks (golden, layer24,
mixed, small, big), each against the CPU step on the same inputs (layer 2, scaled checkpoint, measured on the CPU):
    fp8 grid    every value equals act_quant_dequant(value, 32): no value off the e4m3 x ue8m0 grid (exactly 0).
                Reference, input nudge, bf16 output: 0. scale 1.02 / 0.98, noise1e-2: 1.0 off grid.
    exact match the fraction of values that differ from the CPU step <= FP8_MISMATCH = 2 x the input nudge's worst.
                The input nudge (attn_norm moved inside its bf16 cell, 3 seeds) flips whole e4m3 steps of the fp8
                re-quantized input (DS41-1): 0.260 (golden, small, big), 0.298 (layer24), 0.315 (mixed): layer 0's fixed
                0.15 (DS41-14, from the bf16 precision model) would fail this control. GEMM output rounded to bf16 0.018;
                rope one position late 0.038 (caught by the auto checks); e4m3 rounded toward zero 0.49 (caught by
                the auto vs-CPU checks on every input but the golden-vs-golden one); attn_norm not fp8-quantized
                before the GEMM 0.36-0.43 (inside 2 x the nudge and the auto limits: not separable at this floor).
                halfswap 0.99, rowshift 0.98, quarterzero 0.25. Devices at layers 0 / 1: 0.006-0.011.
Built-in checks plus these catch every standard mistake of testing/mutate.py. BRINGUP_IMPL=mutations runs the built-in
sweep's controls and mistakes with these checks added (the built-in sweep alone reports scale and noise as slipped).
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
from models.demos.deepseek_v41_flash_d_p.reference import kernels as K

S = spec()
STEP = "kv"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
FP8_BLOCK = 32  # act_quant block (columns) of the kv output, ue8m0 scale per block
FP8_OFFGRID = 0.0  # fraction of values off the e4m3 x ue8m0 grid
FP8_MISMATCH = 0.63  # fraction of values != the CPU step on the same inputs: 2 x the input nudge's worst (0.315)


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


def _fp8_errors(got: torch.Tensor, cpu: torch.Tensor) -> dict | None:
    if got.numel() != cpu.numel():
        return None
    t = got.float().reshape(-1, cpu.shape[-1])
    if not bool(torch.isfinite(t).all()):
        return None
    return {
        "offgrid": (K.act_quant_dequant(t, FP8_BLOCK) != t).float().mean().item(),
        "mismatch": (t != cpu.float().reshape(t.shape)).float().mean().item(),
    }


def _fp8_fails(outs: dict, ex, quiet: bool) -> list[str]:
    """The structural checks of the fp8 output on every input; the names of the failed checks."""
    failed = []
    for case in ex.cases:
        got, tag = outs[case.name], f"fp8_{STEP}_L{LAYER:02d}_{case.name}"
        if isinstance(got, Exception):
            failed.append(tag)
            continue
        e = _fp8_errors(got, ex.cpu[case.name])
        bad = []
        if e is None:
            bad.append(f"shape {tuple(got.shape)} or non-finite values, want {tuple(ex.cpu[case.name].shape)}")
        else:
            if not e["offgrid"] <= FP8_OFFGRID:
                bad.append(f"{e['offgrid']:.4f} of the values off the e4m3 x ue8m0 grid (blocks of {FP8_BLOCK})")
            if not e["mismatch"] <= FP8_MISMATCH:
                bad.append(f"{e['mismatch']:.4f} of the values differ from the CPU step (> {FP8_MISMATCH})")
        if bad:
            failed.append(tag)
        if not quiet:
            for k, v in (e or {}).items():
                metrics.record(f"{tag}_{k}", v)
            nums = " ".join(f"{k}={v:.6g}" for k, v in (e or {}).items())
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: {nums}" + (f" ({'; '.join(bad)})" if bad else ""))
    return failed


def _sweep(mesh_device) -> bool:
    """BRINGUP_IMPL=mutations: the built-in sweep (CC.sweep: controls, mistakes) with the fp8 checks added."""
    model_precision.apply(S)
    ex = _expect()
    err_name = f"err_{STEP}_L{LAYER:02d}"

    def verdict(outs, stop_early):
        f = ex.evaluate(outs, quiet=True, stop_early=stop_early)
        ok = CC.error_gate(err_name, outs["golden"], ex.want, ex, quiet=True)
        if ex.thr_literal:
            ok = CC.golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr, ex.mode) and ok
        return f + ([] if ok else [err_name]) + _fp8_fails(outs, ex, quiet=True)

    def stats(outs):
        e = {n: _fp8_errors(o, ex.cpu[n]) for n, o in outs.items()}
        off = max((v or {"offgrid": 1.0})["offgrid"] for v in e.values())
        mis = [(v or {"mismatch": 1.0})["mismatch"] for v in e.values()]
        return f"offgrid max {off:.4f} mismatch {min(mis):.4f}..{max(mis):.4f}"

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
        rows.append((name, "pass" if not f else "FAIL", ", ".join(f) + " | " + stats(outs)))
        ok &= not f
    for k in [k for k in MU.KINDS if MU.applies(k, ex.want)]:
        mut = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        f = verdict(mut, False)
        rows.append((k, "caught" if f else "SLIPPED", ", ".join(f) + " | " + stats(mut)))
        ok &= bool(f)
    print(f"{'ok  ' if ok else 'FAIL'} sweep {STEP} L{LAYER} (built-in + fp8 checks):")
    for r in rows:
        print(f"    {r[0]:<24} {r[1]:<8} {r[2][-400:]}")
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
    fails = _fp8_fails(outs, ex, quiet=False)
    assert ok and not fails, f"fp8 output checks failed: {fails}" if fails else "built-in checks failed"

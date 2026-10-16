# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: compressor of block type enc_source (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.enc_source.compressor.test.1): ds41_ref.py:_compressor, ratio 2. Output [S / 2, 512 + 128] =
cat(ckv, k): ckv = fp4_quant_dequant(rope(latent), 16, e4m3 scale) and k = fp4_quant_dequant(rope(k_norm(latent @
wk)), 32, ue8m0 scale), latent = norm(sum over each group of 2 tokens of kv x softmax(gate)). Both halves are e2m1
values x a per-block scale, a grid bf16 holds exactly. The built-in checks (CHECKS = "auto") calibrate rel L2 on the
input nudge (and, for the golden, on the official code's arithmetic, DS41-38): rel limits 0.0746 (golden), 0.0309
(chunk0), 0.0401 (layer24), 0.0425 (mixed), 0.0309 (small), 0.0318 (big). The freeze sweep let scale 1.02 and
scale 0.98 (rel 0.020) through them (attempt 0, K-33 / K-36). As for the fp8 kv (DS41-14, DS41-46), two
structural checks of the quantized output are added, on every input of the built-in checks (golden, chunk0,
layer24, mixed, small, big), each against the CPU step on the same inputs (layer 2, scaled checkpoint, measured
on the CPU):
    fp4 grid    every ckv value equals fp4_quant_dequant(value, 16, e4m3 scale) and every k value equals
                fp4_quant_dequant(value, 32): nothing off the grid (exactly 0). Reference, input nudge (3 seeds) and
                bf16 output: 0 on every input. scale 1.02 / 0.98: 0.93 (ckv) / 0.86-0.91 (k) off grid; halfswap 0.10 /
                0.77. (Re-quantizing a quantized block is the identity: its amax is 6 s, so the scale comes back.)
    exact match the fraction of output values that differ from the CPU step <= FP4_MISMATCH = 2 x the input nudge's
                worst half (ckv 0.0114, mixed; k 0.0039). Input nudge: whole output 0.0056-0.0099. The golden vs
                the CPU on it 0.0059. Also caught by the auto checks: rowshift 0.73-0.98, quarterzero 0.23, sign 0.11,
                rope one position late 0.037-0.040 (rel 0.059-0.109), ckv block 32 instead of 16 0.32 (rel 0.081),
                ckv ue8m0 scale instead of e4m3 0.63 (rel 0.126), no fp4 at all 1.0. model.py's own bf16 points ahead of
                fp4 (DS41-38, emulated as a bf16 rounding of the fp4 inputs): 0.029, rel 0.036 > the chunk0 / small /
                big limits (DS41-47): keep the step's intermediates in fp32 on the device.
Built-in checks plus these catch every standard mistake of testing/mutate.py. BRINGUP_IMPL=mutations runs the built-in
sweep's controls and mistakes with these checks added (the built-in sweep alone reports scale as slipped).
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
STEP = "compressor"
LAYER = 2
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
CKV_BLOCK = 16  # fp4 block (columns) of the compressed kv half, e4m3 scale per block
K_BLOCK = 32  # fp4 block (columns) of the index-key half, ue8m0 scale per block
FP4_OFFGRID = 0.0  # fraction of values off the e2m1 x scale grid
FP4_MISMATCH = 0.023  # fraction of values != the CPU step on the same inputs: 2 x the input nudge's worst half (0.0114)


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


def _fp4_errors(got: torch.Tensor, cpu: torch.Tensor, d: int) -> dict | None:
    if got.numel() != cpu.numel():
        return None
    t = got.float().reshape(-1, cpu.shape[-1])
    if not bool(torch.isfinite(t).all()):
        return None
    ckv, k = t[:, :d], t[:, d:]
    off = (K.fp4_quant_dequant(ckv, CKV_BLOCK, e4m3_scale=True) != ckv).float().sum() + (
        K.fp4_quant_dequant(k, K_BLOCK) != k
    ).float().sum()
    return {
        "offgrid": (off / t.numel()).item(),
        "mismatch": (t != cpu.float().reshape(t.shape)).float().mean().item(),
    }


def _fp4_fails(outs: dict, ex, quiet: bool) -> list[str]:
    """The structural checks of the fp4 output on every input; the names of the failed checks."""
    d = ex.ref.cfg.head_dim  # the ckv half; the rest is the index key
    failed = []
    for case in ex.cases:
        got, tag = outs[case.name], f"fp4_{STEP}_L{LAYER:02d}_{case.name}"
        if isinstance(got, Exception):
            failed.append(tag)
            continue
        e = _fp4_errors(got, ex.cpu[case.name], d)
        bad = []
        if e is None:
            bad.append(f"shape {tuple(got.shape)} or non-finite values, want {tuple(ex.cpu[case.name].shape)}")
        else:
            if not e["offgrid"] <= FP4_OFFGRID:
                bad.append(f"{e['offgrid']:.4f} of the values off the e2m1 x scale grid")
            if not e["mismatch"] <= FP4_MISMATCH:
                bad.append(f"{e['mismatch']:.4f} of the values differ from the CPU step (> {FP4_MISMATCH})")
        if bad:
            failed.append(tag)
        if not quiet:
            for k, v in (e or {}).items():
                metrics.record(f"{tag}_{k}", v)
            nums = " ".join(f"{k}={v:.6g}" for k, v in (e or {}).items())
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: {nums}" + (f" ({'; '.join(bad)})" if bad else ""))
    return failed


def _sweep(mesh_device) -> bool:
    """BRINGUP_IMPL=mutations: the built-in sweep (CC.sweep: controls, mistakes) with the fp4 checks added."""
    model_precision.apply(S)
    ex = _expect()
    d = ex.ref.cfg.head_dim
    err_name = f"err_{STEP}_L{LAYER:02d}"

    def verdict(outs, stop_early):
        f = ex.evaluate(outs, quiet=True, stop_early=stop_early)
        ok = CC.error_gate(err_name, outs["golden"], ex.want, ex, quiet=True)
        if ex.thr_literal:
            ok = CC.golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr, ex.mode) and ok
        return f + ([] if ok else [err_name]) + _fp4_fails(outs, ex, quiet=True)

    def stats(outs):
        e = {n: _fp4_errors(o, ex.cpu[n], d) for n, o in outs.items()}
        off = max((v or {"offgrid": 1.0})["offgrid"] for v in e.values())
        mis = [(v or {"mismatch": 1.0})["mismatch"] for v in e.values()]
        return f"offgrid max {off:.4f} mismatch {min(mis):.4f}..{max(mis):.4f}"

    nudged = {
        x.name: CC.run_cpu(
            ex.cpu_step, x, [NF.nudge(t, 99991 + j, dt) for j, (t, dt) in enumerate(zip(x.inputs, ex.dtypes))]
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
    print(f"{'ok  ' if ok else 'FAIL'} sweep {STEP} L{LAYER} (built-in + fp4 checks):")
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
    fails = _fp4_fails(outs, ex, quiet=False)
    assert ok and not fails, f"fp4 output checks failed: {fails}" if fails else "built-in checks failed"

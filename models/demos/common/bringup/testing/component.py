# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component and swap tests. Both read golden inputs and compare with golden outputs; neither runs the CPU model
end to end.

component: one step of one layer alone. Metric ``<metric_prefix>`` (default ``pcc_<step>_L<layer>``).
swap:      the whole block of one layer, with the listed steps on the device and every other step on the CPU
           reference. Metrics ``pcc_swap_<boundary>`` for every boundary and ``pcc_swap_out``. When the block output
           drifts, the step swapped in last is the cause, because every earlier swap already passed its gate.

A step deferred to op-gen (task DEFERRED, F46) runs on the CPU reference in the swap test, as if it were not swapped.
Its own component test fails in device mode when the model hands it a CPU bridge: a bridged step is not on the device.

Per-step swap checks (F49, ``run_swap_test(..., checks="steps")``, what the rendered swap template asks for): the
block-out gate alone misses most single-step bugs (a norm or a gate removes a scale, the residual dominates out), so
every swapped step is also gated on its own output, against the CPU step run on the SAME inputs the swapped step
received (captured in the override, so upstream device error is not blamed on it) and against the golden:
    swap_<step>_vs_cpu / _vs_golden   the step's component-test compare mode (COMPARE read from its frozen test file
                                      with ast; default PCC or match), the limit from the noise floor (F80)
    float outputs: swap_<step>_finite, _rel, _row / column own floors (F83), _bias, _reselect: every limit calib x
                   the step's noise floor on the same inputs (input nudge, official arithmetic), never below one
                   rounding of the output to its dtype; no floor means FAIL ("cannot judge", F138)
    swap_<step>_cpu_bridge            device mode: the model handed a CPU bridge, so the step is not on the device
Stateful steps (attention and indexer write caches): the CPU check runs BEFORE the swapped step, on a copy of the
context whose state is a deep copy, so neither the check nor its order changes the state the rest of the block reads;
in reference mode the swapped step then writes the real state exactly as before. Side effects on the reference object
itself (outside ctx, e.g. a per-chunk top-k record) are not isolated: the swapped step's own run follows the check.
"""

from __future__ import annotations

import ast
import copy
import math

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import run_block, step_reference
from models.demos.common.bringup.testing import accuracy_guard, determinism
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    impl_mode,
    reference_ctx,
)
from models.demos.common.bringup.testing.mutate import kind_of, mutated_step, target_step


def _step(ref, layer: int, name: str):
    for st in ref.block_graph(layer):
        if st.name == name:
            return st
    raise KeyError(f"layer {layer} has no step {name!r}; steps: {[s.name for s in ref.block_graph(layer)]}")


def module_under_test(s, ref, mesh, layer: int, name: str, mode: str | None = None):
    """fn(ref_ctx, dev_ctx, *inputs) for the chosen implementation of one step."""
    from models.demos.common.bringup.plan.op_request import deferred_steps

    mode = mode or impl_mode()
    cpu = ref.component(layer, name)
    if mode == "reference" or (mode == "device" and (s.block_type_of(layer), name) in deferred_steps(s)):
        if mode == "device":
            print(f"{name}: deferred to op-gen, runs on the CPU reference")
        return lambda rctx, dctx, *x: cpu(rctx, *x)
    if mode == "stub":
        return lambda rctx, dctx, *x: torch.zeros_like(cpu(rctx, *x))
    kind = kind_of(mode)
    if kind is not None:  # F49: the reference with this step's output altered; other steps run the plain reference
        if target_step() not in (None, name):
            return lambda rctx, dctx, *x: cpu(rctx, *x)
        step = mutated_step(cpu, kind)
        return lambda rctx, dctx, *x: step(rctx, *x)
    dev = s.hooks().device_component(mesh, s, layer, name)
    fn = lambda rctx, dctx, *x: dev(dctx, *x)  # noqa: E731
    fn.cpu_bridge = getattr(dev, "cpu_bridge", False)
    return fn


def run_component_test(
    s,
    step: str,
    layer: int | None = None,
    mesh=None,
    compare_mode: str | None = None,
    thr: float | None = None,
    metric: str | None = None,
    checks: str | None = "auto",
) -> bool:
    """checks="auto" (F56): the error gate vs the golden and the built-in checks by output kind and the second inputs
    (testing/component_checks.py), every limit from the noise floor; BRINGUP_IMPL=mutations runs the freeze sweep
    (CPU) instead. F138: the golden-only path (checks=None, a fixed threshold) is gone, and a frozen test's literal
    THRESHOLD is recorded as a diagnostic, never gated."""
    if checks != "auto":
        raise ValueError(
            f"checks={checks!r}: the golden-only path with a fixed threshold is gone (F138); render the test with "
            "checks='auto' (testing/templates.py)"
        )
    if thr is not None:
        metrics.record(f"{step}_threshold_literal", thr)
        print(
            f"note {step}: the test's THRESHOLD {thr} is recorded, not gated (F138: limits come from the noise floor)"
        )
    ok = _run_auto(s, step, layer, mesh, compare_mode, None, metric)
    accuracy_guard.note(s, ok)  # a perf pick that fails its frozen test is never profiled (testing/accuracy_guard.py)
    return ok


def _deterministic(s, step: str, name: str, run_a, run_b, first) -> bool:
    """Device mode only: A (already run), B, then A again, bit-identical (testing/determinism.py)."""
    if impl_mode() != "device":
        return True
    why = determinism.exempt(s, step)
    if why:
        print(f"skip {step}: determinism check exempt (tests.nondeterministic: {why})")
        return True
    return determinism.check(
        run_a, run_b, first, label=step, n=determinism.repeats(s), metric=name.replace("pcc_", "det_", 1)
    )


_EXPECT = {}  # (layer, step, chunk, mode, thr) -> Expect; reused only with the same reference object (dev proofs)


def _run_auto(s, step, layer, mesh, compare_mode, thr, metric) -> bool:
    from models.demos.common.bringup.plan.op_request import deferred_steps
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing import noise_floor as NF

    g, c = component_golden(s)
    layer = s.representative_layer(s.block_type_of(layer)) if layer is None else layer
    ref = step_reference(s, [*s.state_sources(layer), layer], torch.float32)  # F112: numerics
    st = _step(ref, layer, step)
    want = g.layer(c, layer)[st.output]
    want = want.float() if want.is_floating_point() else want
    # F76: thr None = the noise floor (CC.Expect: calib x the precision model's miss vs the golden); a literal is kept
    name = metric or f"pcc_{step}_L{layer:02d}"
    err_name = f"err_{step}_L{layer:02d}"  # F81: the gated metric (rel L2, or the missed fraction)
    if impl_mode() == "device" and (s.block_type_of(layer), step) in deferred_steps(s):
        print(f"FAIL {step}: deferred to op-gen (task DEFERRED); it runs on the CPU until the op is delivered")
        return False
    key = (layer, step, c, compare_mode, thr)
    ex = _EXPECT.get(key)
    if ex is None or ex.ref is not ref:  # the CPU side is the same for every module run on this reference object
        ex = _EXPECT[key] = CC.Expect(s, ref, layer, st, g, c, want, thr, compare_mode)
    if impl_mode() == "mutations":
        return CC.sweep(ex, err_name)
    fn = module_under_test(s, ref, mesh, layer, step)
    if getattr(fn, "cpu_bridge", False):
        print(f"FAIL {step}: device_component returned a CPU bridge; a deferred step is not on the device")
        return False
    outs = {}
    for case in ex.cases:
        try:
            outs[case.name] = fn(case.rctx(), case.dctx(), *CC._clone(case.inputs))
        except Exception as e:  # noqa: BLE001 - a module that fails on a second input fails that check
            if case.name == "golden":
                raise
            outs[case.name] = e
    print(ex.describe())
    ok = CC.error_gate(err_name, outs["golden"], want, ex)
    if outs["golden"].numel() == want.numel():
        metrics.record(name, CC.gate_value(outs["golden"], want, ex.kind, compare_mode))  # PCC: a diagnostic only
    ok = not ex.evaluate(outs) and ok
    if not ok and impl_mode() == "device" and ex.kind == "float":  # F137: logic or precision, on the CPU only
        from models.demos.common.bringup.testing import devfmt

        gold0 = next(x for x in ex.cases if x.name == "golden")
        devfmt.diagnose(s, ref, layer, step, gold0, ex.cpu["golden"], outs["golden"], err_name)
    ok = NF.waive(s, s.block_type_of(layer), step, ok, outs["golden"], err_name, want)  # F85: an owner waiver, recorded
    # A, B, A: the second inputs above already ran after the golden (B); run the golden again and compare its bytes
    gold = next(x for x in ex.cases if x.name == "golden")
    other = [x for x in ex.cases if x.name != "golden"]
    b = other[0] if other else gold
    b_inputs = CC._clone(b.inputs) if other else [CC._mixed(t, i) for i, t in enumerate(gold.inputs)]
    r_gold, r_b = gold.rctx(), b.rctx()  # the device module ignores the reference context; build it once
    det = _deterministic(
        s,
        step,
        name,
        lambda: fn(r_gold, gold.dctx(), *CC._clone(gold.inputs)),
        lambda: fn(r_b, b.dctx(), *CC._clone(b_inputs)),
        outs["golden"],
    )
    return ok and det


SPARSE_ZEROS = 0.75  # a float output with at least this fraction of exact zeros is a selection (router weights)


def component_test_limits(s, block_type: str, step: str) -> tuple[str | None, float | None]:
    """COMPARE and THRESHOLD of the step's frozen component test: the two module-level literal assignments, parsed
    with ast (the test is never imported). (None, None) when the file or an assignment is missing."""
    from models.demos.common.bringup.testing.templates import component_test_path

    p = component_test_path(s, block_type, step)
    vals = {}
    if p.exists():
        for node in ast.parse(p.read_text()).body:
            target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
            target = node.target if isinstance(node, ast.AnnAssign) else target
            if isinstance(target, ast.Name) and target.id in ("COMPARE", "THRESHOLD") and node.value is not None:
                try:
                    vals[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return vals.get("COMPARE"), vals.get("THRESHOLD")


def component_error(s, block_type: str, step: str, layer: int) -> float | None:
    """The step's own error in its component gate (task C.<block type>.<step>, its PCC vs golden on golden inputs),
    as a relative L2: sqrt(2 (1 - pcc)) (equal to rel L2 for an unbiased error on a zero-mean output, larger
    otherwise). None when that gate has not recorded it."""
    m = metrics.load(f"C.{block_type}.{step}", s.ledger_dir / "results")
    e = (m.get(f"err_{step}_L{layer:02d}") or {}).get("value")  # F81: the rel error itself
    if isinstance(e, (int, float)) and math.isfinite(e):
        return float(e)
    v = (m.get(f"pcc_{step}_L{layer:02d}") or {}).get("value")
    return math.sqrt(2 * max(0.0, 1 - float(v))) if isinstance(v, (int, float)) else None


def _rows(t: torch.Tensor, shape) -> torch.Tensor:
    t = t.double().reshape(shape)
    return t.reshape(shape[0], -1) if len(shape) >= 2 else t.reshape(1, -1)


def step_errors(got: torch.Tensor, want: torch.Tensor) -> dict:
    """rel L2, worst row rel L2, row norm ratio (min, max, median - 1) of ``got`` vs ``want`` (rows = the first
    dim). Rows whose wanted norm is ~0 (below 1e-6 of the mean row norm) use that floor; a row ~0 in both has ratio 1.

    A selection output (at least SPARSE_ZEROS exact zeros in ``want``, e.g. dense router weights): a row whose
    support moved (the top-k of |got| is not ``want``'s nonzero set, k per row) is a near-tie choice, not a
    precision error; ``reselected`` is the fraction of such rows and the other numbers cover the remaining rows."""
    g, w = _rows(got, want.shape), _rows(want, want.shape)
    out = {}
    if w.shape[1] > 1 and (w == 0).float().mean().item() >= SPARSE_ZEROS:
        nz = w != 0
        rank = (-g.abs()).argsort(1).argsort(1)
        moved = ((rank < nz.sum(1, keepdim=True)) != nz).any(1)
        out["reselected"] = moved.float().mean().item()
        g, w = g[~moved], w[~moved]
        if not len(w):
            return {**out, "rel": math.inf, "row": math.inf, "ratio_min": 0.0, "ratio_max": math.inf, "bias": -1.0}
    wn, gn = w.norm(dim=1), g.norm(dim=1)
    floor = 1e-6 * wn.mean().item() + 1e-30
    wc = wn.clamp_min(floor)
    ratio = torch.where((wn < floor) & (gn < floor), torch.ones_like(wn), gn / wc)
    return {
        **out,
        "rel": ((g - w).norm() / w.norm().clamp_min(1e-30)).item(),
        "row": ((g - w).norm(dim=1) / wc).max().item(),
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
        "bias": ratio.median().item() - 1.0,
    }


def _isolated(ctx):
    """A copy of ctx whose state and extra the CPU check may write without touching the block's own."""
    c = copy.copy(ctx)
    c.state = copy.deepcopy(ctx.state)
    c.extra = dict(ctx.extra)
    return c


def _capturing(
    st,
    mut,
    cpu,
    dctx,
    into: dict,
    snapshot: dict | None = None,
    ref=None,
    models: dict | None = None,
    dtypes=None,
    seeds: int | None = None,
):
    """Override for a swapped step that first runs the CPU step on the same inputs (into[step]). With ``snapshot``,
    the context as the steps after this one see it is kept too (snapshot["ctx"], its state a deep copy). With ``ref``
    and ``models`` (F80), the step's noise floor on the same inputs: models[step] = {"outs": the fp32 step on its
    inputs nudged inside their official dtype's rounding cell, one per seed; "scores": (the scores of its top-k, k,
    the nudged runs' scores) for a selection}; snapshot["model"] = those outputs (the tail check's limit)."""
    from models.demos.common.bringup.testing import noise_floor as NF

    def fn(ctx, *x):
        pre = _isolated(ctx) if ref is not None and models is not None else None
        try:
            cctx = _isolated(ctx) if st.stateful else ctx
            into[st.name] = cpu(cctx, *[t.clone() if isinstance(t, torch.Tensor) else t for t in x])
        except Exception as e:  # noqa: BLE001 - reported as a failed check, not a crash of the block
            into[st.name] = e
        y = mut(ctx, dctx, *x)
        if pre is not None and isinstance(into[st.name], torch.Tensor):
            try:
                run = lambda *inp: cpu(_isolated(pre), *inp)  # noqa: E731
                xs = [t.clone() if isinstance(t, torch.Tensor) else t for t in x]
                d = list(dtypes or ["bf16"] * len(xs))
                _, sc, kk = NF.tapped(run, xs)
                outs, nsc = [], []
                for k_ in range(seeds or NF.SEEDS):
                    with NF.inner(k_, dtypes):
                        o, ns, _ = NF.tapped(
                            run, [NF.nudge(t, 1000 * k_ + j, dj) for j, (t, dj) in enumerate(zip(xs, d))]
                        )
                    outs.append(o)
                    nsc.append(ns)
                ok_scores = sc is not None and all(n is not None for n in nsc)
                models[st.name] = {"outs": outs, "scores": (sc, kk, nsc) if ok_scores else None}
            except Exception as e:  # noqa: BLE001 - no floor: that step's limits fall back (printed)
                print(f"note: no noise floor for {st.name} ({type(e).__name__}: {e})")
        if snapshot is not None:
            snapshot["ctx"] = _isolated(ctx)
            if models is not None and st.name in models:
                snapshot["model"] = models[st.name]["outs"]
        return y

    return fn


def _calib(s, kind) -> float:
    from models.demos.common.bringup.testing import component_checks as CC

    return CC.calib(CC.limits(s))


_STEP_CTX: dict = {}  # step -> (floor, chain) of the running swap test (F79: frozen tests wrap _check_step)
_UPSTREAM_WAIVED: dict = {}  # step -> the first owner-waived swapped step before it in the block (F221)


def CC_limits(s) -> dict:
    from models.demos.common.bringup.testing.component_checks import limits

    return limits(s)


def CC_rel(a, b) -> float:
    from models.demos.common.bringup.testing.component_checks import rel

    return rel(a, b)


def _waive_index(s, block_type: str, step: str, ok: bool, got, cpu, tag: str) -> bool:
    """F161: an integer step output under an owner waiver (DeepSeek-V4.1 indexer: bf16 top-k scores) passes its checks
    when it is a top-k whose per-row overlap with the CPU step is structural (>= 1 - struct_scaled_rel, F97): a wrong
    module (a zero stub, another layer's ids) still fails; a waiver covers precision only."""
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing import noise_floor as NF

    if ok or not NF.waiver(s, block_type, step):
        return ok
    floor = 1.0 - CC.limits(s)["struct_scaled_rel"]
    ov = NF.value("topk_overlap", got.reshape(cpu.shape), cpu) if got.numel() == cpu.numel() else 0.0
    if ov < floor:
        print(f"FAIL {tag}: waived, but the overlap with the CPU step {ov:.4f} < {floor} (a structural error)")
        return False
    return NF.waive(s, block_type, step, False, got, tag, None)


def step_mode(mode: str | None, st, cpu: torch.Tensor) -> str:
    """F160: a top-k's ids are a set (the consumer reads them as one; the reference and a device top-k order them
    differently): compared by per-row overlap, never by position (DeepSeek-V4.1 swap indexer: match 0.0015 while the
    set recall was 0.994). An explicit COMPARE "match" of the frozen test is a positional check and also becomes it."""
    from models.demos.common.bringup.testing.component_checks import output_kind

    if mode in (None, "match") and not cpu.is_floating_point() and output_kind(cpu) == "index":
        return "topk_overlap"
    return mode or default_mode(cpu)


def _check_step(s, block_type: str, layer: int, st, got, cpu, golden, bridged: bool, model=None, chain=None) -> bool:
    """F80: model = the step's noise floor on the same inputs ({"outs": nudged fp32 runs, "scores": ...}, limits vs
    the CPU step), chain = the step's outputs when every swapped step runs nudged from the golden block input, one per
    seed (limit vs the golden). Called with the original eight arguments: frozen swap tests patch this function with
    a wrapper of that signature, so the two come from _STEP_CTX (F79)."""
    if model is None and chain is None:
        model, chain = _STEP_CTX.get(st.name, (None, None))
    from models.demos.common.bringup.testing import noise_floor as NF

    outs = (model or {}).get("outs") if isinstance(model, dict) else None
    chain = [c for c in chain if isinstance(c, torch.Tensor)] if isinstance(chain, list) else None
    tag, ok = f"swap_{st.name}", True
    if bridged:
        metrics.record(f"{tag}_cpu_bridge", 1)
        print(f"FAIL {tag}: device_component returned a CPU bridge; the step is not on the device")
        ok = False
    if isinstance(cpu, Exception):
        print(f"FAIL {tag}: the CPU step on the swapped step's inputs raised {type(cpu).__name__}: {cpu}")
        return False
    mode, thr = component_test_limits(s, block_type, st.name)
    mode = step_mode(mode, st, cpu)
    k = _calib(s, st.kind)
    out_dtype = NF.output_dtype(s, block_type, st.name)  # vs the CPU: one rounding to the official output dtype
    _, chunk = component_golden(s)
    ar = NF.arith(s, layer, st.name, chunk) or {}

    def lim(name, floor_outs, want, arith_pcc=None, res_dtype="bf16"):  # the noise floor; F138: nothing else
        if not floor_outs:
            print(f"FAIL {name}: cannot judge: no noise floor (F138: no fixed limit exists)")
            return math.inf
        mv = min(NF.value(mode, o, want) for o in floor_outs)
        if arith_pcc is not None and mode == "pcc":
            mv = min(mv, arith_pcc)
        # the resolution floor: one rounding of the fp32 CPU step's output to its official dtype (F133)
        v = NF.gate_limit(mode, mv, NF.bf16_res(mode, cpu, res_dtype), k, want.numel())
        metrics.record(f"{name}_limit", v)
        return v

    if not cpu.is_floating_point():  # an integer output (match / overlap)
        ok = compare(f"{tag}_vs_cpu", got, cpu, mode, lim(f"{tag}_vs_cpu", outs, cpu, res_dtype=out_dtype))[1] and ok
        if golden is not None:
            ag = (ar.get("vs_golden") or {}).get("pcc_vs_golden")
            lv = lim(f"{tag}_vs_golden", chain, golden, ag, res_dtype=out_dtype)
            ok = compare(f"{tag}_vs_golden", got, golden, mode, lv)[1] and ok
    else:
        # F81: a float output is gated by its rel error (PCC is blind to a scale or an offset): vs the CPU step by
        # the per-statistic checks below, vs the golden here (calib x the nudged block's rel vs the golden)
        from models.demos.common.bringup.testing import component_checks as CC

        if got.numel() == cpu.numel():
            metrics.record(f"{tag}_vs_cpu", NF.value(mode, got, cpu))  # PCC: diagnostics only
        if golden is not None and got.numel() == golden.numel():
            metrics.record(f"{tag}_vs_golden", NF.value(mode, got, golden))
            e = CC.rel(got, golden)
            if chain:
                fl = max(CC.rel(cc, golden) for cc in chain)
                ag = (ar.get("vs_golden") or {}).get("rel") or 0.0
                gl_lim = k * max(fl, ag, CC.rel(NF.round_to(cpu, out_dtype), cpu))
                how = f"{k} x the nudged block's rel vs the golden {max(fl, ag):.3g}"
            else:
                gl_lim, how = math.inf, "no noise floor: not gated"
            metrics.record(f"{tag}_vs_golden_err", e)
            metrics.record(f"{tag}_vs_golden_err_limit", gl_lim)
            good = e <= gl_lim
            print(f"{'ok  ' if good else 'FAIL'} {tag}_vs_golden_err: rel={e:.6f} (<= {gl_lim:.6f}, {how})")
            up = _UPSTREAM_WAIVED.get(st.name)
            if not good and up:  # F221: an earlier waived step's gap reaches this step's input; vs the CPU still gates
                print(f"  {tag}_vs_golden_err: its inputs carry the waived step {up}")
                good = NF.waive(s, block_type, up, False, got, f"{tag}_vs_golden_err", golden)
            ok = good and ok
    if not cpu.is_floating_point():
        sc = (model or {}).get("scores") if isinstance(model, dict) else None
        if sc is not None and got.numel() == cpu.numel():  # F101: an index output's near ties and tie-aware recall
            scores, kk, nsc = sc
            w = scores.shape[-1]
            want = NF.support(cpu, kk, w)
            fl = NF.flips(scores, kk, NF.support(got.reshape(cpu.shape), kk, w), nsc, want)
            for k_, v in fl.items():
                metrics.record(f"{tag}_flip_{k_}", v)
            good = not fl["unexplained"]
            print(
                f"{'ok  ' if good else 'FAIL'} {tag} near ties: flips={fl['rows']} unexplained={fl['unexplained']} "
                f"recall={fl['recall']:.6f} recall_tie={fl['recall_tie']:.6f} worst_row={fl['worst_recall_tie']:.4f}"
            )
            ok = good and ok
        return _waive_index(s, block_type, st.name, ok, got, cpu, tag)  # F161: the owner waiver reaches index outputs
    if got.numel() != cpu.numel():
        print(f"FAIL {tag}: {got.numel()} elements, the CPU step has {tuple(cpu.shape)}")
        return False
    finite = bool(torch.isfinite(got.float()).all())
    metrics.record(f"{tag}_finite", int(finite))
    if not finite:
        print(f"FAIL {tag}: non-finite output")
        return False
    lim = {"swap_step_reselect": math.inf, "swap_step_bias": math.inf}
    err = component_error(s, block_type, st.name, layer) if mode == "pcc" else None
    if err is not None:
        metrics.record(f"{tag}_component_err", err)
    res_e = step_errors(NF.round_to(cpu, out_dtype), cpu)  # one rounding to the official output dtype: an exact step
    fl = None
    if outs and all(isinstance(o, torch.Tensor) and o.is_floating_point() for o in outs):
        # F80: every limit is calib x the floor's statistic on the same inputs: the larger of the input nudge and
        # the official arithmetic (cached; its golden-input statistics), never below one rounding to the output dtype
        me = NF.merge_float(ar.get("vs_step"), NF.worst_float(outs, cpu, step_errors))
        dev = lambda e_: max(1 - e_["ratio_min"], e_["ratio_max"] - 1)  # noqa: E731
        rel_lim = k * max(me["rel"], res_e["rel"])
        lim["swap_step_row"] = k * max(me["row"], res_e["row"])
        lim["swap_step_ratio"] = k * max(dev(me), dev(res_e))
        share = CC_limits(s)["component_bias_share"]
        lim["swap_step_bias"] = max(k * abs(me["bias"]), k * abs(res_e["bias"]), share * rel_lim)
        nres = [step_errors(o, cpu).get("reselected") for o in outs]
        if any(r is not None for r in nres):
            lim["swap_step_reselect"] = k * max(max(r or 0.0 for r in nres), 1.0 / max(cpu.shape[0], 1))
        for k_ in ("swap_step_row", "swap_step_ratio", "swap_step_bias", "swap_step_reselect"):
            metrics.record(f"{tag}_{k_[len('swap_step_'):]}_limit", lim[k_])
        sc = (model or {}).get("scores")
        if sc is not None:  # a selection with tapped scores: every flip must be explained by the nudge
            scores, kk, nsc = sc
            w = scores.shape[-1]
            fl = NF.flips(scores, kk, NF.support(got.float(), kk, w), nsc, NF.support(cpu, kk, w))
            for k_, v in fl.items():
                metrics.record(f"{tag}_flip_{k_}", v)
    else:  # F138: no floor, no limit to judge by (the fixed swap_step_* fallbacks are gone)
        print(f"FAIL {tag}: cannot judge: no noise floor for this step (F138: no fixed limit exists)")
        rel_lim = lim["swap_step_row"] = lim["swap_step_ratio"] = -math.inf
    metrics.record(f"{tag}_rel_limit", rel_lim)
    e = step_errors(got, cpu)
    own = None
    if outs and all(isinstance(o, torch.Tensor) and o.is_floating_point() for o in outs):
        # F83: each row and column against its own floor, and the additive bias per column, replace the worst-row and
        # row-ratio limits (they divided by small norms behind a fixed 1e-6 floor)
        lim["swap_step_row"] = lim["swap_step_ratio"] = math.inf
        nud = max(CC_rel(o, cpu) for o in outs)
        ar_rel = (ar.get("vs_step") or {}).get("rel") or 0.0
        own = NF.own_floor(got, cpu, outs, out_dtype, max(1.0, ar_rel / nud) if nud > 0 else 1.0)
        e.update(own)
    for key, v in e.items():
        metrics.record(f"{tag}_{key}", v)
    dev = max(1 - e["ratio_min"], e["ratio_max"] - 1)
    bad = [
        f"rel {e['rel']:.5f} > {rel_lim:.5f}" if not e["rel"] <= rel_lim else "",
        f"worst row {e['row']:.5f} > {lim['swap_step_row']:.5f}" if not e["row"] <= lim["swap_step_row"] else "",
        (
            f"row norm ratio [{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] outside 1 +- {lim['swap_step_ratio']:.5f}"
            if not dev <= lim["swap_step_ratio"]
            else ""
        ),
        (
            f"median row norm ratio off by {e['bias']:+.5f} (limit {lim['swap_step_bias']:.5f})"
            if not abs(e["bias"]) <= lim["swap_step_bias"]
            else ""
        ),
        (
            f"{e['reselected']:.4f} of the rows select another support (limit {lim['swap_step_reselect']:.4f})"
            if not e.get("reselected", 0.0) <= lim["swap_step_reselect"]
            else ""
        ),
        f"{fl['unexplained']} rows select an entry the nudge does not explain" if fl and fl["unexplained"] else "",
    ]
    bad = [b for b in bad if b]
    warn = [  # F89: the own-floor row / column / bias checks warn, never gate (owner)
        f"row {own['row_at']} is {own['row_floor']:.3g} x its own floor (limit {k:g})"
        if own and own["row_floor"] > k
        else "",
        (
            f"column {own['col_at']} is {own['col_floor']:.3g} x its own floor (limit {k:g})"
            if own and own.get("col_floor", 0) > k
            else ""
        ),
        (
            f"column {own['bias_at']} has an additive bias of {own['bias_signed']:+.3g} of its rms, "
            f"{own['bias_floor']:.3g} x its floor (limit {k:g})"
            if own and own.get("bias_floor", 0) > k and fl is None
            else ""
        ),
    ]
    warn = [w for w in warn if w]
    print(
        f"{'ok  ' if not bad else 'FAIL'} {tag} vs the CPU step on the same inputs: rel={e['rel']:.6f} "
        f"(<= {rel_lim:.5f}{f', component error {err:.5f}' if err is not None else ''}) row={e['row']:.6f} "
        f"ratio=[{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] bias={e['bias']:+.6f}"
        + (f" reselected={e['reselected']:.4f}" if "reselected" in e else "")
        + (f" flips={fl['rows']} unexplained={fl['unexplained']} recall_tie={fl['recall_tie']:.6f}" if fl else "")
        + (
            f" own-floor row={own['row_floor']:.3g} col={own.get('col_floor', 0):.3g} bias={own.get('bias_floor', 0):.3g}"
            if own
            else ""
        )
        + (f" ({'; '.join(bad)})" if bad else "")
    )
    if warn:
        metrics.record(f"{tag}_floor_warn", 1)
        print(f"WARN {tag} (not gating, for the owner's review; do not work on it): {'; '.join(warn)}")
    return NF.waive(s, block_type, st.name, ok and not bad, got, tag, cpu)  # F85: an owner waiver, recorded


def _check_tail(s, ref, layer: int, steps, last, seen: dict, cpu, snapshot: dict) -> bool:
    """The last swapped step's effect on the block output: the steps after it run once more on the CPU from the CPU
    step's output (same inputs, the context as they saw it), and the two block outputs are compared. A small error
    the step's own limits allow can still be one the block amplifies (a gate on a large stream)."""
    tail = steps[[st.name for st in steps].index(last.name) + 1 :]
    if not tail or isinstance(cpu, Exception) or not seen["out"].is_floating_point():
        return True
    from models.demos.common.bringup.testing import component_checks as CC

    tag, lim = f"swap_{last.name}_out", -math.inf  # F138: the noise floor below, else cannot judge

    def run_tail(first: torch.Tensor) -> torch.Tensor:
        env = dict(seen)
        env[last.output] = first
        ctx = _isolated(snapshot["ctx"])  # the context the steps after the last swapped one saw, per run
        for st in tail:
            env[st.output] = ref.component(layer, st.name)(ctx, *[env[i] for i in st.inputs])
        return env["out"].float()

    try:
        want = run_tail(cpu)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {tag}: the CPU tail after {last.name} raised {type(e).__name__}: {e}")
        return False
    e = step_errors(seen["out"], want)
    metrics.record(f"{tag}_rel", e["rel"])
    metrics.record(f"{tag}_row", e["row"])  # diagnosis only: a routing flip moves single rows
    # F72: the limit is the noise floor, not a fixed number: what a correct bf16 device step does to the block out
    # (the model's own rounding steps amplify it), i.e. the same tail from the step's bf16 precision model, times
    # the one calibration margin; no limit when the model cannot run (F138)
    how = "cannot judge: no noise floor"
    model = snapshot.get("model")
    model = [model] if isinstance(model, torch.Tensor) else list(model or [])
    if model and cpu.is_floating_point():  # F98: a correct device step's output is at least rounded to bf16; the
        model.append(cpu.to(torch.bfloat16).float())  # steps after it (a top-k) can flip near-ties on that alone
    # an int step output (a top-k) too: the nudged fp32 step's own selection flips are its floor (DS41-50)
    if model and all(isinstance(m, torch.Tensor) for m in model):
        try:
            em = max(CC.rel(run_tail(m.float() if m.is_floating_point() else m), want) for m in model)
            res = CC.rel(want.to(torch.bfloat16).float(), want)  # the smallest difference a bf16 output can show
            metrics.record(f"{tag}_model_rel", em)
            k = CC.limits(s)["component_calib"]  # one margin over the noise floor; no fixed floor or cap (owner)
            lim, how = k * max(em, res), f"{k} x the noise floor's {em:.6f} (bf16 resolution {res:.6f})"
        except Exception as ex:  # noqa: BLE001
            print(f"FAIL: the tail from {last.name}'s noise floor raised {type(ex).__name__}: {ex}; cannot judge")
    metrics.record(f"{tag}_rel_limit", lim)
    ok = e["rel"] <= lim
    print(
        f"{'ok  ' if ok else 'FAIL'} {tag}: block out vs the same block with the CPU {last.name} output: "
        f"rel={e['rel']:.6f} (<= {lim:.6f}, {how}) worst row={e['row']:.6f}"
    )
    from models.demos.common.bringup.testing import noise_floor as NF

    return NF.waive(s, s.block_type_of(layer), last.name, ok, seen["out"], tag, want)  # F85: the last step's waiver


def _model_chain(s, ref, hooks, steps, layer: int, g, c: int, swapped, block_type: str) -> list | None:
    """F80: the block from the golden input with every swapped step run as the fp32 step on its inputs nudged inside
    their official dtype's rounding cell, every other step on the fp32 CPU; once per seed. Each run's outputs are a
    noise floor vs the golden. The reference object's attributes are restored after each run (a per-chunk record must
    not leak into the block under test). None when it cannot run (printed)."""
    from models.demos.common.bringup.testing import noise_floor as NF

    runs = []
    for seed in range(NF.SWAP_SEEDS):
        saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(ref).items()}
        try:
            rctx, dctx = reference_ctx(ref, layer, g, c, s), device_ctx(layer, g, c, s)
            if hasattr(hooks, "swap_context"):
                hooks.swap_context(s, ref, layer, g, c, rctx, dctx)

            def as_nudged(name, seed=seed):
                cpu = ref.component(layer, name)
                d = NF.input_dtypes(s, block_type, name, len(_step(ref, layer, name).inputs))

                def f(ctx, *x):
                    with NF.inner(seed, d):
                        return cpu(ctx, *[NF.nudge(t, 1000 * seed + j, dj) for j, (t, dj) in enumerate(zip(x, d))])

                return f

            out = {}
            run_block(
                steps,
                lambda n: ref.component(layer, n),
                rctx,
                g.layer(c, layer)["in"].float(),
                rec=lambda n, t: out.__setitem__(n, t.float() if t.is_floating_point() else t),
                overrides={n: as_nudged(n) for n in swapped},
            )
            runs.append(out)
        except Exception as e:  # noqa: BLE001
            print(f"note: the nudged block raised {type(e).__name__}: {e}; fixed limits")
            return None
        finally:
            vars(ref).clear()
            vars(ref).update(saved)
    return runs


def run_swap_test(
    s, block_type: str, swapped: list[str], mesh=None, thr: float | None = None, checks: str | None = None
) -> bool:
    """Block of the block type's representative layer; ``swapped`` steps run under the chosen implementation.

    checks=None: the block-out gate only (pcc_swap_out), plus the ungated trail. checks="steps" (F49): also every
    swapped step's own output, gated (module docstring). An optional model hook ``swap_context(spec, ref, layer,
    golden, chunk, rctx, dctx)`` may put what one block cannot compute into both contexts (e.g. another layer's top-k
    from the golden)."""
    if checks not in (None, "steps"):
        raise ValueError(f"checks={checks!r}: None or 'steps'")
    g, c = component_golden(s)
    layer = s.representative_layer(block_type)
    hooks = s.hooks()
    ref = step_reference(s, [*s.state_sources(layer), layer], torch.float32)  # F112: numerics
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c, s), device_ctx(layer, g, c, s)
    if hasattr(hooks, "swap_context"):
        hooks.swap_context(s, ref, layer, g, c, rctx, dctx)
    overrides, same_input, bridged, snapshot, models = {}, {}, {}, {}, {}
    names = [st.name for st in steps]
    last = max((_step(ref, layer, n) for n in swapped), key=lambda st: names.index(st.name), default=None)
    for name in swapped:
        st = _step(ref, layer, name)
        mut = module_under_test(s, ref, mesh, layer, name)
        bridged[name] = impl_mode() == "device" and getattr(mut, "cpu_bridge", False)
        if checks:
            snap = snapshot if st.name == last.name else None
            from models.demos.common.bringup.testing import noise_floor as NF

            dtypes = NF.input_dtypes(s, block_type, name, len(st.inputs))  # F80: the official dtypes of its inputs
            seeds = NF.SEEDS if st.name == last.name else NF.SWAP_SEEDS  # the earlier steps had theirs in their own
            overrides[name] = _capturing(
                st, mut, ref.component(layer, name), dctx, same_input, snap, ref, models, dtypes, seeds
            )
        else:
            overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
    gl = g.layer(c, layer)
    seen = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        rctx,
        gl["in"].float(),
        rec=lambda n, t: seen.__setitem__(n, t),
        overrides=overrides,
    )
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    if thr is not None:  # F138: a frozen test's literal THRESHOLD is recorded, never gated
        metrics.record("pcc_swap_out_threshold_literal", thr)
    chain = _model_chain(s, ref, hooks, steps, layer, g, c, swapped, block_type)
    gout = gl["out"].float()
    compare("pcc_swap_out", seen["out"], gl["out"], "pcc", 0.0)  # PCC: a diagnostic only
    ok = True
    # F81: the gated metric is the block out's rel error vs the golden, against calib x the nudged block's
    from models.demos.common.bringup.testing import component_checks as CC

    if chain:
        k = max(_calib(s, _step(ref, layer, n).kind) for n in swapped) if swapped else _calib(s, None)
        fp32_out = same_input.get(last.name) if last is not None and last.output == "out" else None
        fp32_out = fp32_out if isinstance(fp32_out, torch.Tensor) else chain[0]["out"]
        fl = max(CC.rel(r["out"], gout) for r in chain)
        res = CC.rel(fp32_out.float().to(torch.bfloat16).float(), fp32_out.float())
        elim, how = k * max(fl, res), f"{k} x the nudged block's rel vs the golden {fl:.3g} (bf16 {res:.3g})"
    else:  # F138: no fixed fallback
        elim, how = -math.inf, "cannot judge: the noise-floor block did not run"
    e = CC.rel(seen["out"], gout) if seen["out"].numel() == gout.numel() else math.inf
    metrics.record("err_swap_out", e)
    metrics.record("err_swap_out_limit", elim)
    print(f"{'ok  ' if e <= elim else 'FAIL'} err_swap_out: rel={e:.6f} (<= {elim:.6f}, {how})")
    out_ok = e <= elim
    if not out_ok:  # F158: an owner waiver of a swapped step covers the block it is in (its gap flows to the block out)
        from models.demos.common.bringup.testing import noise_floor as NF

        waived = [n for n in swapped if NF.waiver(s, block_type, n)]
        if waived:
            print(f"  err_swap_out: the block contains the waived step {waived[0]}")
            out_ok = NF.waive(s, block_type, waived[0], False, seen["out"], "err_swap_out", gout)
    ok = out_ok and ok
    if checks == "steps":
        _STEP_CTX.clear()
        _UPSTREAM_WAIVED.clear()
        from models.demos.common.bringup.testing import noise_floor as NF

        order = sorted(swapped, key=names.index)
        for i, name in enumerate(order):
            up = [n for n in order[:i] if NF.waiver(s, block_type, n)]
            if up:
                _UPSTREAM_WAIVED[name] = up[0]
        for name in swapped:
            out_name = _step(ref, layer, name).output
            _STEP_CTX[name] = (models.get(name), [r.get(out_name) for r in chain] if chain else None)
        for name in swapped:
            st = _step(ref, layer, name)
            golden = gl[st.output] if st.output in gl else None
            golden = golden.float() if golden is not None and golden.is_floating_point() else golden
            ok = _check_step(s, block_type, layer, st, seen[st.output], same_input[name], golden, bridged[name]) and ok
        if last is not None:
            ok = _check_tail(s, ref, layer, steps, last, seen, same_input[last.name], snapshot) and ok
    accuracy_guard.note(s, ok)
    return ok

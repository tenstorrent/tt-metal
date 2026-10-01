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
    swap_<step>_vs_cpu / _vs_golden   the step's component-test compare mode and threshold (COMPARE / THRESHOLD read
                                      from its frozen test file with ast; default PCC or match, thresholds.component)
    float outputs: swap_<step>_finite, _rel (<= thresholds.swap_step_rel, 0.02), _row (worst row rel L2 <=
                   thresholds.swap_step_row, 0.05), _ratio_min / _ratio_max (every row's norm ratio within
                   1 +- thresholds.swap_step_ratio, 0.02), _bias (median row norm ratio within 1 +-
                   thresholds.swap_step_bias, 0.01: a systematic scale; precision error is unbiased)
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

from models.demos.common.bringup.core import defaults, metrics
from models.demos.common.bringup.reference.interface import run_block
from models.demos.common.bringup.testing import accuracy_guard, model_precision
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    impl_mode,
    reference_ctx,
    threshold,
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
    checks: str | None = None,
) -> bool:
    """checks=None: the gate vs the golden only (compare_mode, thr). checks="auto" (F56): also the built-in checks
    by output kind and the second inputs (testing/component_checks.py); BRINGUP_IMPL=mutations runs the freeze sweep
    (CPU) instead."""
    model_precision.apply(s)  # framework rule: steps are checked at maximum precision
    if checks not in (None, "auto"):
        raise ValueError(f"checks={checks!r}: None or 'auto'")
    if checks == "auto":
        ok = _run_auto(s, step, layer, mesh, compare_mode, thr, metric)
    else:
        ok = _run_golden(s, step, layer, mesh, compare_mode, thr, metric)
    accuracy_guard.note(s, ok)  # a perf pick that fails its frozen test is never profiled (testing/accuracy_guard.py)
    return ok


def _run_golden(s, step, layer, mesh, compare_mode, thr, metric) -> bool:
    ref_hooks = s.hooks()
    g, c = component_golden(s)
    layer = s.representative_layer(s.block_type_of(layer)) if layer is None else layer
    ref = ref_hooks.reference(s, layers=[layer], dtype=torch.float32)
    st = _step(ref, layer, step)
    gl = g.layer(c, layer)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    from models.demos.common.bringup.plan.op_request import deferred_steps

    if impl_mode() == "device" and (s.block_type_of(layer), step) in deferred_steps(s):
        print(f"FAIL {step}: deferred to op-gen (task DEFERRED); it runs on the CPU until the op is delivered")
        return False
    fn = module_under_test(s, ref, mesh, layer, step)
    if getattr(fn, "cpu_bridge", False):
        print(f"FAIL {step}: device_component returned a CPU bridge; a deferred step is not on the device")
        return False
    out = fn(reference_ctx(ref, layer, g, c), device_ctx(layer, g, c), *inputs)
    mode = compare_mode or default_mode(want)
    _, ok = compare(
        metric or f"pcc_{step}_L{layer:02d}", out, want, mode, threshold(s, "component") if thr is None else thr
    )
    return ok


_EXPECT = {}  # (layer, step, chunk, mode, thr) -> Expect; reused only with the same reference object (dev proofs)


def _run_auto(s, step, layer, mesh, compare_mode, thr, metric) -> bool:
    from models.demos.common.bringup.plan.op_request import deferred_steps
    from models.demos.common.bringup.testing import component_checks as CC

    g, c = component_golden(s)
    layer = s.representative_layer(s.block_type_of(layer)) if layer is None else layer
    ref = s.hooks().reference(s, layers=[layer], dtype=torch.float32)
    st = _step(ref, layer, step)
    want = g.layer(c, layer)[st.output]
    want = want.float() if want.is_floating_point() else want
    thr = threshold(s, "component") if thr is None else thr
    name = metric or f"pcc_{step}_L{layer:02d}"
    if impl_mode() == "device" and (s.block_type_of(layer), step) in deferred_steps(s):
        print(f"FAIL {step}: deferred to op-gen (task DEFERRED); it runs on the CPU until the op is delivered")
        return False
    key = (layer, step, c, compare_mode, thr)
    ex = _EXPECT.get(key)
    if ex is None or ex.ref is not ref:  # the CPU side is the same for every module run on this reference object
        ex = _EXPECT[key] = CC.Expect(s, ref, layer, st, g, c, want, thr, compare_mode)
    if impl_mode() == "mutations":
        return CC.sweep(ex, name)
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
    ok = CC.golden_gate(name, outs["golden"], want, ex.kind, compare_mode, thr)
    return not ex.evaluate(outs) and ok


SWAP_STEP_DEFAULTS = {k: v for k, v in defaults.get("thresholds").items() if k.startswith("swap_")}  # defaults.yaml
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
    m = metrics.load(f"C.{block_type}.{step}", s.bringup_dir / "results")
    v = (m.get(f"pcc_{step}_L{layer:02d}") or {}).get("value")
    return math.sqrt(2 * max(0.0, 1 - float(v))) if isinstance(v, (int, float)) else None


def _rows(t: torch.Tensor, shape) -> torch.Tensor:
    t = t.float().reshape(shape)
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


def _capturing(st, mut, cpu, dctx, into: dict, snapshot: dict | None = None):
    """Override for a swapped step that first runs the CPU step on the same inputs (into[step]). With ``snapshot``,
    the context as the steps after this one see it is kept too (snapshot["ctx"], its state a deep copy)."""

    def fn(ctx, *x):
        try:
            cctx = _isolated(ctx) if st.stateful else ctx
            into[st.name] = cpu(cctx, *[t.clone() if isinstance(t, torch.Tensor) else t for t in x])
        except Exception as e:  # noqa: BLE001 - reported as a failed check, not a crash of the block
            into[st.name] = e
        y = mut(ctx, dctx, *x)
        if snapshot is not None:
            snapshot["ctx"] = _isolated(ctx)
        return y

    return fn


def _check_step(s, block_type: str, layer: int, st, got, cpu, golden, bridged: bool) -> bool:
    tag, ok = f"swap_{st.name}", True
    if bridged:
        metrics.record(f"{tag}_cpu_bridge", 1)
        print(f"FAIL {tag}: device_component returned a CPU bridge; the step is not on the device")
        ok = False
    if isinstance(cpu, Exception):
        print(f"FAIL {tag}: the CPU step on the swapped step's inputs raised {type(cpu).__name__}: {cpu}")
        return False
    mode, thr = component_test_limits(s, block_type, st.name)
    mode = mode or default_mode(cpu)
    thr = threshold(s, "component") if thr is None else thr
    ok = compare(f"{tag}_vs_cpu", got, cpu, mode, thr)[1] and ok
    if golden is not None:
        ok = compare(f"{tag}_vs_golden", got, golden, mode, thr)[1] and ok
    if not cpu.is_floating_point():
        return ok
    if got.numel() != cpu.numel():
        print(f"FAIL {tag}: {got.numel()} elements, the CPU step has {tuple(cpu.shape)}")
        return False
    finite = bool(torch.isfinite(got.float()).all())
    metrics.record(f"{tag}_finite", int(finite))
    if not finite:
        print(f"FAIL {tag}: non-finite output")
        return False
    lim = {k: s.threshold(k, v) for k, v in SWAP_STEP_DEFAULTS.items()}
    rel_lim, err = lim["swap_step_rel"], component_error(s, block_type, st.name, layer) if mode == "pcc" else None
    if err is not None:  # as accurate in the block as in its own gate: an exact (fp32) step gets a tight limit
        metrics.record(f"{tag}_component_err", err)
        rel_lim = min(rel_lim, max(lim["swap_step_floor"], lim["swap_step_calib"] * err))
    metrics.record(f"{tag}_rel_limit", rel_lim)
    e = step_errors(got, cpu)
    for k, v in e.items():
        metrics.record(f"{tag}_{k}", v)
    dev = max(1 - e["ratio_min"], e["ratio_max"] - 1)
    bad = [
        f"rel {e['rel']:.5f} > {rel_lim:.5f}" if not e["rel"] <= rel_lim else "",
        f"worst row {e['row']:.5f} > {lim['swap_step_row']}" if not e["row"] <= lim["swap_step_row"] else "",
        (
            f"row norm ratio [{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] outside 1 +- {lim['swap_step_ratio']}"
            if not dev <= lim["swap_step_ratio"]
            else ""
        ),
        (
            f"median row norm ratio off by {e['bias']:+.5f} (limit {lim['swap_step_bias']})"
            if not abs(e["bias"]) <= lim["swap_step_bias"]
            else ""
        ),
        (
            f"{e['reselected']:.4f} of the rows select another support (limit {lim['swap_step_reselect']})"
            if not e.get("reselected", 0.0) <= lim["swap_step_reselect"]
            else ""
        ),
    ]
    bad = [b for b in bad if b]
    print(
        f"{'ok  ' if not bad else 'FAIL'} {tag} vs the CPU step on the same inputs: rel={e['rel']:.6f} "
        f"(<= {rel_lim:.5f}{f', component error {err:.5f}' if err is not None else ''}) row={e['row']:.6f} "
        f"ratio=[{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] bias={e['bias']:+.6f}"
        + (f" reselected={e['reselected']:.4f}" if "reselected" in e else "")
        + (f" ({'; '.join(bad)})" if bad else "")
    )
    return ok and not bad


def _check_tail(s, ref, layer: int, steps, last, seen: dict, cpu, snapshot: dict) -> bool:
    """The last swapped step's effect on the block output: the steps after it run once more on the CPU from the CPU
    step's output (same inputs, the context as they saw it), and the two block outputs are compared. A small error
    the step's own limits allow can still be one the block amplifies (a gate on a large stream)."""
    tail = steps[[st.name for st in steps].index(last.name) + 1 :]
    if not tail or isinstance(cpu, Exception) or not seen["out"].is_floating_point():
        return True
    tag, lim = f"swap_{last.name}_out", s.threshold("swap_out_rel", SWAP_STEP_DEFAULTS["swap_out_rel"])
    env = dict(seen)
    env[last.output] = cpu
    ctx = snapshot["ctx"]  # a private copy: the context the steps after the last swapped one saw
    try:
        for st in tail:
            env[st.output] = ref.component(layer, st.name)(ctx, *[env[i] for i in st.inputs])
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {tag}: the CPU tail after {last.name} raised {type(e).__name__}: {e}")
        return False
    e = step_errors(seen["out"], env["out"].float())
    metrics.record(f"{tag}_rel", e["rel"])
    metrics.record(f"{tag}_row", e["row"])  # diagnosis only: a routing flip moves single rows
    ok = e["rel"] <= lim
    print(
        f"{'ok  ' if ok else 'FAIL'} {tag}: block out vs the same block with the CPU {last.name} output: "
        f"rel={e['rel']:.6f} (<= {lim}) worst row={e['row']:.6f}"
    )
    return ok


def run_swap_test(
    s, block_type: str, swapped: list[str], mesh=None, thr: float | None = None, checks: str | None = None
) -> bool:
    """Block of the block type's representative layer; ``swapped`` steps run under the chosen implementation.

    checks=None: the block-out gate only (pcc_swap_out), plus the ungated trail. checks="steps" (F49): also every
    swapped step's own output, gated (module docstring). An optional model hook ``swap_context(spec, ref, layer,
    golden, chunk, rctx, dctx)`` may put what one block cannot compute into both contexts (e.g. another layer's top-k
    from the golden)."""
    model_precision.apply(s)  # framework rule: steps are checked at maximum precision
    if checks not in (None, "steps"):
        raise ValueError(f"checks={checks!r}: None or 'steps'")
    g, c = component_golden(s)
    layer = s.representative_layer(block_type)
    hooks = s.hooks()
    ref = hooks.reference(s, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    if hasattr(hooks, "swap_context"):
        hooks.swap_context(s, ref, layer, g, c, rctx, dctx)
    overrides, same_input, bridged, snapshot = {}, {}, {}, {}
    names = [st.name for st in steps]
    last = max((_step(ref, layer, n) for n in swapped), key=lambda st: names.index(st.name), default=None)
    for name in swapped:
        st = _step(ref, layer, name)
        mut = module_under_test(s, ref, mesh, layer, name)
        bridged[name] = impl_mode() == "device" and getattr(mut, "cpu_bridge", False)
        if checks:
            snap = snapshot if st.name == last.name else None
            overrides[name] = _capturing(st, mut, ref.component(layer, name), dctx, same_input, snap)
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
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", threshold(s, "block") if thr is None else thr)
    if checks == "steps":
        for name in swapped:
            st = _step(ref, layer, name)
            golden = gl[st.output] if st.output in gl else None
            golden = golden.float() if golden is not None and golden.is_floating_point() else golden
            ok = _check_step(s, block_type, layer, st, seen[st.output], same_input[name], golden, bridged[name]) and ok
        if last is not None:
            ok = _check_tail(s, ref, layer, steps, last, seen, same_input[last.name], snapshot) and ok
    accuracy_guard.note(s, ok)
    return ok

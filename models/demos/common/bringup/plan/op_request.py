# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Op requests (F46): a component step TTNN has no proper op for, deferred to the op code generator (op-gen,
tt_metal/third_party/tt_ops_code_gen). The step stays on the CPU reference through testing/cpu_bridge.py, the rest of
the bring-up continues, and the owner decides whether to launch op-gen. ``<bringup_dir>/op_requests/<op>/``:

    request.yaml     op, model, tasks, component, layers, status (draft | approved | exported | delivered), evidence
                     (searched, tried, why_not_fork), interface (per-device inputs / output, chunks, params),
                     tolerance, acceptance (the frozen component test and its golden: the final check on real data)
    op_prompt.txt    op-gen's prompt: first line ``# golden: <op>``, then the math, signature, rules, import path
    feature_spec.py  single-value TARGET axes, the model's per-device shapes as INPUTS, INVALID = [] (strings, not
                     ttnn enums; op-export translates them)
    reference.py     ``pytorch_<op>``: the step as a standalone pure-torch function (no model imports)
    bind.py          model side, not exported: the weights and scalars pytorch_<op> takes beyond the step's inputs

    python -m models.demos.common.bringup.plan.op_request new <op> --task C.<bt>.<step> --spec S
    python -m models.demos.common.bringup.plan.op_request refresh <dir> --spec S    # shapes + feature_spec.py again
    python -m models.demos.common.bringup.plan.op_request check <dir> --spec S      # what the orchestrator runs

``new`` fills everything mechanical from the spec, the ledger, the reference and the frozen test's golden; the agent
writes the evidence, the prompt prose (``<<AGENT ...>>`` markers) and the bodies of reference.py and bind.py. The
checker rejects a request with a marker left, thin evidence, a malformed prompt or suite, or a reference that does
not reproduce the model's CPU step on a random input.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import keyword
import math
import re
import sys
import textwrap
import types
from pathlib import Path

import yaml

REQUESTS = "op_requests"
STATUSES = ("draft", "approved", "exported", "delivered")
FILES = ("request.yaml", "op_prompt.txt", "feature_spec.py", "reference.py", "bind.py")
MARK = "<<AGENT"  # a generator placeholder the agent must replace
OP_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
PURE_IMPORTS = {"torch", "math", "numpy", "typing", "__future__", "functools"}
CHECK_ROWS = 32  # S of the checker's random input
MIN_WHY_NOT_FORK = 40
MIN_OUTCOME = 20


# ---------------------------------------------------------------- where and what
def root(spec) -> Path:
    return spec.bringup_dir / REQUESTS


def request_dir(spec, op: str) -> Path:
    return root(spec) / op


def load(d: Path) -> dict:
    return yaml.safe_load((Path(d) / "request.yaml").read_text()) or {}


def save(d: Path, req: dict) -> None:
    (Path(d) / "request.yaml").write_text(yaml.safe_dump(req, sort_keys=False, width=120))


def all_requests(spec) -> list[tuple[Path, dict]]:
    r = root(spec)
    return [(p.parent, load(p.parent)) for p in sorted(r.glob("*/request.yaml"))] if r.exists() else []


def for_task(spec, tid: str) -> list[Path]:
    return [d for d, req in all_requests(spec) if tid in (req.get("tasks") or [])]


def deferrable(task: dict) -> bool:
    """Only a component task of the implement step may be deferred (not swaps, integration, other roles)."""
    return task.get("step") == "implement" and "step" in (task.get("brief") or {})


def deferred_steps(spec) -> set[tuple[str, str]]:
    """(block type, step) of every DEFERRED task: the harness runs these on the CPU reference."""
    from models.demos.common.bringup.core.ledger import Ledger

    led = Ledger(spec.bringup_dir)
    if not led.tasks_path.exists():
        return set()
    tasks = led.tasks()
    out = set()
    for tid in led.deferred():
        b = tasks[tid].get("brief") or {}
        if "step" in b:
            out.add((b["block_type"], b["step"]))
    return out


def evidence_summary(req: dict) -> str:
    ev = req.get("evidence") or {}
    tried = ev.get("tried") or []
    first = tried[0] if tried else {}
    return (
        f"searched {len(ev.get('searched') or [])} places; tried {len(tried)}"
        + (f" (first: {first.get('what')}: {first.get('outcome')})" if first else "")
        + f"; not a fork because: {ev.get('why_not_fork', '')}"
    )


# ---------------------------------------------------------------- shapes
def chunks(spec) -> list[int]:
    """Every chunk length the model runs (ladder rungs and the target): the op's S values."""
    got = {int(r["chunk"]) for r in spec.data.get("ladder", [])}
    if spec.get("target.chunk"):
        got.add(int(spec.get("target.chunk")))
    return sorted(got)


def per_device(shape: list, placement: str, mesh: list[int]) -> list:
    """replicate | shard:<dim> (over every chip) | shard2d:<dim on mesh rows>,<dim on mesh cols> (either may be
    'none'). 'S' (the chunk length) stays symbolic."""
    out = list(shape)
    if placement == "replicate":
        return out
    if placement.startswith("shard:"):
        pairs = [(int(placement.split(":")[1]), mesh[0] * mesh[1])]
    elif placement.startswith("shard2d:"):
        a, b = placement.split(":")[1].split(",")
        pairs = [(int(x), n) for x, n in ((a, mesh[0]), (b, mesh[1])) if x.strip() != "none"]
    else:
        raise ValueError(f"placement {placement!r}: replicate, shard:<dim> or shard2d:<dim>,<dim>")
    for dim, n in pairs:
        v = out[dim]
        out[dim] = f"S/{n}" if v == "S" else v // n
    return out


def concrete(shape: list, s: int) -> tuple:
    """A symbolic shape at chunk length s ('S' and 'S/<n>' resolved)."""
    res = []
    for v in shape:
        if v == "S":
            res.append(s)
        elif isinstance(v, str) and v.startswith("S/"):
            res.append(s // int(v[2:]))
        else:
            res.append(int(v))
    return tuple(res)


def cases(req: dict) -> list[tuple]:
    """INPUTS entries: one per chunk length, each a tuple of every tensor input's per-device shape."""
    ins = req["interface"]["inputs"]
    return [tuple(concrete(i["per_device_shape"], s) for i in ins) for s in req["interface"]["chunks"]]


def arg(name: str) -> str:
    """A boundary name as a Python parameter name ('in' -> 'in_')."""
    n = re.sub(r"\W", "_", name)
    return n + "_" if keyword.iskeyword(n) or not n.isidentifier() else n


# ---------------------------------------------------------------- generator
FEATURE_SPEC = '''"""Op-gen feature spec for {op} ({model} {tasks}): the model's own per-device calls, one case per chunk length.

Written by models/demos/common/bringup/plan/op_request.py from request.yaml (`refresh` rewrites it). Strings, not
ttnn enums: op-export translates them for the op-gen suite. Every TARGET axis has one value (the bring-up's Phase 0).
"""

TARGET = {target!r}

# Per-device input shapes in signature order: {names}
INPUTS = {inputs}

INVALID = []

LOOSE_CASES = []
'''

REFERENCE = '''"""Pure-torch reference of {model} {bt}.{step} for op-gen (op request {op}).

No model imports: op-gen's helpers copy this function. Compute in fp32 and return the output in the input's dtype.
The checker runs it against the model's CPU step on a random input (bind.py supplies the weights and scalars).

The model's CPU step, for reference ({where}):
{source}
"""

from __future__ import annotations

import torch


def pytorch_{op}({args}):
    """<<AGENT: the step's math>>"""
    raise NotImplementedError("<<AGENT: write the standalone pure-torch reference>>")
'''

BIND = '''"""Model-side binding of op request {op} (checked, never exported): what pytorch_{op} takes beyond the step's inputs
({inputs}), taken from the model's CPU reference for one layer. Weights also go in request.yaml
interface.inputs (kind: weight), scalars in interface.params, with the same values.
"""


def arguments(ref, layer, ctx):
    """-> (extra positional tensors after the step inputs, keyword scalars)."""
    return [], {{}}
'''

PROMPT = """# golden: {op}
<<AGENT: the math in one paragraph and a formula. What the step computes, in terms of its inputs.>>

The function signature must be:

    {op}(
{sig}
        *,
{params}        compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # Keyword-only
    ) -> ttnn.Tensor

This supports these call patterns:
    {op}({call})    # the model's call, one per device (single-device op)

<<AGENT: any other call pattern the model uses, one line each; delete this line if none.>>

The model ({model}, {tasks}) calls it with these per-device shapes ({names}), one per chunk length:
{shape_lines}
Inputs: {input_lines}. Output: {output_line}.

The op MUST follow the registry model (see eval/op_template.py):

- Declare INPUT_TAGGERS, SUPPORTED, EXCLUSIONS inline in the op file. Phase 0 has no shape-derived axes:
  INPUT_TAGGERS = {{}}.
- Implement validate() that raises a support-refusal from ttnn.operations._op_contract (UnsupportedAxisValue /
  ExcludedCell) for cells outside the declared support contract: SUPPORTED per-axis, then EXCLUSIONS (cell-level).
  Do NOT declare or check INVALID; INVALID is a test-harness concept that lives in feature_spec.py; the test
  runner skips INVALID cells before they reach the op.
- The public entry point calls validate() as its first line.

Phase 0 SUPPORTED (what the bring-up needs; the golden suite pins each axis to this one value):

{supported}

Accuracy: PCC >= {pcc} and relative RMS error <= {rel_rms} against the fp32 reference (the model's frozen component
test gates the step at PCC {pcc} on real data).

## Rules

- <<AGENT: "When X: MUST / MUST NOT ..." lines: layout, alignment, precision, what the entry point must not do on
  the host.>>

Import path: from ttnn.operations.{op} import {op}
"""


def _torch_dtype_name(t, target_dtype: str) -> str:
    return target_dtype if t.is_floating_point() else "uint32"


def _tolerance(task: dict) -> dict:
    """From the component gate's threshold (">= 0.99"): pcc, and the relative RMS a pure-noise error of that PCC has."""
    conds = [str(c).split() for c in (task["gate"].get("metrics") or {}).values()]
    pcc = next((float(v) for op, v in conds if op in (">=", ">")), 0.99)
    return {"pcc": pcc, "rel_rms": round(max(0.02, math.sqrt(2 * (1 - pcc))), 3)}


def _source_of(fn) -> tuple[str, str]:
    try:
        return textwrap.indent(textwrap.dedent(inspect.getsource(fn)), "    "), f"{inspect.getsourcefile(fn)}"
    except (OSError, TypeError):
        return "    (source not available)", "unknown"


def generate(spec, tid: str, op: str, force: bool = False) -> Path:
    """Write <bringup_dir>/op_requests/<op>/ for task tid: everything mechanical filled, the rest marked <<AGENT."""
    import torch

    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import parse_layers
    from models.demos.common.bringup.testing.harness import component_golden

    if not OP_NAME.match(op):
        raise ValueError(f"op name {op!r}: one snake_case token")
    led = Ledger(spec.bringup_dir)
    task = led.task(tid)
    if not deferrable(task):
        raise ValueError(f"{tid} is not a component task of the implement step; only those may be deferred")
    b = task["brief"]
    bt, step, layer = b["block_type"], b["step"], int(b["layer"])
    d = request_dir(spec, op)
    if d.exists() and not force:
        raise FileExistsError(f"{d} exists (use --force to rewrite it)")
    d.mkdir(parents=True, exist_ok=True)

    ref = spec.hooks().reference(spec, layers=[layer], dtype=torch.float32)
    st = next(s for s in ref.block_graph(layer) if s.name == step)
    g, c = component_golden(spec)
    gl = g.layer(c, layer)
    target_dtype = {"bf16": "bfloat16", "fp32": "float32", "bf8": "bfloat8_b"}.get(
        str(spec.get("target.dtype", "bf16")), "bfloat16"
    )

    def tensor_entry(name, t, kind):
        shape = [int(n) for n in t.shape]
        if g.chunk in shape:  # the sequence dim: the first one of the chunk's length
            shape[shape.index(g.chunk)] = "S"
        e = {
            "name": name,
            "kind": kind,
            "shape": shape,
            "dtype": _torch_dtype_name(t, target_dtype),
            "layout": "TILE" if t.is_floating_point() else "ROW_MAJOR",
            "placement": "replicate",
        }
        if not t.is_floating_point():
            e["range"] = [0, int(t.max().item()) + 1]
        e["per_device_shape"] = per_device(shape, e["placement"], spec.mesh)
        return e

    inputs = [tensor_entry(n, gl[n], "activation") for n in st.inputs]
    out = tensor_entry(st.output, gl[st.output], "output")
    info = spec.data["block_types"][bt]
    layers = [i for i in parse_layers(info["layers"], spec.num_layers) if i in spec.layers()]
    req = {
        "op": op,
        "model": spec.model,
        "tasks": [tid],
        "component": {"block_type": bt, "step": step, "kind": st.kind, "stateful": bool(st.stateful)},
        "layers": layers,
        "representative_layer": layer,
        "status": "draft",
        "evidence": {
            "searched": [f"{MARK}: repo map rows and greps you checked, one per item>>"],
            "tried": [
                {"what": f"{MARK}: an op, composition or fork>>", "outcome": f"{MARK}: the gate or test outcome>>"}
            ],
            "why_not_fork": f"{MARK}: why no fork of an existing op fits>>",
        },
        "interface": {
            "inputs": inputs,
            "output": out,
            "chunks": chunks(spec),
            "params": {},
            "memory_layout": "INTERLEAVED",
        },
        "tolerance": _tolerance(task),
        "acceptance": {
            "test": (task.get("tests") or [None])[0],
            "golden": (task.get("freeze_extra") or [None])[0],
            "metric": next(iter(task["gate"].get("metrics") or {}), None),
            "threshold": next(iter((task["gate"].get("metrics") or {}).values()), None),
        },
    }
    save(d, req)
    src, where = _source_of(ref.component(layer, step))
    args = ", ".join(arg(i["name"]) for i in inputs)
    (d / "reference.py").write_text(
        REFERENCE.format(model=spec.model, bt=bt, step=step, op=op, where=where, source=src, args=args)
    )
    (d / "bind.py").write_text(BIND.format(op=op, inputs=args))
    refresh(spec, d)
    (d / "op_prompt.txt").write_text(_prompt(load(d)))
    return d


def _target(req: dict) -> dict:
    io = req["interface"]
    first = io["inputs"][0]
    return {
        "dtype": [first["dtype"]],
        "layout": [first["layout"]],
        "memory_layout": [io.get("memory_layout", "INTERLEAVED")],
    }


def refresh(spec, d: Path) -> None:
    """Recompute per-device shapes from each tensor's placement and rewrite feature_spec.py (after the agent edits
    request.yaml's interface: placements, weight inputs). The prompt is the agent's once written; it is not touched."""
    d = Path(d)
    req = load(d)
    io = req["interface"]
    for t in io["inputs"] + [io["output"]]:
        t["per_device_shape"] = per_device(t["shape"], t.get("placement", "replicate"), spec.mesh)
    save(d, req)
    names = ", ".join(i["name"] for i in io["inputs"])
    inputs = "[\n" + "".join(f"    {c!r},\n" for c in cases(req)) + "]"
    (d / "feature_spec.py").write_text(
        FEATURE_SPEC.format(
            op=req["op"],
            model=req["model"],
            tasks=", ".join(req["tasks"]),
            target=_target(req),
            names=names,
            inputs=inputs,
        )
    )


def _prompt(req: dict) -> str:
    io, target = req["interface"], _target(req)
    ins = io["inputs"]
    tensor_line = lambda t: f"{t['name']} {t['dtype']} {t['layout']}"  # noqa: E731
    sig = "".join(f"        {arg(i['name'])}: ttnn.Tensor,\n" for i in ins).rstrip("\n")
    params = "".join(f"        {k}: {type(v).__name__} = {v!r},\n" for k, v in io.get("params", {}).items())
    shape_lines = "".join(f"- {c}\n" for c in cases(req))
    supported = "\n".join(f"- {k}: {v[0]}" for k, v in target.items())
    return PROMPT.format(
        op=req["op"],
        sig=sig,
        params=params,
        call=", ".join(arg(i["name"]) for i in ins) + "".join(f", {k}={v!r}" for k, v in io.get("params", {}).items()),
        model=req["model"],
        tasks=", ".join(req["tasks"]),
        names=", ".join(i["name"] for i in ins),
        shape_lines=shape_lines,
        input_lines="; ".join(tensor_line(i) for i in ins),
        output_line=f"{tensor_line(io['output'])}, per-device shape {io['output']['per_device_shape']}",
        supported=supported,
        pcc=req["tolerance"]["pcc"],
        rel_rms=req["tolerance"]["rel_rms"],
    )


# ---------------------------------------------------------------- checker
def _import_file(path: Path, name: str):
    """Execute a request file as a fresh module. No bytecode cache: an agent's rewrite within the same second and
    of the same size would otherwise run the stale .pyc."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


def impure_imports(path: Path) -> list[str]:
    """Imports in reference.py outside torch / math / numpy / typing: op-gen's helpers may not import the model."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        mods = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        bad += [m for m in mods if m.split(".")[0] not in PURE_IMPORTS]
    return bad


def _random_input(entry: dict, s: int, gen):
    import torch

    shape = concrete(entry["shape"], s)
    if entry.get("dtype", "bfloat16").startswith(("uint", "int")):
        lo, hi = entry.get("range", [0, 16])
        return torch.randint(int(lo), int(hi), shape, generator=gen)
    return torch.randn(shape, generator=gen)


def _check_reference(spec, d: Path, req: dict) -> list[str]:
    """pytorch_<op> reproduces the model's CPU step on a random input of CHECK_ROWS rows."""
    import torch

    from models.demos.common.bringup.core import metrics

    op, comp, io = req["op"], req["component"], req["interface"]
    errs = [
        f"reference.py imports {m} (pure torch only: op-gen's helpers must not import the model)"
        for m in impure_imports(d / "reference.py")
    ]
    if errs:
        return errs
    try:
        fn = getattr(_import_file(d / "reference.py", f"_opreq_ref_{op}"), f"pytorch_{op}")
        bind = _import_file(d / "bind.py", f"_opreq_bind_{op}")
    except Exception as e:  # noqa: BLE001 - any import problem is a rejection
        return [f"reference.py / bind.py do not import: {e!r}"]
    layer = int(req.get("representative_layer", spec.representative_layer(comp["block_type"])))
    ref = spec.hooks().reference(spec, layers=[layer], dtype=torch.float32)
    steps = {s.name: s for s in ref.block_graph(layer)}
    st = steps.get(comp["step"])
    if st is None:
        return [f"component step {comp['step']!r} is not in the {comp['block_type']} graph ({list(steps)})"]
    acts = [i for i in io["inputs"] if i.get("kind") == "activation"]
    if [i["name"] for i in acts] != list(st.inputs):
        return [f"activation inputs {[i['name'] for i in acts]} differ from the step's inputs {list(st.inputs)}"]
    gen = torch.Generator().manual_seed(0)
    x = [_random_input(i, CHECK_ROWS, gen) for i in acts]
    s = CHECK_ROWS
    want = ref.component(layer, st.name)(ref.chunk_context(layer, 0, s, ref.new_state(s)), *[t.clone() for t in x])
    try:
        extra, params = bind.arguments(ref, layer, ref.chunk_context(layer, 0, s, ref.new_state(s)))
        got = fn(*[t.clone() for t in x], *extra, **params)
    except Exception as e:  # noqa: BLE001
        return [f"pytorch_{op} raised {e!r} on a random input"]
    weights = [i for i in io["inputs"] if i.get("kind") == "weight"]
    if [list(w.shape) for w in extra] != [list(concrete(w["shape"], s)) for w in weights]:
        errs.append(
            f"bind.py weights {[list(w.shape) for w in extra]} differ from interface weights {[w['shape'] for w in weights]}"
        )
    if dict(params) != dict(io.get("params") or {}):
        errs.append(f"bind.py params {params} differ from interface.params {io.get('params')}")
    if tuple(got.shape) != tuple(want.shape):
        return errs + [f"pytorch_{op} output shape {tuple(got.shape)} != the step's {tuple(want.shape)}"]
    if want.is_floating_point():
        p = metrics.pcc(got.float(), want.float())
        diff = (got.float() - want.float()).abs().max().item()
        tol = 1e-3 * max(1.0, want.float().abs().max().item())
        if p < 0.99999 or diff > tol:
            errs.append(f"pytorch_{op} does not reproduce the model's step: pcc {p:.6f}, max abs diff {diff:.3g}")
    elif not torch.equal(got, want):
        errs.append(f"pytorch_{op} does not reproduce the model's step (integer output differs)")
    return errs


def _check_evidence(ev: dict) -> list[str]:
    errs = []
    if not [s for s in ev.get("searched") or [] if str(s).strip()]:
        errs.append("evidence.searched is empty: list the repo map rows and greps you checked")
    tried = ev.get("tried") or []
    if not tried:
        errs.append("evidence.tried is empty: list each op, composition or fork you tried and its outcome")
    for k, t in enumerate(tried):
        if not isinstance(t, dict) or not str(t.get("what", "")).strip():
            errs.append(f"evidence.tried[{k}] has no 'what'")
        elif len(str(t.get("outcome", "")).strip()) < MIN_OUTCOME:
            errs.append(
                f"evidence.tried[{k}] ({t['what']}): quote the gate or test outcome (at least {MIN_OUTCOME} chars)"
            )
    if len(str(ev.get("why_not_fork", "")).strip()) < MIN_WHY_NOT_FORK:
        errs.append(
            f"evidence.why_not_fork: say why no fork of an existing op fits (at least {MIN_WHY_NOT_FORK} chars)"
        )
    return errs


def check(spec, d: Path) -> list[str]:
    """Every reason the request in d cannot defer its task (empty = valid)."""
    from models.demos.common.bringup.core.ledger import Ledger

    d = Path(d)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        return [f"missing {', '.join(missing)} in {d}"]
    try:
        req = load(d)
    except yaml.YAMLError as e:
        return [f"request.yaml does not parse: {e}"]
    errs = [
        f"request.yaml: missing '{k}'"
        for k in (
            "op",
            "model",
            "tasks",
            "component",
            "layers",
            "status",
            "evidence",
            "interface",
            "tolerance",
            "acceptance",
        )
        if k not in req
    ]
    if errs:
        return errs
    op = req["op"]
    if not OP_NAME.match(str(op)):
        errs.append(f"op {op!r}: one snake_case token")
    if d.name != op:
        errs.append(f"op {op!r} differs from its folder {d.name!r}")
    if req["model"] != spec.model:
        errs.append(f"model {req['model']!r} is not this bring-up ({spec.model})")
    if req["status"] not in STATUSES:
        errs.append(f"status {req['status']!r} not in {STATUSES}")
    if req["status"] != "delivered":
        for p in (f"ttnn/ttnn/operations/{op}", f"ttnn/ttnn/bringup/{op}"):
            if (spec.repo / p).exists():
                errs.append(f"{p} exists: an op of this name is already in TTNN; use it or pick another name")
    tasks = Ledger(spec.bringup_dir).tasks()
    for tid in req["tasks"] or ["(none)"]:
        t = tasks.get(tid)
        if t is None or not deferrable(t):
            errs.append(f"task {tid}: not a component task of the implement step")
        elif (t["brief"]["block_type"], t["brief"]["step"]) != (
            req["component"].get("block_type"),
            req["component"].get("step"),
        ):
            errs.append(f"task {tid} is {t['brief']['block_type']}.{t['brief']['step']}, not the request's component")
    marked = [f for f in FILES if MARK in (d / f).read_text()]
    if marked:
        errs.append(f"generator markers ({MARK} ...) left in {', '.join(marked)}")
    errs += _check_evidence(req["evidence"])
    lines = [x for x in (d / "op_prompt.txt").read_text().splitlines() if x.strip()]
    if not lines or lines[0].strip() != f"# golden: {op}":
        errs.append(f"op_prompt.txt: the first line must be '# golden: {op}'")
    if not lines or lines[-1].strip() != f"Import path: from ttnn.operations.{op} import {op}":
        errs.append(f"op_prompt.txt: the last line must be 'Import path: from ttnn.operations.{op} import {op}'")
    if "## Rules" not in (d / "op_prompt.txt").read_text():
        errs.append("op_prompt.txt: no '## Rules' section")
    try:
        fs = _import_file(d / "feature_spec.py", f"_opreq_fs_{op}")
        multi = [k for k, v in fs.TARGET.items() if not isinstance(v, list) or len(v) != 1]
        if multi:
            errs.append(f"feature_spec.py: TARGET axes {multi} must have exactly one value")
        n_in = len(req["interface"]["inputs"])
        cs = list(fs.INPUTS) + [c["inputs"] for c in getattr(fs, "LOOSE_CASES", [])]
        if not cs:
            errs.append("feature_spec.py: no case (INPUTS or LOOSE_CASES)")
        if any(len(c) != n_in for c in cs):
            errs.append(f"feature_spec.py: every case needs {n_in} input shapes (interface.inputs)")
        if fs.INVALID != []:
            errs.append("feature_spec.py: INVALID must be []")
    except Exception as e:  # noqa: BLE001
        errs.append(f"feature_spec.py does not import: {e!r}")
    if errs:
        return errs
    return _check_reference(spec, d, req)


def set_status(d: Path, status: str, **extra) -> dict:
    req = load(d)
    req["status"] = status
    req.update(extra)
    save(d, req)
    return req


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup.plan.op_request")
    ap.add_argument("command", choices=["new", "refresh", "check"])
    ap.add_argument("target", help="new: op name; refresh / check: the request folder (or op name)")
    ap.add_argument("--task", help="new: the component task (C.<block type>.<step>)")
    ap.add_argument("--spec")
    ap.add_argument("--force", action="store_true", help="new: rewrite an existing request")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.command == "new":
        d = generate(spec, a.task, a.target, force=a.force)
        print(f"wrote {d}: fill every {MARK} marker (evidence, prompt prose, reference.py, bind.py), then `check`")
        return 0
    d = Path(a.target) if Path(a.target).exists() else request_dir(spec, a.target)
    if a.command == "refresh":
        refresh(spec, d)
        print(f"refreshed {d / 'feature_spec.py'}")
        return 0
    errs = check(spec, d)
    for e in errs:
        print(f"REJECTED {e}")
    print("valid" if not errs else f"{len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

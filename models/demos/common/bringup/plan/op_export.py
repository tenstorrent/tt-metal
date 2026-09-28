# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Op requests out to op-gen and generated ops back in (F46). Commands: ``$B op-export``, ``$B op-ready``.

export(spec, op, codegen_root): the approved request becomes op-gen's input in the codegen tree
(tt_metal/third_party/tt_ops_code_gen): ``eval/prompts/<op>.txt`` and the golden suite ``eval/golden_tests/<op>/``
(feature_spec.py with ttnn enums, helpers.py with the request's standalone reference, axes.py, test_golden.py,
conftest.py, test_regression.py; the rms_norm / fake_op idioms). It marks the request exported and returns the steps
the owner takes next. It never commits, pushes or launches anything: launching op-gen is the owner's call.

ready(spec, ops, sources): a delivered op (op-gen's ``ttnn/ttnn/operations/<op>/``) becomes the Python bring-up op
``ttnn.bringup.<op>``: copied into ttnn/ttnn/bringup/<op>/ (imports and kernel paths moved with it), registered in
PYTHON_OPS, with a CHANGELOG.md and an INDEX.md row. The request is marked delivered, its deferred tasks go back to TODO
with a brief that names the op, and everything downstream of them is reset (runs.rerun_from), so ``orchestrator
resume`` redoes the swaps, M.1, the ladder, K.1, X.*, P.* and O.1.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import time
from pathlib import Path

from models.demos.common.bringup.plan import op_request as OR

CODEGEN = "tt_metal/third_party/tt_ops_code_gen"
BRINGUP_OPS = "ttnn/ttnn/bringup"
HEADER = "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
DTYPES = {
    "bfloat16": "ttnn.bfloat16",
    "float32": "ttnn.float32",
    "bfloat8_b": "ttnn.bfloat8_b",
    "uint32": "ttnn.uint32",
    "int32": "ttnn.int32",
}
LAYOUTS = {"TILE": "ttnn.TILE_LAYOUT", "ROW_MAJOR": "ttnn.ROW_MAJOR_LAYOUT"}
MEMORY = {
    m: f"ttnn.TensorMemoryLayout.{m}" for m in ("INTERLEAVED", "HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED")
}
AXES = {"dtype": DTYPES, "layout": LAYOUTS, "memory_layout": MEMORY}


class Enum(str):
    """A ttnn enum expression, printed without quotes."""

    def __repr__(self):
        return str(self)


def enum(axis: str, v):
    return Enum(AXES[axis][v]) if axis in AXES and v in AXES[axis] else v


# ---------------------------------------------------------------- export
FEATURE_SPEC = (
    HEADER
    + '''
"""{op} feature spec: TARGET universe + golden-test INPUTS, from the bring-up op request {model}/{op}.

Every TARGET axis has one value: the Phase 0 corner the model needs. INPUTS are the model's own per-device calls,
one per chunk length (signature order: {names}). No INVALID cells.
"""

import ttnn


TARGET = {target}


INVALID = []


INPUTS = {inputs}


LOOSE_CASES = {loose}
'''
)

HELPERS = (
    HEADER
    + '''
"""Shared helpers for {op} golden tests (registry model), from the bring-up op request {model}/{op}.

Provides:
- pytorch_{op}: the reference, the model's own CPU step as a standalone function (fp32 inside).
- create_ttnn_input_tensor: the one torch -> ttnn chokepoint.
- TOLERANCES: {{dtype: (pcc, rel_rms)}}, from the model's frozen component test threshold.
- run_{op}: builds random inputs for one case and checks the op against the reference.
"""

from __future__ import annotations

import torch
import ttnn
{extra_imports}
from eval.metrics import CheckOutputError, check_output  # noqa: F401  (re-export)
from eval.oom import oom_guard

from eval.golden_tests.{op}.axes import observed as {op}  # type: ignore


# --- Reference (the op request's reference.py) ----------------------------

{reference}


_TORCH_DTYPE = {{
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}}

# Tensor inputs in signature order. The first carries the dtype / layout / memory_layout axes; the others (weights,
# indices) keep the format the model uses.
INPUT_SPECS = {input_specs}
PARAMS = {params}
OUTPUT_DTYPE = {output_dtype}

TOLERANCES = {tolerances}


def create_ttnn_input_tensor(tensor, device, *, dtype, layout,
                             memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, memory_config=None):
    """Single chokepoint for torch.Tensor -> ttnn.Tensor (DRAM interleaved in Phase 0)."""
    if memory_config is None:
        if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            from eval.sharding import auto_shard_config
            memory_config = auto_shard_config(list(tensor.shape), memory_layout, layout=layout, dtype=dtype, device=device)
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _random(shape, spec, scale):
    if spec["kind"] == "int":
        lo, hi = spec["range"]
        return torch.randint(lo, hi, tuple(shape), dtype=torch.int32)
    return torch.randn(tuple(shape), dtype=torch.float32) * scale


def run_{op}(inputs, *, dtype, layout, memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
             device, extras=None, **_):
    """Build the case's tensors, dispatch {op}, check against pytorch_{op}. Raises CheckOutputError on a miss.

    `inputs` is a tuple of per-device shapes in signature order. extras: "pcc_threshold" overrides the PCC gate,
    "scale" scales the float inputs (test_regression.py).
    """
    extras = extras or {{}}
    torch.manual_seed(0)
    torch_inputs, ttnn_inputs = [], []
    for k, (shape, spec) in enumerate(zip(inputs, INPUT_SPECS)):
        t_dtype, t_layout = (dtype, layout) if k == 0 else (spec["dtype"], spec["layout"])
        t = _random(shape, spec, extras.get("scale", 1.0))
        if spec["kind"] == "float":
            t = t.to(_TORCH_DTYPE[t_dtype])
        torch_inputs.append(t)
        with oom_guard("input"):
            ttnn_inputs.append(create_ttnn_input_tensor(
                t, device, dtype=t_dtype, layout=t_layout,
                memory_layout=memory_layout if k == 0 else ttnn.TensorMemoryLayout.INTERLEAVED))
    expected = pytorch_{op}(*torch_inputs, **PARAMS)
    with oom_guard("op"):
        ttnn_output = {op}(*ttnn_inputs, **PARAMS)
    tol = TOLERANCES.get(dtype, next(iter(TOLERANCES.values())))
    if extras.get("pcc_threshold") is not None:
        tol = (extras["pcc_threshold"], tol[1])
    check_output(ttnn_output, expected, shape=list(expected.shape), dtype=OUTPUT_DTYPE or dtype,
                 expected_layout=layout, tolerance=tol)
'''
)

AXES_PY = (
    HEADER
    + '''
"""Observe-only runtime axis tagging for {op} (the rms_norm idiom): classify_call rebuilds the registry cell from a
real call, observe() records it, observed() wraps the op so golden and regression tests tag uniformly. INPUT_TAGGERS
come from the op; the op still owns the support gate (validate raises)."""

from __future__ import annotations

import warnings

from eval import metrics_plugin
from eval.feature_matrix import apply_input_taggers
from ttnn.operations.{op} import {op} as _raw, INPUT_TAGGERS  # type: ignore


def classify_call(*tensors, **kwargs):
    """The cell of a {op} call: the first tensor's dtype, layout and memory layout, plus the op's taggers."""
    x = tensors[0]
    axes = {{
        "dtype": x.dtype,
        "layout": x.layout,
        "memory_layout": x.memory_config().memory_layout,
    }}
    axes.update(apply_input_taggers(INPUT_TAGGERS, tuple(list(t.shape) for t in tensors), axes))
    return axes


def observe(axes):
    metrics_plugin.record_axes(axes)


def observed(*args, **kwargs):
    """Tag the call, then dispatch the real op inside the perf window. Tagging never breaks dispatch."""
    try:
        observe(classify_call(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{op} classify_call failed (row will be untagged): {{exc!r}}")
    with metrics_plugin.op_window():
        return _raw(*args, **kwargs)
'''
)

TEST_GOLDEN = (
    HEADER
    + '''
"""Auto-parameterized golden tests for {op} (registry model); per-op wiring around eval/golden_harness.py."""

from __future__ import annotations

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.{op}.feature_spec import (
    INPUTS, INVALID, LOOSE_CASES, TARGET,
)
from eval.golden_tests.{op}.helpers import run_{op}
from ttnn.operations.{op} import (  # type: ignore
    EXCLUSIONS, INPUT_TAGGERS, SUPPORTED,
)


@pytest.mark.parametrize(
    "inputs,axes",
    parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID),
)
def test_op(inputs, axes, device):
    run_{op}(inputs, device=device, **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    parametrize_loose_cases(
        LOOSE_CASES, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID,
    ),
)
def test_op_loose(inputs, axes, extras, device):
    run_{op}(inputs, device=device, extras=extras, **axes)
'''
)

CONFTEST = (
    HEADER
    + '''
"""Pytest configuration for {op} golden tests: the numerics marker, and one device per module."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numerics: numerical-stability / data-distribution regression tests — "
        "not registry-driven, run unconditionally in the full suite",
    )


def pytest_collection_modifyitems(items):
    # Open the device once per module for this single-device op's tests.
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
'''
)

TEST_REGRESSION = (
    HEADER
    + '''
"""Numerical-stability regression tests for {op}: the model's first call at small and large input magnitude.
Not registry-driven; tagged @pytest.mark.numerics."""

from __future__ import annotations

import pytest

from eval.golden_tests.{op}.feature_spec import INPUTS, TARGET
from eval.golden_tests.{op}.helpers import run_{op}

_AXES = {{k: v[0] for k, v in TARGET.items()}}


@pytest.mark.numerics
@pytest.mark.parametrize("scale", [0.01, 10.0], ids=["small", "large"])
def test_magnitude(scale, device):
    run_{op}(INPUTS[0], device=device, extras={{"scale": scale}}, **_AXES)
'''
)


def _reference_parts(path: Path) -> tuple[str, str]:
    """reference.py without its docstring and its torch / __future__ imports: (other imports, the code)."""
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    imports, code = [], []
    for n in tree.body:
        seg = "\n".join(
            lines[min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1 : n.end_lineno]
        )
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if not all(m in ("torch", "__future__") for m in mods):
                imports.append(seg)
        elif not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            code.append(seg)
    return "\n".join(imports), "\n\n\n".join(code)


def _pretty(v, indent="    ") -> str:
    """A list / dict literal one element per line (repr keeps ttnn enums bare)."""
    if isinstance(v, list):
        return "[\n" + "".join(f"{indent}{x!r},\n" for x in v) + "]"
    return repr(v)


def suite_files(spec, op: str) -> dict[str, str]:
    """eval/... path -> text of op-gen's prompt and golden suite for the request."""
    d = OR.request_dir(spec, op)
    req = OR.load(d)
    io = req["interface"]
    fs = OR._import_file(d / "feature_spec.py", f"_opreq_export_{op}")
    target = {k: [enum(k, x) for x in v] for k, v in fs.TARGET.items()}
    loose = [
        {k: (v if k in ("inputs", "extras") else enum(k, v)) for k, v in c.items()}
        for c in getattr(fs, "LOOSE_CASES", [])
    ]
    specs = [
        {
            "name": i["name"],
            "kind": "int" if i["dtype"].startswith(("int", "uint")) else "float",
            **({"range": tuple(i.get("range", [0, 16]))} if i["dtype"].startswith(("int", "uint")) else {}),
            "dtype": enum("dtype", i["dtype"]),
            "layout": enum("layout", i["layout"]),
        }
        for i in io["inputs"]
    ]
    first = enum("dtype", io["inputs"][0]["dtype"])
    tol = {first: (req["tolerance"]["pcc"], req["tolerance"]["rel_rms"])}
    extra_imports, reference = _reference_parts(d / "reference.py")
    fmt = dict(op=op, model=req["model"], names=", ".join(i["name"] for i in io["inputs"]))
    g = f"eval/golden_tests/{op}"
    return {
        f"eval/prompts/{op}.txt": (d / "op_prompt.txt").read_text(),
        f"{g}/__init__.py": HEADER,
        f"{g}/feature_spec.py": FEATURE_SPEC.format(
            **fmt, target=repr(target), inputs=_pretty(list(fs.INPUTS)), loose=_pretty(loose)
        ),
        f"{g}/helpers.py": HELPERS.format(
            **fmt,
            extra_imports=(extra_imports + "\n") if extra_imports else "",
            reference=reference,
            input_specs=_pretty(specs),
            params=repr(io.get("params") or {}),
            output_dtype=repr(enum("dtype", io["output"]["dtype"])),
            tolerances=repr(tol),
        ),
        f"{g}/axes.py": AXES_PY.format(**fmt),
        f"{g}/test_golden.py": TEST_GOLDEN.format(**fmt),
        f"{g}/conftest.py": CONFTEST.format(**fmt),
        f"{g}/test_regression.py": TEST_REGRESSION.format(**fmt),
    }


def export(spec, op: str, codegen_root: Path | None = None) -> tuple[list[Path], list[str]]:
    """Write op-gen's input for an approved request into the codegen tree. -> (files written, the owner's next steps)."""
    from models.demos.common.bringup.plan.approvals import op_request_approved

    d = OR.request_dir(spec, op)
    errs = OR.check(spec, d)
    if errs:
        raise ValueError(f"op request {op} does not pass its check:\n" + "\n".join(errs))
    if not op_request_approved(spec, op):
        raise PermissionError(
            f"op request {op} is not approved (or changed since): the owner approves it first "
            f"(python -m models.demos.common.bringup approve op-request {op} --spec {spec.path})"
        )
    root = Path(codegen_root) if codegen_root else spec.repo / CODEGEN
    if not (root / "eval").is_dir():
        raise FileNotFoundError(f"{root} is not an op-gen tree (no eval/)")
    written = []
    for rel, text in suite_files(spec, op).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written.append(p)
    OR.set_status(d, "exported", exported={"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "to": str(root)})
    try:
        sub = str(root.resolve().relative_to(spec.repo))
    except ValueError:
        sub = str(root)
    steps = [
        f"cd {sub} && git checkout -b bringup/{spec.model}/{op} && git add eval/prompts/{op}.txt eval/golden_tests/{op} "
        f'&& git commit -m "Golden suite and prompt for {op} (bring-up op request {spec.model}/{op})"',
        f"git -C {sub} push -u origin bringup/{spec.model}/{op}   # the owner's call",
        f'git add {sub} && git commit -m "Bump tt_ops_code_gen: op-gen request {op}" -- {sub}   # the gitlink, in tt-metal',
        "git push   # the tt-metal branch; run_eval.py clones the pushed branch with its submodule (the owner's call)",
        f"python3 .claude/eval/run_eval.py .claude/eval/prompts/{op}.txt --runs 1   # launch op-gen (the owner's call; "
        f".claude is a checkout of tt_ops_code_gen: bring it to the commit above first)",
        f"when it delivers: python -m models.demos.common.bringup op-ready {op} --from <clone>/ttnn/ttnn/operations/{op} "
        f"--spec {spec.path}, then orchestrator resume",
    ]
    return written, steps


# ---------------------------------------------------------------- ready
def _git(cwd: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def register_python_op(init: Path, name: str, folder: str, fn: str) -> None:
    """Add name -> (folder, fn) to PYTHON_OPS in ttnn/ttnn/bringup/__init__.py, rewriting only that assignment."""
    src = init.read_text()
    node = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PYTHON_OPS" for t in n.targets)
    )
    ops = ast.literal_eval(node.value)
    if name in ops:
        raise ValueError(f"{name} is already in PYTHON_OPS")
    ops[name] = (folder, fn)
    body = "PYTHON_OPS = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in ops.items()) + "}"
    lines = src.splitlines()
    init.write_text("\n".join(lines[: node.lineno - 1] + [body] + lines[node.end_lineno :]) + "\n")


def _move_paths(dst: Path, op: str) -> list[str]:
    """ttnn.operations.<op> -> ttnn.bringup.<op> (imports) and ttnn/operations/<op> -> ttnn/bringup/<op> (kernel
    paths) in every text file of the copy."""
    changed = []
    for f in sorted(dst.rglob("*")):
        if not f.is_file():
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        new = re.sub(rf"\bttnn\.operations\.{op}\b", f"ttnn.bringup.{op}", text)
        new = new.replace(f"ttnn/operations/{op}", f"ttnn/bringup/{op}")
        if new != text:
            f.write_text(new)
            changed.append(str(f.relative_to(dst)))
    return changed


def _source(sources: list[Path], op: str) -> Path:
    for s in sources:
        s = Path(s)
        for cand in (s / op, s):
            if cand.name == op and (cand / "__init__.py").exists():
                return cand
    raise FileNotFoundError(f"no generated {op}/ (with __init__.py) under {[str(s) for s in sources]}")


def ready(spec, ops: list[str], sources: list[Path], bringup_ops: Path | None = None) -> dict:
    """Bring delivered ops in as ttnn.bringup.<op> and reset the deferred tasks and their dependents. -> summary."""
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.runs import rerun_from

    led = Ledger(spec.bringup_dir)
    bops = Path(bringup_ops) if bringup_ops else spec.repo / BRINGUP_OPS
    plan = []
    for op in ops:  # check everything before changing anything
        d = OR.request_dir(spec, op)
        if not (d / "request.yaml").exists():
            raise FileNotFoundError(f"no op request {op}")
        req = OR.load(d)
        if req["status"] != "exported":
            raise ValueError(f"op request {op} is {req['status']}; only an exported request can be delivered")
        if (bops / op).exists():
            raise FileExistsError(f"{bops / op} exists")
        plan.append((op, d, req, _source(sources, op)))
    out = {"ops": [], "reset": [], "paths": []}
    for op, d, req, src in plan:
        dst = bops / op
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        moved = _move_paths(dst, op)
        register_python_op(bops / "__init__.py", op, op, op)
        clone = Path(_git(src, "rev-parse", "--show-toplevel") or src)
        sha = _git(src, "rev-parse", "--short=11", "HEAD") or "unknown"
        req_rel = str(d.relative_to(spec.repo))
        (dst / "CHANGELOG.md").write_text(
            f"# {op} (ttnn.bringup.{op})\n\n"
            f"- Source: generated by op-gen from op request {spec.model}/{op} (`{req_rel}`)\n"
            f"- Source clone: `{clone}` @ `{sha}`\n"
            f"- Python: `ttnn.bringup.{op}` (was `ttnn.operations.{op}`; PYTHON_OPS in `__init__.py`)\n"
            f"- Used by: {spec.model} ({', '.join(req['tasks'])})\n\n"
            f"Mechanical changes on copy (op_export.py): imports and kernel paths moved from ttnn/operations/{op} to "
            f"ttnn/bringup/{op}" + (f" ({', '.join(moved)})" if moved else "") + ".\n\n"
            "## Changes\n\n"
            "<!-- One entry per change, newest last (none yet: the op is as op-gen delivered it). -->\n"
        )
        idx = bops / "INDEX.md"
        if idx.exists():
            idx.write_text(
                idx.read_text().rstrip("\n")
                + f"\n| `{op}` | op-gen, op request `{spec.model}/{op}` (clone `{clone}` @ `{sha}`) | none (as generated) | "
                f"{spec.model} |\n"
            )
        OR.set_status(
            d,
            "delivered",
            delivered={
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": str(src),
                "commit": sha,
                "op": f"ttnn.bringup.{op}",
            },
        )
        for tid in req["tasks"]:
            b = dict(led.task(tid).get("brief") or {})
            b["details"] = f"use ttnn.bringup.{op} (op-gen, request {req_rel})"
            led.update_task_def(tid, brief=b)
            out["reset"] += [t for t in rerun_from(led, tid) if t not in out["reset"]]
            led.update(tid, deferred=None)
        out["ops"].append(op)
        out["paths"] += [str(p.relative_to(spec.repo)) for p in (dst, bops / "__init__.py", idx, d) if p.exists()]
    return out

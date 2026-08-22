# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Recover a traced test's ``device_params`` from the repo, without re-tracing.

``tracer_pytest_plugin`` writes a ``_device_params.json`` sidecar for every NEW trace, but the
existing corpus was captured before that existed and re-tracing every model is not feasible. The
information is not lost, though: every traced execution records ``source`` (e.g.
``models/demos/llama3_70b_galaxy/demo/text_demo.py::test_demo_text``), and that test's
``device_params`` parametrization is still in the repo. This module reads it back out.

Static AST parsing, deliberately -- resolving via ``pytest --collect-only`` would import the model
module (heavy deps, sometimes weights) inside the sweep worker for a handful of integers.

Two properties keep this safe, because a WRONG device parameter is worse than an absent one (absent
leaves the replayer on its documented default; wrong silently changes the workload):

- ``source`` is ``file::test_name`` with no parametrization id, so a test with several
  ``device_params`` variants cannot be pinned to the one that was traced. Only keys on which every
  variant AGREES are returned; a key whose value differs between variants is dropped.
- Any parse/lookup failure returns ``{}`` rather than a guess.

Values are returned in the same shape the sidecar uses: JSON scalars where possible, otherwise the
source text of the expression (``"ttnn.DispatchCoreAxis.COL"``), which the consumer maps back to a
real enum through an explicit allowlist.
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache
from pathlib import Path

from loguru import logger

# Same key set the tracer sidecar records, so both paths produce identical dicts.
DEVICE_PARAM_KEYS = (
    "fabric_config",
    "fabric_tensix_config",
    "fabric_manager",
    "fabric_router_config",
    "reliability_mode",
    "worker_l1_size",
    "l1_small_size",
    "trace_region_size",
    "num_command_queues",
    "dispatch_core_axis",
    "dispatch_core_type",
)

_SENTINEL = object()

# Depth cap for the literal reader below. A device parameter is a byte size, a queue count, a bool
# or an enum -- depth 1 in practice -- so anything deeper is not one, and refusing it costs nothing.
_MAX_LITERAL_DEPTH = 4


def _literal(node, depth=0):
    """The Python value of a literal AST node, or _SENTINEL if it is not one.

    Deliberately NOT ast.literal_eval: that accepts arbitrarily large and deeply nested structures
    (CWE-400) and is flagged wherever its input is not provably constant. This reads only the node
    types a device parameter can be, recursing at most _MAX_LITERAL_DEPTH, so the work is bounded by
    the shape of what we accept rather than by the size of what we are handed.
    """
    if depth > _MAX_LITERAL_DEPTH:
        return _SENTINEL
    if isinstance(node, ast.Constant):  # str, int, float, bool, None
        return node.value
    # -1 / +1 parse as a UnaryOp around the constant, not as a negative constant.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _literal(node.operand, depth + 1)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
        return _SENTINEL
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_literal(e, depth + 1) for e in node.elts]
        if any(i is _SENTINEL for i in items):
            return _SENTINEL
        return tuple(items) if isinstance(node, ast.Tuple) else (set(items) if isinstance(node, ast.Set) else items)
    if isinstance(node, ast.Dict):
        out = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:  # {**spread} -- not a literal we can read
                return _SENTINEL
            key = _literal(key_node, depth + 1)
            value = _literal(value_node, depth + 1)
            if key is _SENTINEL or value is _SENTINEL:
                return _SENTINEL
            out[key] = value
        return out
    return _SENTINEL


def _render(node):
    """A literal as its Python value, anything else as its source text."""
    value = _literal(node)
    if value is not _SENTINEL:
        return value
    try:
        return ast.unparse(node)  # enums and other names keep their source form
    except Exception:
        return _SENTINEL


def _is_parametrize(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr == "parametrize"
    return isinstance(func, ast.Name) and func.id == "parametrize"


def _device_params_entries(call: ast.Call) -> list[dict]:
    """The dicts from a ``parametrize("device_params", [...])`` call, or [] if it is not one.

    Also accepts the combined form ``parametrize("device_params, other", ...)`` by refusing it:
    the argvalues are then tuples whose positional meaning we would have to infer, and inferring is
    exactly what this module does not do.
    """
    if not _is_parametrize(call) or len(call.args) < 2:
        return []
    argnames = _render(call.args[0])
    if argnames != "device_params":
        return []
    argvalues = call.args[1]
    if not isinstance(argvalues, (ast.List, ast.Tuple)):
        return []

    entries = []
    for entry in argvalues.elts:
        # pytest.param({...}, id=...) wraps the real dict.
        if isinstance(entry, ast.Call) and not isinstance(entry.func, ast.Dict):
            inner = [a for a in entry.args if isinstance(a, ast.Dict)]
            entry = inner[0] if len(inner) == 1 else entry
        if isinstance(entry, ast.Dict):
            entries.append(entry)
    return entries


def _decorator_calls(node) -> list[ast.Call]:
    return [d for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Call)]


def _collect_for_function(tree: ast.Module, test_name: str | None) -> list[ast.Dict]:
    """device_params dicts visible to ``test_name``: its own decorators, its class's, and pytestmark.

    ``test_name`` of None means the source named a file with no ``::test`` (which the tracer stores
    verbatim, and which most documented tracing commands use): collect from every test function in
    the file instead. The agreement rule in _merge then does the disambiguation -- a file whose tests
    declare different device_params yields nothing, exactly as a multi-variant test does.
    """
    dicts: list[ast.Dict] = []

    def module_pytestmark():
        found = []
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
                continue
            marks = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            for mark in marks:
                if isinstance(mark, ast.Call):
                    found.extend(_device_params_entries(mark))
        return found

    def _wanted(name):
        return name == test_name if test_name is not None else name.startswith("test_")

    def walk(body, in_scope_decorators):
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, in_scope_decorators + _decorator_calls(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _wanted(node.name):
                for call in in_scope_decorators + _decorator_calls(node):
                    dicts.extend(_device_params_entries(call))

    walk(tree.body, [])
    if not dicts:
        dicts = module_pytestmark()
    return dicts


def _merge(entries: list[dict]) -> dict:
    """Keep only keys every parametrization agrees on -- see the module docstring."""
    if not entries:
        return {}
    merged = {}
    for key in DEVICE_PARAM_KEYS:
        values = [e[key] for e in entries if key in e]
        if len(values) != len(entries):
            continue  # not declared by every variant -> cannot attribute it to the traced run
        first = values[0]
        if all(v == first for v in values[1:]):
            merged[key] = first
    return merged


@lru_cache(maxsize=512)
def resolve_device_params(source: str, repo_root: str | None = None) -> tuple:
    """``device_params`` for a traced ``source`` (``path/to/test.py::test_name``).

    Returned as a sorted tuple of pairs so the result stays hashable/cacheable; use
    ``dict(resolve_device_params(...))``. Empty when the source is unparseable, the test is not
    found, nothing is declared, or the declarations disagree.
    """
    if not source or not isinstance(source, str):
        return ()
    path_part, separator, test_part = source.partition("::")
    if separator:
        test_name = test_part.split("[", 1)[0].split("::")[-1].strip()
        if not test_name:
            return ()
    else:
        # File-only source (``models/.../demo.py``): generic_ops_tracer stores test_path verbatim and
        # most documented tracing commands omit ``::test``. Resolve across the file's tests.
        test_name = None
    if not path_part.endswith(".py"):
        return ()

    root = Path(repo_root or os.environ.get("TT_METAL_HOME") or ".")
    path = Path(path_part)
    candidate = path if path.is_absolute() else root / path
    if not candidate.is_file():
        return ()

    try:
        tree = ast.parse(candidate.read_text())
    except (OSError, SyntaxError, ValueError) as exc:
        logger.debug("Could not parse {} for device_params: {}", candidate, exc)
        return ()

    entries = []
    for node in _collect_for_function(tree, test_name):
        rendered = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = _render(key_node) if key_node is not None else None
            if key not in DEVICE_PARAM_KEYS:
                continue  # includes TRACE_MODEL_KEY_PARAM and other non-device knobs
            value = _render(value_node)
            if value is not _SENTINEL:
                rendered[key] = value
        entries.append(rendered)

    return tuple(sorted(_merge(entries).items()))

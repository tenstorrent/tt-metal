# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every name a function reads is bound in a scope it can see (no device, no import of the module under test).

Two served holds of 2026-09-26 came from names that pyflakes' undefined-name check would have flagged at test time: a
hook read ``fused_kernels``, an import local to ``__init__``, and a lifted helper read ``full_hidden``, a local of the
method it was lifted from.  This static walks every Python file of the model directory with the standard library's
``ast`` and resolves each read name against the scopes Python would search -- the function and its enclosing
functions (comprehensions and lambdas included), the module, the builtins -- and fails on any name bound nowhere on
that path.  Class bodies are not a scope methods can read from, as in Python.  Names bound anywhere in a scope count
for the whole scope (Python's compile-time binding rule), ``global`` / ``nonlocal`` declarations, star imports and
``__class__`` are honoured; a file that uses a star import is checked only for names the module binds nowhere at all."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".git"}
_MATCH_BINDERS = tuple(getattr(ast, n) for n in ("MatchAs", "MatchStar", "MatchMapping") if hasattr(ast, n))
BUILTIN_NAMES = set(dir(builtins)) | {
    "__file__",
    "__name__",
    "__doc__",
    "__spec__",
    "__loader__",
    "__package__",
    "__builtins__",
    "__class__",
    "__debug__",
    "__path__",
}


def _bindings(node: ast.AST, into: set[str]) -> None:
    """Names bound by a statement or expression node, not descending into nested function / class / lambda scopes
    (their own bindings are theirs), but descending into comprehensions (their targets are readable inside them; we
    let them leak into the enclosing scope, a superset that never hides an undefined name outside a comprehension)."""

    for child in ast.iter_child_nodes(node):
        _collect(child, into)


def _collect(node: ast.AST, into: set[str]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        into.add(node.name)
        for dec in node.decorator_list:
            _collect(dec, into)
        return  # the body is another scope
    if isinstance(node, ast.Lambda):
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        into.add(node.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            into.add((alias.asname or alias.name).split(".")[0])
    elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
        into.update(node.names)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        into.add(node.name)
    elif isinstance(node, ast.arg):
        into.add(node.arg)
    elif isinstance(node, _MATCH_BINDERS):  # 3.10+ match statements; no-ops before
        for attr in ("name", "rest"):
            bound = getattr(node, attr, None)
            if isinstance(bound, str):
                into.add(bound)
    _bindings(node, into)


def _arguments(fn) -> set[str]:
    args = fn.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _star_import(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))


def _check_scope(node, visible: set[str], out: list[str], path: Path, module_names: set[str], star: bool) -> None:
    """``node`` is a function or lambda whose body is a scope; ``visible`` the names its enclosing scopes bind."""

    local: set[str] = set(_arguments(node))
    body = node.body if isinstance(node.body, list) else [node.body]
    for statement in body:
        _collect(statement, local)
    scope = visible | local
    # reads in this scope (not inside nested functions / classes, which are checked with their own visible set)
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in n.decorator_list:
                stack.append(dec)
            for default in (*n.args.defaults, *n.args.kw_defaults):
                if default is not None:
                    stack.append(default)
            _check_scope(n, scope, out, path, module_names, star)
            continue
        if isinstance(n, ast.Lambda):
            _check_scope(n, scope, out, path, module_names, star)
            continue
        if isinstance(n, ast.ClassDef):
            _check_class(n, scope, out, path, module_names, star)
            for base in (*n.bases, *n.decorator_list):
                stack.append(base)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in scope and n.id not in BUILTIN_NAMES and not (star and n.id in module_names):
                if not star or n.id not in module_names:
                    out.append(
                        f"{path.relative_to(MODEL_DIR)}:{n.lineno}: undefined name '{n.id}' in {getattr(node, 'name', '<lambda>')}"
                    )
            continue
        stack.extend(ast.iter_child_nodes(n))


def _check_class(
    node: ast.ClassDef, visible: set[str], out: list[str], path: Path, module_names: set[str], star: bool
) -> None:
    """A class body reads its own bindings and the enclosing scope's; its methods read the enclosing scope only."""

    class_names: set[str] = set()
    for statement in node.body:
        _collect(statement, class_names)
    body_scope = visible | class_names
    stack: list[ast.AST] = []
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in statement.decorator_list:
                stack.append(dec)
            for default in (*statement.args.defaults, *statement.args.kw_defaults):
                if default is not None:
                    stack.append(default)
            _check_scope(statement, visible, out, path, module_names, star)
        else:
            stack.append(statement)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            _check_scope(n, body_scope, out, path, module_names, star)
            continue
        if isinstance(n, ast.ClassDef):
            _check_class(n, body_scope, out, path, module_names, star)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in body_scope and n.id not in BUILTIN_NAMES and not (star and n.id in module_names):
                out.append(f"{path.relative_to(MODEL_DIR)}:{n.lineno}: undefined name '{n.id}' in class {node.name}")
            continue
        stack.extend(ast.iter_child_nodes(n))


def undefined_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_names: set[str] = set()
    for statement in tree.body:
        _collect(statement, module_names)
    star = _star_import(tree)
    out: list[str] = []
    visible = module_names
    stack = list(tree.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            _check_scope(n, visible, out, path, module_names, star)
            continue
        if isinstance(n, ast.ClassDef):
            _check_class(n, visible, out, path, module_names, star)
            for base in (*n.bases, *n.decorator_list):
                stack.append(base)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in module_names and n.id not in BUILTIN_NAMES and not star:
                out.append(f"{path.relative_to(MODEL_DIR)}:{n.lineno}: undefined name '{n.id}' at module level")
            continue
        stack.extend(ast.iter_child_nodes(n))
    return out


def model_python_files() -> list[Path]:
    return sorted(p for p in MODEL_DIR.rglob("*.py") if not (SKIP_DIRS & set(p.relative_to(MODEL_DIR).parts)))


# Files with undefined names when this check landed (2026-09-26), by file name: two layer-0 diagnostic tools outside
# the public tree whose contract tests the full set already fails.  Their counts may only shrink; every other file
# must be clean.
KNOWN_LEGACY: dict[str, int] = {
    "layer0_attention_gr_down_contract.py": 183,
    "layer0_collective_diagnostic_contract.py": 128,
}


def test_every_python_file_of_the_model_reads_only_names_bound_in_a_scope_it_can_see() -> None:
    files = model_python_files()
    assert len(files) > 100, "the model directory's Python files"
    assert len([path for path in files if path.name in KNOWN_LEGACY]) <= len(KNOWN_LEGACY), "legacy names are unique"
    findings: dict[Path, list[str]] = {}
    for path in files:
        lines = undefined_names(path)
        if lines:
            findings[path] = lines
    new = {path: lines for path, lines in findings.items() if path.name not in KNOWN_LEGACY}
    assert not new, "undefined names:\n" + "\n".join(line for lines in new.values() for line in lines)
    for path, lines in findings.items():
        cap = KNOWN_LEGACY.get(path.name)
        if cap is not None:
            assert len(lines) <= cap, f"{path.name}: {len(lines)} undefined names, more than the {cap} recorded"


def test_the_check_catches_the_two_served_holds_forms() -> None:
    """A local import read from another method, and a lifted helper reading its origin's local."""

    source = """
class M:
    def __init__(self):
        from os import path as fused_kernels
        self.h = fused_kernels.join
    def hook(self, x):
        return fused_kernels.qsa_rows.admits(x)
    def step(self, full_hidden):
        return self.helper(full_hidden)
    def helper(self, qg):
        return retag(qg, reference=full_hidden)
def retag(t, reference):
    return t
"""
    path = MODEL_DIR / "tests" / "_names_probe.py"
    path.write_text(source, encoding="utf-8")
    try:
        found = undefined_names(path)
    finally:
        path.unlink()
    assert any("'fused_kernels' in hook" in f for f in found) and any(
        "'full_hidden' in helper" in f for f in found
    ), found
    assert len(found) == 2, found

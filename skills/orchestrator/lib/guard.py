# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""No-shortcuts guard: lint + traced-op assertion + reference cross-check.

This module is the static safety net for the Bringup Orchestrator. It is
pure-Python: it reads source files and walks their AST, never importing
torch or ttnn. Three responsibilities:

1. ``lint_block`` flags host-resident shortcuts inside ``forward`` methods
   of a TTNN block file (``.cpu()``, ``.numpy()``, ``torch.nn.functional``,
   ``torch.matmul``, plus the ``# TODO: move to ttnn`` comment marker).
2. ``assert_traced_ops`` checks that a traced op list for a block of a
   given kind (norm, linear, attention, mlp, decoder_layer, embedding)
   contains at least one kernel from each required option set.
3. ``cross_check_reference`` is a best-effort grep diff: a host-resident
   sub-op that exists in the reference implementation is permitted in the
   new block. Only sub-ops introduced by the new block are flagged.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintViolation:
    """A single forbidden pattern occurrence in a block file."""

    file: str
    line: int
    column: int
    pattern: str
    snippet: str


@dataclass(frozen=True)
class HostResidentSubOp:
    """A host-resident sub-op present in a block but absent from the reference."""

    file: str
    line: int
    op: str
    snippet: str


# ---------------------------------------------------------------------------
# Required-kernel table
# ---------------------------------------------------------------------------


# Each kind maps to a list of OPTION sets. For each option set, at least
# one kernel name must appear in the traced op list. Missing option sets
# become entries in the assert_traced_ops return value.
KIND_REQUIRED_KERNELS: dict[str, list[set[str]]] = {
    "norm": [
        {"ttnn.rms_norm", "ttnn.layer_norm"},
    ],
    "linear": [
        {"ttnn.linear", "ttnn.matmul"},
    ],
    "attention": [
        {"ttnn.linear", "ttnn.matmul"},
        {"ttnn.softmax", "ttnn.transformer.scaled_dot_product_attention"},
    ],
    "mlp": [
        {"ttnn.linear", "ttnn.matmul"},
        {"ttnn.silu", "ttnn.gelu", "ttnn.geglu", "ttnn.relu"},
    ],
    "decoder_layer": [
        {"ttnn.linear", "ttnn.matmul"},
    ],
    "embedding": [
        {"ttnn.embedding"},
    ],
}


# ---------------------------------------------------------------------------
# Forbidden patterns inside forward(_*)
# ---------------------------------------------------------------------------


# Patterns we surface to the user. Method patterns are matched via AST
# attribute-call inspection; attribute patterns are matched via the dotted
# name reconstructed from nested ast.Attribute nodes.
_FORBIDDEN_METHOD_CALLS = {"cpu", "numpy"}  # x.cpu() / x.numpy()
_FORBIDDEN_DOTTED = {
    "torch.nn.functional": "torch.nn.functional",
    "torch.matmul": "torch.matmul",
}
_TODO_RE = re.compile(r"#\s*TODO[: ]\s*move to ttnn", re.IGNORECASE)


def _is_test_path(path: Path) -> bool:
    """Return True if ``path`` should be skipped entirely (test file).

    Exempt if:
        - filename starts with ``test_`` or equals ``test.py``
        - any ancestor directory is named exactly ``tests``

    We deliberately do NOT match parent names that merely *start with*
    ``test_`` — pytest's own ``tmp_path`` puts files under
    ``/tmp/pytest-of-USER/pytest-N/test_<funcname>0/...`` and we don't
    want to silently exempt those during self-tests.
    """
    name = path.name
    if name.startswith("test_") or name == "test.py":
        return True
    for ancestor in path.parents:
        if ancestor.name == "tests":
            return True
    return False


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct a dotted name like ``torch.nn.functional`` from an Attribute chain.

    Returns None if the chain doesn't terminate in a simple Name.
    """
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


class _ForwardVisitor(ast.NodeVisitor):
    """Walks every ``forward`` / ``forward_*`` function and collects violations.

    Does NOT recurse into ``test_*`` functions.
    """

    def __init__(self, file_path: Path, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = source_lines
        self.violations: list[LintViolation] = []
        # Stack of booleans: are we currently inside a forward(_*) body?
        self._in_forward = [False]

    # FunctionDef / AsyncFunctionDef ----------------------------------------

    def _enter_func(self, node: ast.AST) -> None:
        name = getattr(node, "name", "")
        if name.startswith("test_"):
            # Skip the entire test function — do not descend.
            return
        if name == "forward" or name.startswith("forward_"):
            self._in_forward.append(True)
        else:
            self._in_forward.append(self._in_forward[-1])
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self._in_forward.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._enter_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._enter_func(node)

    # Forbidden constructs --------------------------------------------------

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _record(self, node: ast.AST, pattern: str) -> None:
        lineno = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        self.violations.append(
            LintViolation(
                file=str(self.file_path),
                line=lineno,
                column=col + 1 if col is not None else 0,
                pattern=pattern,
                snippet=self._snippet(lineno),
            )
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self._in_forward[-1]:
            func = node.func
            # x.cpu() / x.numpy()
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _FORBIDDEN_METHOD_CALLS
                and not node.args
                and not node.keywords
            ):
                self._record(node, f".{func.attr}()")
            # torch.matmul(...) (call form) — handled in visit_Attribute too,
            # but a Call whose func is the Attribute fires here first. We
            # let visit_Attribute pick it up via generic recursion.
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if self._in_forward[-1]:
            dotted = _dotted_name(node)
            if dotted is not None:
                for forbidden in _FORBIDDEN_DOTTED:
                    if dotted == forbidden or dotted.startswith(forbidden + "."):
                        self._record(node, _FORBIDDEN_DOTTED[forbidden])
                        # Skip recursion: the inner Attribute chain
                        # (e.g. ``torch.nn.functional`` inside
                        # ``torch.nn.functional.relu``) would otherwise
                        # produce a second, duplicate violation.
                        return
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_block(file_path: PathLike) -> list[LintViolation]:
    """Scan a TTNN block file for forbidden host-resident patterns.

    Walks the file's AST. Inside every function whose name is ``forward``
    or starts with ``forward_``, looks for ``.cpu()``, ``.numpy()``,
    ``torch.nn.functional``, ``torch.matmul``. Also regex-scans the whole
    file for ``# TODO: move to ttnn`` (case-insensitive).

    Test files (path contains ``/tests/`` or filename starts with
    ``test_``) and ``test_*`` functions are exempt.

    Returns ``[]`` for clean files. Missing or unparseable files return
    ``[]`` with a UserWarning.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        warnings.warn(f"lint_block: file not found: {path}", UserWarning, stacklevel=2)
        return []
    if _is_test_path(path):
        return []
    try:
        source = path.read_text()
    except OSError as exc:
        warnings.warn(f"lint_block: cannot read {path}: {exc}", UserWarning, stacklevel=2)
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        warnings.warn(f"lint_block: cannot parse {path}: {exc}", UserWarning, stacklevel=2)
        return []

    lines = source.splitlines()
    visitor = _ForwardVisitor(path, lines)
    visitor.visit(tree)
    violations = list(visitor.violations)

    # Whole-file TODO scan (independent of AST).
    for idx, line in enumerate(lines, start=1):
        if _TODO_RE.search(line):
            violations.append(
                LintViolation(
                    file=str(path),
                    line=idx,
                    column=(line.find("#") + 1) if "#" in line else 0,
                    pattern="# TODO: move to ttnn",
                    snippet=line.strip(),
                )
            )

    # Sort for deterministic output.
    violations.sort(key=lambda v: (v.line, v.column, v.pattern))
    return violations


def assert_traced_ops(traced_op_list: list[str], kind: str) -> list[str]:
    """Report which required-kernel option sets are not satisfied by ``traced_op_list``.

    For each option set in ``KIND_REQUIRED_KERNELS[kind]``, at least one
    kernel name must appear in ``traced_op_list``. Each unsatisfied option
    set yields a string like ``"any of {ttnn.linear, ttnn.matmul}"`` —
    kernels sorted alphabetically, comma-space separated.

    Raises ``ValueError`` if ``kind`` is unknown.
    """
    if kind not in KIND_REQUIRED_KERNELS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(KIND_REQUIRED_KERNELS)}")
    traced = set(traced_op_list)
    missing: list[str] = []
    for option_set in KIND_REQUIRED_KERNELS[kind]:
        if not (option_set & traced):
            rendered = "{" + ", ".join(sorted(option_set)) + "}"
            missing.append(f"any of {rendered}")
    return missing


# ---------------------------------------------------------------------------
# Reference cross-check
# ---------------------------------------------------------------------------


# Regex tags for the four host-resident sub-op categories. Order matters
# only for diagnostics; matches are stored under their canonical ``op`` key.
_HOST_OP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cpu", re.compile(r"\.cpu\s*\(\s*\)")),
    ("numpy", re.compile(r"\.numpy\s*\(\s*\)")),
    ("torch.nn.functional", re.compile(r"\btorch\.nn\.functional\b")),
    ("torch.matmul", re.compile(r"\btorch\.matmul\b")),
]


def _scan_host_ops(path: Path) -> list[HostResidentSubOp]:
    """Return one HostResidentSubOp per matching line in ``path``."""
    try:
        text = path.read_text()
    except OSError:
        return []
    out: list[HostResidentSubOp] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for op, pat in _HOST_OP_PATTERNS:
            if pat.search(line):
                out.append(
                    HostResidentSubOp(
                        file=str(path),
                        line=lineno,
                        op=op,
                        snippet=line.strip(),
                    )
                )
    return out


def _iter_py_files(path: Path):
    if path.is_dir():
        yield from sorted(path.rglob("*.py"))
    elif path.is_file():
        yield path


def cross_check_reference(
    block_path: PathLike,
    reference_impl_path: PathLike,
) -> list[HostResidentSubOp]:
    """Flag host-resident sub-ops in ``block_path`` that the reference doesn't also use.

    Best-effort grep diff:
        - Scan ``block_path`` for ``.cpu()``, ``.numpy()``,
          ``torch.nn.functional``, ``torch.matmul``.
        - Scan the reference (file or directory) for the same four ops.
        - Match by ``op`` token only (not by line content). If the
          reference uses ``.cpu()`` anywhere, all ``.cpu()`` lines in the
          block are allowed.

    Returns the list of HostResidentSubOp entries from the block whose
    ``op`` is NOT seen in the reference.
    """
    block = Path(block_path)
    ref = Path(reference_impl_path)
    if not block.exists() or not block.is_file():
        warnings.warn(f"cross_check_reference: block not found: {block}", UserWarning, stacklevel=2)
        return []

    block_ops = _scan_host_ops(block)
    if not block_ops:
        return []

    ref_op_kinds: set[str] = set()
    if ref.exists():
        for ref_file in _iter_py_files(ref):
            for entry in _scan_host_ops(ref_file):
                ref_op_kinds.add(entry.op)
    else:
        warnings.warn(
            f"cross_check_reference: reference not found: {ref}",
            UserWarning,
            stacklevel=2,
        )

    return [entry for entry in block_ops if entry.op not in ref_op_kinds]

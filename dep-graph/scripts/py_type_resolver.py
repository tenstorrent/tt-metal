"""Jedi-backed type-aware resolution for Python refs.

(B4 in the Phase 3 plan; pivoted from Pyright because Pyright's CLI doesn't
expose per-expression types — only its LSP server does. Jedi provides direct
Python APIs for the queries we need.)

For each Python file we want to resolve, we open a `jedi.Script` once and
run `goto(line, col)` at the position of each unresolved ref's symbol. The
returned `Name.full_name` is matched against our node table's
`qualified_name` to produce an intra-Python `calls` edge.

Pure refs that resolve to torch / numpy / stdlib are silently dropped — we
don't have nodes for those.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

try:
    import jedi
except ImportError:  # graceful: callers can detect that B4 isn't available
    jedi = None  # type: ignore[assignment]


def jedi_available() -> bool:
    return jedi is not None


def resolve_refs(
    refs_by_file: dict[str, list[tuple[int, int, int]]],
    *,
    follow_imports: bool = True,
    project_root: str | None = None,
) -> dict[tuple[str, int, int], str]:
    """For each (file → list of (ref_index, line, col)), return a map of
    (file, line, col) → resolved Jedi `full_name`.

    `project_root` controls how Jedi computes module dotted names: a file at
    `<root>/foo/bar.py` resolves to `foo.bar`. For tt-metal we want
    `/workspace/ttnn` so that `/workspace/ttnn/ttnn/operations/binary.py`
    resolves to `ttnn.operations.binary` (matching `py_index`'s
    `module-root`).

    Skips entries whose Jedi resolution returns no Name objects.
    """
    out: dict[tuple[str, int, int], str] = {}
    if not jedi_available():
        return out

    project = None
    if project_root:
        try:
            project = jedi.Project(path=project_root)
        except Exception:
            project = None

    for file_path, positions in refs_by_file.items():
        try:
            source = open(file_path).read()
        except OSError:
            continue
        try:
            script = jedi.Script(source, path=file_path, project=project)
        except Exception:
            continue
        for _ref_idx, line, col in positions:
            try:
                defs = script.goto(line, col, follow_imports=follow_imports)
            except Exception:
                continue
            for d in defs:
                fn = d.full_name
                if not fn:
                    continue
                out[(file_path, line, col)] = fn
                break  # first hit wins
    return out

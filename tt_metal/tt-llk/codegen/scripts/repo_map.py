# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic repository map for the LLK issue solver.

Agents spend most of their actions rediscovering this tree. Measured on trial04:
of 343 inspection commands, 150 went to test sources, 86 to LLK kernel source and
61 to the pipeline's own files, and almost every probe was unique -- the same
understanding rebuilt from scratch, one narrow shell read at a time, on every run.

This builds that understanding once, without a model, keyed by base commit, and
caches it so the second and later runs at a base pay nothing. It is an *index*
with pointers, never an inlined dump: the point is to answer "which module, which
marker, which symbol, which selector" in one read so the agent can go straight to
the few files that matter.

It is deliberately static. Parsing the AST needs no provisioned test environment
and cannot execute repository code, so it runs at admission before anything
expensive. That costs exact parametrize cardinality, which is computed at
collection time; axis names, markers and node names are all statically known and
answer the questions the probes were actually asking.

Usage:
    python codegen/scripts/repo_map.py --worktree /path/to/tt-metal
    python codegen/scripts/repo_map.py --worktree WT --cache-dir /shared/maps
    python codegen/scripts/repo_map.py --worktree WT --summary-out map.md
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "repo-map-v1"

# Markers the pipeline routes on. llk_host in particular decides whether a leaf
# seals to the host wrapper or to silicon; trial04 lost 19m14s to that question.
ROUTING_MARKERS = ("llk_host", "perf", "nightly", "long", "quasar", "accuracy")

_SYMBOL = re.compile(
    r"^\s*(?:inline\s+|static\s+|constexpr\s+|template\s*<[^>]*>\s*)*"
    r"(?:void|bool|int|uint\d+_t|float|auto|[A-Za-z_][\w:]*)\s+"
    r"(llk_[A-Za-z0-9_]+|_llk_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)


def _git(worktree: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(worktree), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _marker_names(decorator: ast.expr) -> list[str]:
    """Collect pytest.mark.<name> from a decorator, including mark.parametrize."""
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    if "mark" in parts:
        return parts[parts.index("mark") + 1 :]
    # the repository's own parametrize helper, used alongside pytest's
    return ["parametrize"] if parts and parts[-1] == "parametrize" else []


def _parametrize_info(decorator: ast.expr) -> tuple[bool, list[str], list[str]]:
    """(is_parametrize, axis names it states, sweep constants it unpacks).

    Three forms appear in this tree and only two of them name their axes:
    ``parametrize("a,b", [...])``, ``parametrize(axis=values)``, and
    ``parametrize(**DATACOPY_SWEEP)``. The third form carries the big matrices,
    so reading decorators alone reports the most heavily parametrized tests in
    the suite as having no axes at all. The caller recovers those from the
    function signature; the unpacked constant is reported so a reader can jump
    straight to the sweep definition instead of searching for it.
    """
    if not isinstance(decorator, ast.Call):
        return False, [], []
    names = _marker_names(decorator)
    if not names or names[-1] != "parametrize":
        return False, [], []
    axes: list[str] = []
    sweeps: list[str] = []
    if decorator.args and isinstance(decorator.args[0], ast.Constant):
        raw = decorator.args[0].value
        if isinstance(raw, str):
            axes.extend(a.strip() for a in raw.split(",") if a.strip())
    for kw in decorator.keywords:
        if kw.arg:
            axes.append(kw.arg)
        elif isinstance(kw.value, ast.Name):  # **SWEEP_CONSTANT
            sweeps.append(kw.value.id)
        elif isinstance(kw.value, ast.Attribute):  # **module.SWEEP
            sweeps.append(kw.value.attr)
    return True, axes, sweeps


def _signature_axes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """A parametrized test's axes are its parameters, minus pytest's own fixtures."""
    fixtures = {"request", "tmp_path", "capsys", "monkeypatch", "caplog"}
    args = [
        a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    ]
    return [a for a in args if a not in fixtures]


def _module_level_marks(tree: ast.Module) -> tuple[set[str], list[str]]:
    """Markers applied to a whole module via ``pytestmark``.

    This is not a rare form: ``test_perf_header_gate.py`` marks itself
    ``llk_host`` this way, and that single marker decides host-versus-silicon
    routing. A decorator-only scan reports it as unmarked, which is worse than
    reporting nothing. Names that cannot be resolved statically -- module-level
    aliases such as ``skip_for_wormhole`` -- are returned separately rather than
    guessed at.
    """
    marks: set[str] = set()
    unresolved: list[str] = []

    def absorb(value: ast.expr) -> None:
        if isinstance(value, (ast.List, ast.Tuple)):
            for item in value.elts:
                absorb(item)
            return
        names = _marker_names(value)
        if names:
            marks.update(n for n in names if n != "parametrize")
        elif isinstance(value, ast.Name):
            unresolved.append(value.id)

    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            if node.value is not None:
                absorb(node.value)
    return marks, sorted(set(unresolved))


def _scan_test_module(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    tests = []
    inherited, unresolved = _module_level_marks(tree)
    module_markers: set[str] = set(inherited)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        markers: set[str] = set(inherited)
        axes: list[str] = []
        sweeps: list[str] = []
        parametrized = False
        for dec in node.decorator_list:
            for name in _marker_names(dec):
                if name != "parametrize":
                    markers.add(name)
            is_param, dec_axes, dec_sweeps = _parametrize_info(dec)
            parametrized = parametrized or is_param
            axes.extend(dec_axes)
            sweeps.extend(dec_sweeps)
        if parametrized and not axes:
            axes = _signature_axes(node)
        module_markers |= markers
        entry: dict[str, Any] = {
            "name": node.name,
            "line": node.lineno,
            "markers": sorted(markers),
            "param_axes": sorted(set(axes)),
        }
        if sweeps:
            entry["param_sweeps"] = sorted(set(sweeps))
        tests.append(entry)
    if not tests:
        return None
    result = {
        "module": path.name,
        "path": path.relative_to(root).as_posix(),
        "markers": sorted(module_markers),
        "module_level_markers": sorted(inherited),
        "test_count": len(tests),
        "tests": tests,
    }
    if unresolved:
        # Aliases defined in the module; a reader must open the file to resolve
        # them. Naming them is the honest signal that the marker set is partial.
        result["unresolved_module_marks"] = unresolved
    return result


def _scan_headers(llk: Path) -> dict[str, Any]:
    """Symbol index for LLK headers, grouped by the arch directory that owns them."""
    by_arch: dict[str, dict[str, list[str]]] = {}
    for header in sorted(llk.rglob("*.h")):
        rel = header.relative_to(llk).as_posix()
        if rel.startswith("tests/"):
            continue
        try:
            text = header.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        symbols = sorted({m.group(1) for m in _SYMBOL.finditer(text)})
        if not symbols:
            continue
        # tt_llk_<arch>/... or common/...
        arch = rel.split("/", 1)[0]
        by_arch.setdefault(arch, {})[rel] = symbols
    return by_arch


def _scan_harness(llk: Path) -> dict[str, Any]:
    """The run_test.sh contract, so a role need not read the wrapper to call it."""
    script = llk / ".claude/scripts/run_test.sh"
    if not script.is_file():
        return {}
    text = script.read_text(encoding="utf-8", errors="replace")
    subcommands = sorted(
        set(re.findall(r"^\s*(compile|run|simulate|host)\)", text, re.M))
    )
    flags = sorted(set(re.findall(r"(--[a-z][a-z0-9-]+)\)", text)))
    return {
        "path": script.relative_to(llk).as_posix(),
        "subcommands": subcommands,
        "flags": flags,
    }


def build_map(worktree: Path) -> dict[str, Any]:
    worktree = worktree.resolve(strict=True)
    llk = worktree / "tt_metal/tt-llk"
    if not llk.is_dir():
        raise ValueError("worktree does not contain tt_metal/tt-llk")
    dirty = bool(_git(worktree, "status", "--porcelain"))
    tests_root = llk / "tests/python_tests"

    modules = []
    for path in sorted(tests_root.rglob("*.py")):
        rel = path.relative_to(llk).as_posix()
        if "/helpers/" in rel or path.name == "conftest.py":
            continue
        scanned = _scan_test_module(path, llk)
        if scanned:
            modules.append(scanned)

    host_nodes = [
        f"{m['path']}::{t['name']}"
        for m in modules
        for t in m["tests"]
        if "llk_host" in t["markers"]
    ]
    headers = _scan_headers(llk)
    kernels = (
        sorted(p.name for p in (llk / "tests/sources").glob("*.cpp"))
        if (llk / "tests/sources").is_dir()
        else []
    )

    document: dict[str, Any] = {
        "schema": "tt.issue-solver.repo-map",
        "version": 1,
        "generator": GENERATOR_VERSION,
        "base_commit": _git(worktree, "rev-parse", "HEAD") or "unknown",
        "worktree_dirty": dirty,
        # A reader has to know whether this reflects the pristine base or the
        # base plus the candidate's edits. The writer is required to add tests,
        # including llk_host markers, so a map built at setup goes stale the
        # moment it does -- and a stale host-marked list is exactly the kind of
        # wrong answer that routes a leaf to the wrong executor.
        "describes": "base+candidate" if dirty else "base",
        "counts": {
            "test_modules": len(modules),
            "test_functions": sum(m["test_count"] for m in modules),
            "host_marked_nodes": len(host_nodes),
            "kernel_sources": len(kernels),
            "indexed_headers": sum(len(v) for v in headers.values()),
            "indexed_symbols": sum(
                len(s) for v in headers.values() for s in v.values()
            ),
        },
        "routing_markers": list(ROUTING_MARKERS),
        "host_marked_nodes": host_nodes,
        "test_modules": modules,
        "kernel_sources": kernels,
        "llk_symbols_by_arch": headers,
        "harness": _scan_harness(llk),
    }
    document["map_id"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in document.items() if k != "map_id"}, sort_keys=True
        ).encode()
    ).hexdigest()
    return document


def render_summary(document: dict[str, Any]) -> str:
    """A compact index an agent reads once, with pointers instead of contents."""
    c = document["counts"]
    lines = [
        f"# Repository map ({document['generator']}, base {document['base_commit'][:12]})",
        "",
        f"Describes: **{document.get('describes', 'base')}**."
        + (
            "  This includes the candidate's current edits."
            if document.get("describes") == "base+candidate"
            else "  The candidate's own edits are not in it yet."
        ),
        "",
        f"{c['test_modules']} test modules / {c['test_functions']} test functions; "
        f"{c['kernel_sources']} kernel sources; "
        f"{c['indexed_symbols']} LLK symbols across {c['indexed_headers']} headers.",
        "",
        "Read the full index at the JSON path this summary came with when you need "
        "a module's axes or a symbol's file. Open source files only for the ones it names.",
        "",
        f"## Host-marked nodes ({c['host_marked_nodes']})",
        "",
        "These carry `pytest.mark.llk_host`, so they seal to the host wrapper. "
        "Anything not listed here seals to silicon and needs compiled artifacts.",
        "",
    ]
    lines += [f"- `{n}`" for n in document["host_marked_nodes"]] or ["- (none)"]
    lines += ["", "## Test modules by marker", ""]
    by_marker: dict[str, list[str]] = {}
    for m in document["test_modules"]:
        for marker in m["markers"] or ["(unmarked)"]:
            by_marker.setdefault(marker, []).append(m["module"])
    for marker in sorted(by_marker):
        mods = sorted(set(by_marker[marker]))
        shown = ", ".join(f"`{x}`" for x in mods[:12])
        more = f" +{len(mods) - 12} more" if len(mods) > 12 else ""
        lines.append(f"- **{marker}** ({len(mods)}): {shown}{more}")
    harness = document.get("harness") or {}
    if harness:
        lines += [
            "",
            "## Harness",
            "",
            f"`{harness['path']}` subcommands: "
            + ", ".join(f"`{s}`" for s in harness["subcommands"]),
            "",
            "Flags: " + ", ".join(f"`{f}`" for f in harness["flags"][:20]),
        ]
    lines += ["", "## LLK symbol index", ""]
    for arch, headers in sorted(document["llk_symbols_by_arch"].items()):
        total = sum(len(v) for v in headers.values())
        lines.append(f"- **{arch}**: {total} symbols in {len(headers)} headers")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worktree", required=True)
    p.add_argument("--out", help="write the full JSON index here")
    p.add_argument("--summary-out", help="write the compact markdown index here")
    p.add_argument("--cache-dir", help="reuse/store a map keyed by base commit")
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = p.parse_args(argv)

    worktree = Path(args.worktree)
    cached = None
    cache_path = None
    if args.cache_dir:
        base = _git(worktree, "rev-parse", "HEAD") or "unknown"
        cache_path = Path(args.cache_dir) / f"repo-map-{GENERATOR_VERSION}-{base}.json"
        # A dirty worktree is the candidate's own edits; the cache is keyed by base
        # only, so never serve or store a map built from a modified tree.
        dirty = bool(_git(worktree, "status", "--porcelain"))
        if cache_path.is_file() and not args.force and not dirty:
            try:
                cached = json.loads(cache_path.read_text())
            except (OSError, ValueError):
                cached = None

    document = cached or build_map(worktree)
    document["served_from_cache"] = bool(cached)
    if cache_path and not cached and not document["worktree_dirty"]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(document, indent=1) + "\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(render_summary(document))
    if not args.out and not args.summary_out:
        print(json.dumps(document["counts"], indent=1))
        print(
            f"map_id {document['map_id'][:16]} cached={document['served_from_cache']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

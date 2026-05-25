"""Python AST indexer for the tt-metal dependency graph.

Vertical-slice version. Walks one or more Python files / directories and emits:

    - nodes:    modules, classes, functions (with decorators & locations)
    - edges:    intra-Python call edges (caller -> callee, resolved by name
                in the module-local scope where possible)
    - refs:     unresolved dotted-name references to be stitched against the
                C++ binding map (e.g., `ttnn.add` references)
    - diagnostics

This is intentionally minimal: full Python name resolution is undecidable in
general. We resolve same-module names and `<module>.<attr>` chains; the
stitcher takes care of cross-language matching.

Usage:
    python py_index.py \
        --root /workspace/ttnn/ttnn \
        --file /workspace/ttnn/ttnn/operations/binary.py \
        --module-root /workspace/ttnn \
        --out /workspace/dep-graph/cache/py_index.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ─── helpers ────────────────────────────────────────────────────────────────


def module_dotted_path(file_path: Path, module_roots: list[Path]) -> str:
    """Return the dotted module path for `file_path`, anchored at the closest module_root.

    E.g. file=/workspace/ttnn/ttnn/operations/binary.py, root=/workspace/ttnn
         → "ttnn.operations.binary"
    """
    abspath = file_path.resolve()
    best: tuple[int, Path] | None = None
    for root in module_roots:
        root_abs = root.resolve()
        try:
            rel = abspath.relative_to(root_abs)
        except ValueError:
            continue
        # Prefer the most specific (longest) root.
        if best is None or len(str(root_abs)) > len(str(best[1])):
            best = (len(rel.parts), root_abs)
    if best is None:
        return abspath.stem
    root_abs = best[1]
    rel = abspath.relative_to(root_abs)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def attr_chain(node: ast.AST) -> list[str] | None:
    """Flatten ast.Attribute / ast.Name chains into a list of names.

    `ttnn.add` → ["ttnn", "add"]
    `ttnn.operations.core.add` → ["ttnn", "operations", "core", "add"]
    Anything else (subscript, call result, etc.) → None.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


def decorator_label(node: ast.expr) -> str:
    """Render a decorator AST as a string label."""
    if isinstance(node, ast.Call):
        chain = attr_chain(node.func)
        if chain:
            return "@" + ".".join(chain) + "(...)"
        return "@<expr>(...)"
    chain = attr_chain(node)
    if chain:
        return "@" + ".".join(chain)
    return "@<expr>"


# ─── data records ──────────────────────────────────────────────────────────


@dataclass
class Node:
    id: str
    language: str
    kind: str  # "module" | "function" | "method" | "class"
    name: str
    qualified_name: str
    file: str
    line_start: int
    line_end: int
    decorators: list[str] = field(default_factory=list)
    is_binding_caller: bool = False


@dataclass
class Edge:
    src: str
    dst: str
    kind: str  # "calls"
    site_file: str
    site_line: int


@dataclass
class PyRef:
    src: str
    target_chain: list[str]   # e.g. ["ttnn", "add"]
    site_file: str
    site_line: int
    kind: str   # "attr_access" | "call"


@dataclass
class PyRegistration:
    """A Python op registered under a stable name via `register_python_operation`.

    The two forms detected:
      1. Decorator on a FunctionDef:
            @ttnn.register_python_operation(name="ttnn.from_torch", ...)
            def from_torch(...): ...

      2. Call form (module-level):
            ttnn.register_python_operation(name="ttnn.X")(impl)

         where `impl` is either a local Python name or a dotted attribute
         chain like `ttnn._ttnn.operations.core.unsqueeze_to_4D`. For the
         attribute-chain case we leave resolution to the stitcher and
         record `impl_chain` instead of `impl_node_id`.
    """
    python_name: str              # the `name=` kwarg, e.g. "ttnn.from_torch"
    impl_node_id: str | None      # set when impl is a Python def in the same module
    impl_chain: list[str] | None  # set when impl is an attribute chain
    site_file: str
    site_line: int
    decorator_label: str          # e.g. "@ttnn.register_python_operation"


# ─── indexer ───────────────────────────────────────────────────────────────


REGISTRATION_FUNCS = ("register_python_operation",)  # callable suffix


def _registration_decorator(call: ast.Call) -> tuple[str, str] | None:
    """If `call` is a `*.register_python_operation(name="…", …)` invocation,
    return (name_value, decorator_label). Otherwise None.

    Accepts forms where the trailing attribute is one of REGISTRATION_FUNCS.
    """
    chain = attr_chain(call.func)
    if not chain or chain[-1] not in REGISTRATION_FUNCS:
        return None
    name_value: str | None = None
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            name_value = kw.value.value
            break
    if name_value is None:
        return None
    return name_value, "@" + ".".join(chain)


class FileIndexer(ast.NodeVisitor):
    def __init__(self, file_path: Path, module_dotted: str) -> None:
        self.file_path = str(file_path.resolve())
        self.module_dotted = module_dotted
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.refs: list[PyRef] = []
        self.registrations: list[PyRegistration] = []
        self.scope_stack: list[str] = []  # current node-id stack
        self.qualname_stack: list[str] = []  # for qualified names

    # ─ ids ─

    def _mk_id(self, kind: str, qualname: str, line: int) -> str:
        return f"py:{kind}:{self.module_dotted}.{qualname}:{line}"

    def _module_id(self) -> str:
        return f"py:module:{self.module_dotted}"

    # ─ entry ─

    def index(self, tree: ast.Module) -> None:
        # Module node
        last_line = max((n.lineno for n in ast.walk(tree) if hasattr(n, "lineno")), default=0)
        mod_id = self._module_id()
        self.nodes[mod_id] = Node(
            id=mod_id,
            language="python",
            kind="module",
            name=self.module_dotted.rsplit(".", 1)[-1] or self.module_dotted,
            qualified_name=self.module_dotted,
            file=self.file_path,
            line_start=1,
            line_end=last_line,
        )
        self.scope_stack.append(mod_id)
        self.qualname_stack.append("")
        for stmt in tree.body:
            self.visit(stmt)
        self.qualname_stack.pop()
        self.scope_stack.pop()

    # ─ statements that open a scope ─

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, kind="function")

    def _visit_func(self, node, kind: str) -> None:
        parent_qual = self.qualname_stack[-1]
        qualname = f"{parent_qual}.{node.name}" if parent_qual else node.name
        # Inside a class, kind="method"
        if parent_qual and self.nodes[self.scope_stack[-1]].kind == "class":
            kind = "method"
        nid = self._mk_id(kind, qualname, node.lineno)
        self.nodes[nid] = Node(
            id=nid,
            language="python",
            kind=kind,
            name=node.name,
            qualified_name=f"{self.module_dotted}.{qualname}",
            file=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            decorators=[decorator_label(d) for d in node.decorator_list],
        )
        # Decorator-form registration: @<chain>.register_python_operation(name="ttnn.X")
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                hit = _registration_decorator(d)
                if hit is not None:
                    name_value, label = hit
                    self.registrations.append(PyRegistration(
                        python_name=name_value,
                        impl_node_id=nid,
                        impl_chain=None,
                        site_file=self.file_path,
                        site_line=d.lineno,
                        decorator_label=label,
                    ))
                    self.nodes[nid].is_binding_caller = True
        self.scope_stack.append(nid)
        self.qualname_stack.append(qualname)
        for stmt in node.body:
            self.visit(stmt)
        self.qualname_stack.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent_qual = self.qualname_stack[-1]
        qualname = f"{parent_qual}.{node.name}" if parent_qual else node.name
        nid = self._mk_id("class", qualname, node.lineno)
        self.nodes[nid] = Node(
            id=nid,
            language="python",
            kind="class",
            name=node.name,
            qualified_name=f"{self.module_dotted}.{qualname}",
            file=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            decorators=[decorator_label(d) for d in node.decorator_list],
        )
        self.scope_stack.append(nid)
        self.qualname_stack.append(qualname)
        for stmt in node.body:
            self.visit(stmt)
        self.qualname_stack.pop()
        self.scope_stack.pop()

    # ─ expressions: calls and attribute accesses ─

    def _emit_ref(self, src: str, chain: list[str], site_line: int, kind: str) -> None:
        # Dedup on (src, chain, line, kind) — avoids the common case where an
        # attribute appears both as a Call's func and as part of an arg walk.
        sig = (src, tuple(chain), site_line, kind)
        if sig in self._seen_refs:
            return
        self._seen_refs.add(sig)
        self.refs.append(PyRef(
            src=src,
            target_chain=chain,
            site_file=self.file_path,
            site_line=site_line,
            kind=kind,
        ))
        if len(chain) >= 2 and chain[0] in ("ttnn", "tt_metal"):
            self.nodes[src].is_binding_caller = True

    def visit_Call(self, node: ast.Call) -> None:
        chain = attr_chain(node.func)
        src = self.scope_stack[-1]
        if chain:
            self._emit_ref(src, chain, node.lineno, "call")
        # Call-form registration: <chain>.register_python_operation(name="…")(impl)
        # Detected here when node.func is itself a Call to register_python_operation
        # and node.args has exactly one positional argument (the impl).
        if isinstance(node.func, ast.Call) and len(node.args) == 1:
            hit = _registration_decorator(node.func)
            if hit is not None:
                name_value, label = hit
                impl = node.args[0]
                impl_chain = attr_chain(impl)
                impl_id: str | None = None
                if impl_chain and len(impl_chain) == 1:
                    # Bare Name: try to resolve to a same-module def already seen.
                    target_name = impl_chain[0]
                    for nid, n in self.nodes.items():
                        if n.kind in ("function", "method") and n.name == target_name:
                            impl_id = nid
                            break
                self.registrations.append(PyRegistration(
                    python_name=name_value,
                    impl_node_id=impl_id,
                    impl_chain=impl_chain if impl_id is None else None,
                    site_file=self.file_path,
                    site_line=node.lineno,
                    decorator_label=label + "(call-form)",
                ))
        # Recurse normally into all children (func + args + keywords). visit_Attribute
        # will dedup the func-as-attribute against the call ref above.
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Attribute access. Captures things like the `ttnn.add` argument in
        # `ttnn.attach_golden_function(ttnn.add, …)`.
        chain = attr_chain(node)
        if chain and len(chain) >= 2 and chain[0] in ("ttnn", "tt_metal"):
            src = self.scope_stack[-1]
            self._emit_ref(src, chain, node.lineno, "attr_access")
        self.generic_visit(node)

    _seen_refs: set


def index_file(file_path: Path, module_roots: list[Path]) -> FileIndexer:
    src = file_path.read_text()
    tree = ast.parse(src, filename=str(file_path))
    module_dotted = module_dotted_path(file_path, module_roots)
    idx = FileIndexer(file_path, module_dotted)
    idx._seen_refs = set()
    idx.index(tree)
    return idx


# ─── main ──────────────────────────────────────────────────────────────────


def _gather_files(args) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for f in args.file:
        p = Path(f).resolve()
        if p not in seen:
            seen.add(p)
            files.append(p)
    for d in args.dir:
        root = Path(d).resolve()
        for p in root.rglob("*.py"):
            # Skip tests + __pycache__ + venv etc.
            sp = str(p)
            if "/tests/" in sp or "/__pycache__/" in sp or "/.tox/" in sp:
                continue
            if p not in seen:
                seen.add(p)
                files.append(p)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=[], help="Python file to index (repeatable)")
    ap.add_argument("--dir", action="append", default=[], help="Directory to walk for .py files (repeatable)")
    ap.add_argument(
        "--module-root",
        action="append",
        default=[],
        help="Root directory whose immediate subdirs are top-level package names (repeatable)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = _gather_files(args)
    if not files:
        print("error: at least one --file or --dir is required", file=sys.stderr)
        sys.exit(2)
    roots = [Path(r) for r in args.module_root] or [Path(".")]

    all_nodes: dict[str, Node] = {}
    all_edges: list[Edge] = []
    all_refs: list[PyRef] = []
    all_regs: list[PyRegistration] = []
    diags: list[dict] = []

    for fp in files:
        try:
            idx = index_file(fp, roots)
        except SyntaxError as e:
            diags.append({"file": str(fp), "error": f"syntax: {e}"})
            continue
        except Exception as e:
            diags.append({"file": str(fp), "error": f"{type(e).__name__}: {e}"})
            continue
        all_nodes.update(idx.nodes)
        all_edges.extend(idx.edges)
        all_refs.extend(idx.refs)
        all_regs.extend(idx.registrations)

    out = {
        "nodes": [asdict(n) for n in all_nodes.values()],
        "edges": [asdict(e) for e in all_edges],
        "refs": [asdict(r) for r in all_refs],
        "registrations": [asdict(r) for r in all_regs],
        "diagnostics": diags,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"[py_index] {len(files)} files | {len(out['nodes'])} nodes, "
        f"{len(out['edges'])} edges, {len(out['refs'])} refs, "
        f"{len(out['registrations'])} registrations, {len(out['diagnostics'])} diags -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

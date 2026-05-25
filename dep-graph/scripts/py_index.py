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
class ImportEntry:
    """A single name brought into a file via `import` or `from ... import`.

    Relative imports are resolved to absolute module dotted paths up-front so
    later resolution doesn't need to know the current file's module context.
    """
    local_name: str          # name bound in this file (after `as`)
    target_module: str       # absolute dotted module path
    target_name: str | None  # the name in the source module; None means "module itself"
    is_module: bool          # True for `import X`, False for `from X import Y`
    line: int


def _resolve_relative_module(current_module_dotted: str, level: int, module: str | None) -> str | None:
    """Resolve a `from .foo import bar`-style relative module reference.

    `from . import x`           in pkg.sub.mod  → "pkg.sub"
    `from .core import x`       in pkg.sub.mod  → "pkg.sub.core"
    `from ..parent import x`    in pkg.sub.mod  → "pkg.parent"
    """
    parts = current_module_dotted.split(".")
    # `level` counts dots; level=1 means "current package" (drop the leaf module name).
    if level > len(parts):
        return None
    base = parts[:-level] if level > 0 else parts
    if module:
        return ".".join(base + module.split("."))
    return ".".join(base) if base else None


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

        # Resolution tables for intra-Python call edges.
        #   module_defs:   bare-name → node_id for module-level functions/classes.
        #                  Last def wins (matches Python's name-rebinding semantics).
        #   class_methods: class_qualname → {method_name: method_node_id}
        #                  Used to resolve `self.X` / `cls.X` calls.
        #   imports:       local_name → ImportEntry; only module-level imports
        #                  are recorded (function-local imports are ignored).
        self.module_defs: dict[str, str] = {}
        self.class_methods: dict[str, dict[str, str]] = {}
        self.imports: dict[str, ImportEntry] = {}

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
        # Post-pass: resolve bare-name and self.method refs into real edges.
        self._resolve_local_refs()

    # ─ statements that open a scope ─

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, kind="function")

    def _visit_func(self, node, kind: str) -> None:
        parent_qual = self.qualname_stack[-1]
        qualname = f"{parent_qual}.{node.name}" if parent_qual else node.name
        # Inside a class, kind="method"
        parent_is_class = (
            parent_qual and self.nodes[self.scope_stack[-1]].kind == "class"
        )
        if parent_is_class:
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
        # Resolution table tracking:
        #   - module-level functions/classes go into module_defs (bare-name lookup).
        #   - class-level methods go into class_methods[<class qualname>] (self.X lookup).
        if len(self.scope_stack) == 1:
            # Module top level — overwrite last-wins.
            self.module_defs[node.name] = nid
        elif parent_is_class:
            class_qn = self.nodes[self.scope_stack[-1]].qualified_name
            self.class_methods.setdefault(class_qn, {})[node.name] = nid
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
        if len(self.scope_stack) == 1:
            self.module_defs[node.name] = nid
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

    # ─── imports (module-level only) ──────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        # Only top-level `import X` / `import X as Y`. Function-local imports
        # are ignored because their scope is too narrow to model usefully.
        if len(self.scope_stack) != 1:
            return
        for alias in node.names:
            # `import a.b.c` binds the *top* name `a` in the file's namespace.
            local = alias.asname or alias.name.split(".")[0]
            self.imports[local] = ImportEntry(
                local_name=local,
                target_module=alias.name,
                target_name=None,
                is_module=True,
                line=node.lineno,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if len(self.scope_stack) != 1:
            return
        if node.level:
            target_module = _resolve_relative_module(self.module_dotted, node.level, node.module)
        else:
            target_module = node.module
        if not target_module:
            return
        for alias in node.names:
            if alias.name == "*":
                continue  # wildcard imports: skipped; would need __all__ tracking
            local = alias.asname or alias.name
            self.imports[local] = ImportEntry(
                local_name=local,
                target_module=target_module,
                target_name=alias.name,
                is_module=False,
                line=node.lineno,
            )

    # ─── post-pass: resolve intra-module Python edges ─────────────────────

    def _enclosing_class_qn(self, node_id: str) -> str | None:
        """For a method node, return its enclosing class's qualified_name.

        Heuristic: if the method's qualified_name is `<mod>.<Class>.<method>`,
        the parent is `<mod>.<Class>`. Walks up by stripping the trailing
        component and checking whether the result names a recorded class.
        """
        n = self.nodes.get(node_id)
        if n is None or n.kind != "method":
            return None
        qn = n.qualified_name
        # Drop the trailing ".<method>" — also strip module dotted prefix to get
        # the bare class qualname relative to the module.
        if "." not in qn:
            return None
        parent_qn = qn.rsplit(".", 1)[0]
        # Verify parent is actually a class node we've seen.
        for nid, nn in self.nodes.items():
            if nn.kind == "class" and nn.qualified_name == parent_qn:
                return parent_qn
        return None

    def _resolve_local_refs(self) -> None:
        """Turn bare-name and self.method refs into real Python `calls` edges.

        Refs that resolve become edges and are dropped from `self.refs`.
        Refs that don't resolve (e.g. `torch.relu`, `ttnn.add`) are left in
        place for the cross-language stitcher to handle.
        """
        kept: list[PyRef] = []
        for r in self.refs:
            chain = r.target_chain
            target_id: str | None = None

            if len(chain) == 1:
                # Bare name: look up against module top-level defs.
                target_id = self.module_defs.get(chain[0])
            elif len(chain) == 2 and chain[0] in ("self", "cls"):
                # self.X / cls.X: look up against the enclosing class's methods.
                class_qn = self._enclosing_class_qn(r.src)
                if class_qn:
                    target_id = self.class_methods.get(class_qn, {}).get(chain[1])

            if target_id and target_id != r.src:
                self.edges.append(Edge(
                    src=r.src,
                    dst=target_id,
                    kind="calls" if r.kind == "call" else "binds",
                    site_file=r.site_file,
                    site_line=r.site_line,
                ))
                continue   # resolved; do not keep as a ref

            kept.append(r)
        self.refs = kept

    def resolve_cross_module_refs(
        self, module_defs_global: dict[tuple[str, str], str]
    ) -> int:
        """Resolve remaining refs against this file's imports.

        Patterns handled:
          - `from X import Y` then `Y(...)`           — chain == [Y]
          - `from X import Y as Z` then `Z(...)`       — chain == [Z]
          - `import X` then `X.attr(...)`              — chain == [X, attr]
          - `import X as A` then `A.attr(...)`         — chain == [A, attr]

        Returns the number of newly emitted edges.
        """
        kept: list[PyRef] = []
        emitted = 0
        for r in self.refs:
            chain = r.target_chain
            if not chain:
                kept.append(r)
                continue
            imp = self.imports.get(chain[0])
            target_id: str | None = None

            if imp is None:
                pass  # nothing imported under this name
            elif imp.is_module:
                # `import X` then `X.attr(...)` — chain[1] is the attribute in X.
                if len(chain) >= 2:
                    target_id = module_defs_global.get((imp.target_module, chain[1]))
            else:
                # `from X import Y` then `Y(...)` — chain[0] alias resolves to (X, Y).
                if len(chain) == 1:
                    assert imp.target_name is not None
                    target_id = module_defs_global.get((imp.target_module, imp.target_name))
                # `from X import Y` then `Y.method(...)` — needs per-class method
                # globals which we don't yet track across files. Drop for now.

            if target_id and target_id != r.src:
                self.edges.append(Edge(
                    src=r.src, dst=target_id,
                    kind="calls" if r.kind == "call" else "binds",
                    site_file=r.site_file, site_line=r.site_line,
                ))
                emitted += 1
                continue

            kept.append(r)
        self.refs = kept
        return emitted


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

    # Pass 1: index each file (collects nodes, intra-module edges, refs, imports).
    indexers: list[FileIndexer] = []
    for fp in files:
        try:
            idx = index_file(fp, roots)
        except SyntaxError as e:
            diags.append({"file": str(fp), "error": f"syntax: {e}"})
            continue
        except Exception as e:
            diags.append({"file": str(fp), "error": f"{type(e).__name__}: {e}"})
            continue
        indexers.append(idx)

    # Build global module-level def table: (module_dotted, name) → node_id.
    # Only direct module-level functions/classes are reachable as cross-module imports;
    # nested defs and methods are excluded (the latter would need self/cls handling).
    module_defs_global: dict[tuple[str, str], str] = {}
    for fi in indexers:
        for nid, n in fi.nodes.items():
            if n.kind in ("function", "class"):
                # Only direct module-level defs — qualified_name shape is "<module_dotted>.<name>".
                if n.qualified_name == f"{fi.module_dotted}.{n.name}":
                    module_defs_global[(fi.module_dotted, n.name)] = nid

    # Re-export pass: an `__init__.py` that does `from .core import foo` makes
    # `foo` available as `<package>.foo`. Propagate these aliases into the
    # global table so cross-package callers resolve.
    # Iterates to a fixed point (capped) to handle chained re-exports.
    reexport_iters = 0
    while reexport_iters < 5:
        reexport_iters += 1
        changed = False
        for fi in indexers:
            for local_name, imp in fi.imports.items():
                if imp.is_module:
                    continue  # `import X` doesn't re-export anything
                assert imp.target_name is not None
                target = module_defs_global.get((imp.target_module, imp.target_name))
                if target is None:
                    continue
                key = (fi.module_dotted, local_name)
                if module_defs_global.get(key) != target:
                    module_defs_global[key] = target
                    changed = True
        if not changed:
            break

    # Pass 2: cross-module resolution. Each file resolves its remaining refs
    # against its own import table + the global def table.
    xmod_total = 0
    for fi in indexers:
        xmod_total += fi.resolve_cross_module_refs(module_defs_global)

    # Aggregate.
    for fi in indexers:
        all_nodes.update(fi.nodes)
        all_edges.extend(fi.edges)
        all_refs.extend(fi.refs)
        all_regs.extend(fi.registrations)

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
        f"{len(out['edges'])} edges ({xmod_total} via cross-module imports), "
        f"{len(out['refs'])} refs, {len(out['registrations'])} registrations, "
        f"{len(out['diagnostics'])} diags -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

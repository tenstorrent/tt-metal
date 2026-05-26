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
from collections import defaultdict
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
    # 0-based column of the LAST component of the chain (the actual symbol
    # being referenced). Needed for Jedi's line/col position queries — without
    # this, the resolver can't pinpoint `execute` in `cursor.execute(...)`.
    site_col: int = 0


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
        #   dunder_all:    explicit `__all__ = [...]` declaration at module level,
        #                  or None if not defined. Used when this module is the
        #                  target of a `from <self> import *` in another file.
        #   wildcard_imports: target modules of `from X import *` statements;
        #                  expanded into concrete ImportEntry records in a
        #                  post-pass once every module's public names are known.
        self.module_defs: dict[str, str] = {}
        self.class_methods: dict[str, dict[str, str]] = {}
        self.imports: dict[str, ImportEntry] = {}
        self.dunder_all: list[str] | None = None
        self.wildcard_imports: list[tuple[str, int]] = []
        # class node id → list of (base_chain, lineno) for B6 inherits resolution.
        self.class_base_chains: dict[str, list[tuple[list[str], int]]] = {}

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
        # Visit the function body for calls/refs.
        for stmt in node.body:
            self.visit(stmt)
        # Also visit parameter annotations, return annotation, default values,
        # and decorator argument expressions. Without these, references inside
        # type hints like `def foo(x: ttnn.MeshDevice = None) -> ttnn.Tensor`
        # would silently not produce edges.
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if args.vararg and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)
        for default in args.defaults + args.kw_defaults:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        for dec in node.decorator_list:
            # Visit the decorator expression itself. Registration decorators are
            # already matched separately (above) by structural pattern; visiting
            # their AST again is harmless — _emit_ref dedups.
            self.visit(dec)
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
        # B6: capture base-class chains for cross-file inheritance resolution.
        base_chains: list[tuple[list[str], int]] = []
        for base in node.bases:
            chain = attr_chain(base)
            if chain:
                base_chains.append((chain, base.lineno))
        if base_chains:
            self.class_base_chains[nid] = base_chains
        self.scope_stack.append(nid)
        self.qualname_stack.append(qualname)
        for stmt in node.body:
            self.visit(stmt)
        # Visit base-class expressions and decorator expressions so refs
        # they contain (e.g. `class X(ttnn.SomeBase)`) become edges. The
        # caller for these refs is the class node itself; that matches
        # the natural "X inherits from Y" reading even before we add a
        # dedicated `inherits` edge kind (Phase 3 B6).
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            if kw.value is not None:
                self.visit(kw.value)
        for dec in node.decorator_list:
            self.visit(dec)
        self.qualname_stack.pop()
        self.scope_stack.pop()

    # ─ expressions: calls and attribute accesses ─

    def _emit_ref(
        self, src: str, chain: list[str], site_line: int, kind: str, site_col: int = 0,
    ) -> None:
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
            site_col=site_col,
            kind=kind,
        ))
        if len(chain) >= 2 and chain[0] in ("ttnn", "tt_metal"):
            self.nodes[src].is_binding_caller = True

    def visit_Call(self, node: ast.Call) -> None:
        chain = attr_chain(node.func)
        src = self.scope_stack[-1]
        if chain:
            # For Attribute chains (`obj.method`), use the column of the LAST
            # component — that's the position Jedi needs to resolve the method.
            # For a bare Name, the start column is fine.
            tgt = node.func
            if isinstance(tgt, ast.Attribute) and tgt.end_col_offset is not None:
                # Position one char inside the trailing identifier.
                col = max(0, tgt.end_col_offset - 1)
                line = tgt.end_lineno or node.lineno
            else:
                col = getattr(tgt, "col_offset", 0)
                line = node.lineno
            self._emit_ref(src, chain, line, "call", site_col=col)
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
            col = max(0, (node.end_col_offset or node.col_offset) - 1)
            line = node.end_lineno or node.lineno
            self._emit_ref(src, chain, line, "attr_access", site_col=col)
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
                # Wildcards are queued here; the global post-pass expands them
                # using each target module's public-name set (`__all__` or
                # the inferred public top-level names).
                self.wildcard_imports.append((target_module, node.lineno))
                continue
            local = alias.asname or alias.name
            self.imports[local] = ImportEntry(
                local_name=local,
                target_module=target_module,
                target_name=alias.name,
                is_module=False,
                line=node.lineno,
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only module-level `__all__ = [...]` / `__all__ = (...)` assignments
        # count. Function-local or class-local assignments to a name `__all__`
        # are not Python's export-list semantics.
        if len(self.scope_stack) == 1:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.append(elt.value)
                        self.dunder_all = names
        self.generic_visit(node)

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

    def resolve_self_method_via_mro(
        self,
        class_methods_global: dict[str, dict[str, str]],
        class_parents: dict[str, list[str]],
    ) -> int:
        """A4: walk parent classes (via class_parents) when resolving self.X.

        Refs of shape `["self", "X"]` or `["cls", "X"]` that weren't resolved
        in the immediate class get a second chance here: we walk up the
        recorded parents (BFS, capped depth) looking for a matching method.
        """
        kept: list[PyRef] = []
        emitted = 0
        for r in self.refs:
            chain = r.target_chain
            if not (len(chain) == 2 and chain[0] in ("self", "cls")):
                kept.append(r)
                continue
            class_qn = self._enclosing_class_qn(r.src)
            if not class_qn:
                kept.append(r)
                continue
            # Look up the class's node id locally so we can walk class_parents.
            class_node_id: str | None = None
            for nid, n in self.nodes.items():
                if n.kind == "class" and n.qualified_name == class_qn:
                    class_node_id = nid
                    break
            if class_node_id is None:
                kept.append(r)
                continue

            method_name = chain[1]
            # BFS up the parent chain.
            seen: set[str] = {class_node_id}
            frontier: list[str] = [class_node_id]
            target_id: str | None = None
            depth = 0
            while frontier and depth < 6 and target_id is None:
                next_frontier: list[str] = []
                for cid in frontier:
                    methods = class_methods_global.get(cid, {})
                    if method_name in methods:
                        target_id = methods[method_name]
                        break
                    for parent in class_parents.get(cid, []):
                        if parent not in seen:
                            seen.add(parent)
                            next_frontier.append(parent)
                frontier = next_frontier
                depth += 1

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


def _expand_wildcard_imports(indexers: list["FileIndexer"]) -> int:
    """Resolve `from X import *` statements to concrete ImportEntry records.

    A module's public name set is:
      - its explicit `__all__` if defined, else
      - every top-level def/class/import-alias name that doesn't start with `_`.

    Wildcard expansion runs once (no fixed-point) — the only chained-wildcard
    case in tt-metal would be A doing `import *` from B, which itself does
    `import *` from C. Not observed; if it ever shows up, iterate this pass.
    """
    by_module = {fi.module_dotted: fi for fi in indexers}

    def public_names(fi: "FileIndexer") -> list[str]:
        if fi.dunder_all is not None:
            return list(fi.dunder_all)
        names: set[str] = set()
        # Module-level defs.
        for nid, n in fi.nodes.items():
            if (
                n.kind in ("function", "class")
                and n.qualified_name == f"{fi.module_dotted}.{n.name}"
                and not n.name.startswith("_")
            ):
                names.add(n.name)
        # Import aliases bound in this module.
        for local_name in fi.imports:
            if not local_name.startswith("_"):
                names.add(local_name)
        return sorted(names)

    public_map = {fi.module_dotted: public_names(fi) for fi in indexers}
    total_expanded = 0
    for fi in indexers:
        for target_mod, line in fi.wildcard_imports:
            names = public_map.get(target_mod)
            if not names:
                continue
            for name in names:
                if name in fi.imports:
                    continue   # do not shadow an explicit import in this file
                fi.imports[name] = ImportEntry(
                    local_name=name,
                    target_module=target_mod,
                    target_name=name,
                    is_module=False,
                    line=line,
                )
                total_expanded += 1
    return total_expanded


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

    # Wildcard import expansion. Must run BEFORE the re-export pass so that
    # wildcard-imported names are available for further propagation.
    wildcards_expanded = _expand_wildcard_imports(indexers)

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

    # Build a shared chain resolver used by both B5 (imports edges) and B6
    # (inherits edges). Resolution semantics match resolve_cross_module_refs:
    # bare names look up in local module_defs, dotted-chain names go through
    # the file's imports table + the global def table.
    def _resolve_chain_to_node(fi: "FileIndexer", chain: list[str]) -> str | None:
        if not chain:
            return None
        if len(chain) == 1:
            local = fi.module_defs.get(chain[0])
            if local:
                return local
        imp = fi.imports.get(chain[0])
        if imp is None:
            return None
        if imp.is_module:
            if len(chain) >= 2:
                return module_defs_global.get((imp.target_module, chain[1]))
            return None
        # `from X import Y` → chain[0] is the alias Y
        assert imp.target_name is not None
        if len(chain) == 1:
            return module_defs_global.get((imp.target_module, imp.target_name))
        return None

    # Pass 3: emit `imports` edges. Each ImportEntry on a file becomes an edge
    # from that file's module-node to the imported entity (function/class for
    # `from X import Y`, module-node for `import X` when X is in scope).
    module_id_by_dotted = {fi.module_dotted: fi._module_id() for fi in indexers}
    imports_edges = 0
    for fi in indexers:
        src_mod_id = fi._module_id()
        for local_name, imp in fi.imports.items():
            if imp.is_module:
                # `import X` — target is the module node, if we indexed it.
                dst = module_id_by_dotted.get(imp.target_module)
            else:
                # `from X import Y` — target is the entity Y in module X.
                assert imp.target_name is not None
                dst = module_defs_global.get((imp.target_module, imp.target_name))
            if not dst or dst == src_mod_id:
                continue
            fi.edges.append(Edge(
                src=src_mod_id,
                dst=dst,
                kind="imports",
                site_file=fi.file_path,
                site_line=imp.line,
            ))
            imports_edges += 1

    # Pass 3.5 (B4): Jedi-backed type-aware resolution. For each ref that
    # remains unresolved after cross-module + re-export passes, query Jedi
    # at the ref's (line, col) for the actual symbol it resolves to. If the
    # resolution's `full_name` matches a node in our `qualified_name` table,
    # emit a real edge. This catches `receiver.method()`-style calls where
    # `receiver` is a typed parameter or a locally-constructed instance.
    from py_type_resolver import resolve_refs, jedi_available  # type: ignore
    qn_to_node: dict[str, str] = {}
    for fi in indexers:
        for nid, n in fi.nodes.items():
            qn_to_node[n.qualified_name] = nid
    b4_edges = 0
    if jedi_available():
        # Group remaining refs by file so we open each Jedi Script once.
        # Skip refs we definitely don't want Jedi for:
        #   - chains starting with ttnn/tt_metal (cross-language stitcher)
        #   - chains starting with self/cls (MRO pass)
        #   - chain length 1 (bare names — already resolved or unresolvable)
        refs_by_file: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        ref_index_map: dict[tuple[str, int, int], list[tuple["FileIndexer", int]]] = defaultdict(list)
        for fi in indexers:
            for ridx, r in enumerate(fi.refs):
                chain = r.target_chain
                if not chain:
                    continue
                if chain[0] in ("ttnn", "tt_metal"):
                    continue
                if chain[0] in ("self", "cls"):
                    continue
                if len(chain) < 2:
                    continue
                if r.site_col <= 0 or r.site_line <= 0:
                    continue
                key = (r.site_file, r.site_line, r.site_col)
                refs_by_file[r.site_file].append((ridx, r.site_line, r.site_col))
                ref_index_map[key].append((fi, ridx))

        # Pass the first module-root as Jedi's project root so the resolved
        # full_name matches our `qualified_name` (e.g. `ttnn.operations.binary`
        # rather than Jedi's default `ttnn.ttnn.operations.binary`).
        project_root = str(roots[0]) if roots else None
        resolved = resolve_refs(refs_by_file, project_root=project_root)
        # Walk every fi's refs; for those that resolved, emit an edge and mark
        # the ref for deletion. Build a per-fi index-set of resolved positions.
        resolved_positions_per_fi: dict[id, set[int]] = defaultdict(set)
        for (file_path, line, col), full_name in resolved.items():
            node_id_match = qn_to_node.get(full_name)
            if node_id_match is None:
                continue
            for fi, ridx in ref_index_map[(file_path, line, col)]:
                r = fi.refs[ridx]
                if node_id_match == r.src:
                    continue
                fi.edges.append(Edge(
                    src=r.src,
                    dst=node_id_match,
                    kind="calls" if r.kind == "call" else "binds",
                    site_file=r.site_file,
                    site_line=r.site_line,
                ))
                b4_edges += 1
                resolved_positions_per_fi[id(fi)].add(ridx)
        # Drop resolved refs from each fi.refs to avoid double-counting later.
        for fi in indexers:
            resolved_set = resolved_positions_per_fi.get(id(fi), set())
            if resolved_set:
                fi.refs = [r for i, r in enumerate(fi.refs) if i not in resolved_set]

    # Pass 4: resolve class bases → emit `inherits` edges and build the
    # global class_parents map (consumed by A4's MRO walk for self.method).
    class_parents: dict[str, list[str]] = {}
    inherits_edges = 0
    for fi in indexers:
        for class_id, base_entries in fi.class_base_chains.items():
            for chain, line in base_entries:
                target = _resolve_chain_to_node(fi, chain)
                if not target or target == class_id:
                    continue
                fi.edges.append(Edge(
                    src=class_id,
                    dst=target,
                    kind="inherits",
                    site_file=fi.file_path,
                    site_line=line,
                ))
                class_parents.setdefault(class_id, []).append(target)
                inherits_edges += 1

    # Pass 5: MRO-aware `self.method` resolution (A4). After class_parents is
    # built, walk up the inheritance chain to resolve self.X / cls.X calls
    # that didn't match the immediate class. Each file's class_methods table
    # is consulted in MRO order.
    a4_edges = 0
    # Build a global class_methods view so we can resolve across files.
    class_methods_global: dict[str, dict[str, str]] = {}
    for fi in indexers:
        for class_qn, methods in fi.class_methods.items():
            # class_qn is a "module.Class" qualified name; map to its node id.
            class_node_id = None
            for nid, n in fi.nodes.items():
                if n.kind == "class" and n.qualified_name == class_qn:
                    class_node_id = nid
                    break
            if class_node_id is None:
                continue
            class_methods_global[class_node_id] = methods
    for fi in indexers:
        a4_edges += fi.resolve_self_method_via_mro(class_methods_global, class_parents)

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
        f"{len(out['edges'])} edges "
        f"({xmod_total} via cross-module imports, "
        f"{wildcards_expanded} wildcard names expanded, "
        f"{imports_edges} imports-edges, "
        f"{inherits_edges} inherits-edges, "
        f"{a4_edges} via MRO, "
        f"{b4_edges} via Jedi type-resolver), "
        f"{len(out['refs'])} refs, {len(out['registrations'])} registrations, "
        f"{len(out['diagnostics'])} diags -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

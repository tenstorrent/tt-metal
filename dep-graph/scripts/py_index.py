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


def quick_scan_for_types(file_path: Path, module_roots: list[Path]) -> tuple[
    list[tuple[str, str, str]],         # classes: (module_dotted, class_qualname, class_name)
    list[tuple[str, str, list[str]]],   # funcs:   (module_dotted, func_qualname, return_type_chain)
    list[tuple[str, str, list[str]]],   # module_vars: (module_dotted, var_name, type_chain)
]:
    """Fast pre-scan: collect class defs, function return annotations, and
    module-level variable type-inferences from constructor calls.

    Used by the type-propagator pre-pass to build global tables BEFORE the
    main per-file walks run. The main walk consults these globals to
    type-infer local-variable assignments like `x = Foo()` (constructor →
    type Foo) or `x = make_foo()` (function → type from `-> Foo` annotation).
    Module-level vars enable patterns like:
        # in utility_functions.py
        profiler = Profiler()
        # in other_file.py
        from utility_functions import profiler
        profiler.start()    # → resolves to Profiler.start

    Returns (classes, funcs, module_vars).
    """
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except Exception:
        return [], [], []
    module = module_dotted_path(file_path, module_roots)
    classes: list[tuple[str, str, str]] = []
    funcs: list[tuple[str, str, list[str]]] = []
    module_vars: list[tuple[str, str, list[str]]] = []

    def walk(node: ast.AST, qn_parts: list[str]) -> None:
        if isinstance(node, ast.ClassDef):
            new_parts = qn_parts + [node.name]
            classes.append((module, ".".join(new_parts), node.name))
            for ch in node.body:
                walk(ch, new_parts)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_parts = qn_parts + [node.name]
            if node.returns is not None:
                rt = _peel_receiver_type(node.returns)
                if rt:
                    funcs.append((module, ".".join(new_parts), rt))
            for ch in node.body:
                walk(ch, new_parts)

    for ch in tree.body:
        walk(ch, [])

    # Module-level `var = ClassName(...)` pattern. Only direct, top-level
    # assignments — no walking into if/else, try, etc.
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            if isinstance(stmt.value, ast.Call):
                fn_chain = attr_chain(stmt.value.func)
                if fn_chain:
                    module_vars.append((module, target, fn_chain))
    return classes, funcs, module_vars


def load_pytype_stubs(stub_root: Path) -> dict[tuple[str, str], dict[str, list[str]]]:
    """Parse pytype-generated .pyi stubs to extract inferred parameter types.

    Pytype produces .pyi stub files under `pyi/<package>/<module>.pyi` with
    function signatures whose parameter annotations include types pytype
    INFERRED — even when the source had no annotation. We harvest those to
    fill in `local_types` for unannotated parameters.

    Returns dict[(module_dotted, function_qualname)] → {param_name: type_chain}
    where qualname is dotted (e.g. "ClassName.method" or "function").

    Annotation types that resolve to `Any`, `nothing`, or single-letter
    generic typevars (`_T0`) are skipped — they don't help resolve methods.
    """
    out: dict[tuple[str, str], dict[str, list[str]]] = {}
    pyi_dir = stub_root / "pyi"
    if not pyi_dir.exists():
        return out

    def collect(node: ast.AST, qualname_parts: list[str], module_dotted: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join(qualname_parts + [node.name])
            params: dict[str, list[str]] = {}
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.annotation is None:
                    continue
                rt = _peel_receiver_type(arg.annotation)
                if not rt:
                    continue
                # Skip uninformative inferences.
                last = rt[-1]
                if last in ("Any", "nothing"):
                    continue
                if len(last) <= 3 and last.startswith("_T") and last[2:].isdigit():
                    continue  # generic typevar like _T0
                params[arg.arg] = rt
            if params:
                out[(module_dotted, qualname)] = params
            # Nested functions/classes can still have stub entries.
            for child in node.body:
                collect(child, qualname_parts + [node.name], module_dotted)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                collect(child, qualname_parts + [node.name], module_dotted)

    for pyi_file in pyi_dir.rglob("*.pyi"):
        rel = pyi_file.relative_to(pyi_dir)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        module_dotted = ".".join(parts)
        if not module_dotted:
            continue
        try:
            tree = ast.parse(pyi_file.read_text(), filename=str(pyi_file))
        except Exception:
            continue
        for child in tree.body:
            collect(child, [], module_dotted)
    return out


def _peel_receiver_type(ann: ast.expr | None) -> list[str] | None:
    """Extract a single dotted type chain from a (possibly wrapped) annotation.

    Handles:
        ttnn.MemoryConfig           → ["ttnn", "MemoryConfig"]
        Optional[ttnn.Tensor]       → ["ttnn", "Tensor"]
        Union[ttnn.Tensor, None]    → ["ttnn", "Tensor"]
        ttnn.Tensor | None          → ["ttnn", "Tensor"]
        list[ttnn.Tensor]           → None  (the var is a list, not Tensor)
        str / int / object          → ["str"] etc. (caller filters externals)

    Returns None when the annotation is something we can't reduce to a single
    receiver type (Union of incompatibles, generics, callables, etc.).
    """
    if ann is None:
        return None
    chain = attr_chain(ann)
    if chain:
        return chain
    # Subscript: Optional[X], Union[X, None], etc.
    if isinstance(ann, ast.Subscript):
        outer = attr_chain(ann.value)
        if outer in (["Optional"], ["typing", "Optional"]):
            return _peel_receiver_type(ann.slice)
        if outer in (["Union"], ["typing", "Union"]):
            slice_node = ann.slice
            # ast.Tuple under Subscript for Union[X, Y, ...]
            if isinstance(slice_node, ast.Tuple):
                non_none = [e for e in slice_node.elts
                            if not (isinstance(e, ast.Constant) and e.value is None)
                            and attr_chain(e) != ["None"]]
                if len(non_none) == 1:
                    return _peel_receiver_type(non_none[0])
        return None  # list[X], dict[K, V], Callable[...] etc. — not a method receiver
    # PEP 604 BinOp: X | None
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        left = _peel_receiver_type(ann.left)
        right = _peel_receiver_type(ann.right)
        if right == ["None"]:
            return left
        if left == ["None"]:
            return right
        return left  # ambiguous union of types — pick the first
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
    # Type of the receiver (chain[0]) when it's a known annotated local /
    # parameter — e.g. `memory_config: ttnn.MemoryConfig` makes this
    # `["ttnn", "MemoryConfig"]`. The stitcher uses receiver_type[-1] to
    # look up the receiver's class binding and resolve `chain[1]` as a
    # method/field on that class. None when the receiver type isn't known.
    receiver_type: list[str] | None = None


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


def _resolve_relative_module(
    current_module_dotted: str,
    level: int,
    module: str | None,
    is_package: bool = False,
) -> str | None:
    """Resolve a `from .foo import bar`-style relative module reference.

    `from . import x`           in pkg.sub.mod  → "pkg.sub"
    `from .core import x`       in pkg.sub.mod  → "pkg.sub.core"
    `from ..parent import x`    in pkg.sub.mod  → "pkg.parent"

    When the importer is itself a package (`pkg/__init__.py`, where
    module_dotted is already the package name `pkg.sub`), level=1 means
    the current package — *don't* strip the trailing part:
    `from .core import x` in pkg/sub/__init__.py → "pkg.sub.core" (not "pkg.core").
    """
    parts = current_module_dotted.split(".")
    # For non-packages, level=1 means parent package (strip leaf module name).
    # For packages (__init__.py), level=1 means the package itself (strip nothing).
    strip = (level - 1) if is_package else level
    if strip > len(parts):
        return None
    base = parts[:-strip] if strip > 0 else parts
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
    # Class-level registry of pytype-inferred types, set by index_file/_gather.
    # Maps (module_dotted, function_qualname) → {param_name: type_chain}.
    inferred_types: dict[tuple[str, str], dict[str, list[str]]] = {}

    # Class-name registry built by the quick pre-scan. Maps a bare class name
    # to its full qualified name(s). When we see `x = Foo(...)` in a function
    # body, we ask this map "is Foo a known class?" — if yes, infer x's type
    # as the class.
    classes_by_name: dict[str, list[str]] = {}

    # Function return-type registry built by the quick pre-scan. Maps
    # (module_dotted, function_qualname) → return type chain. When we see
    # `x = func(...)` in a function body, we look up func's return type and
    # type x accordingly.
    func_returns: dict[tuple[str, str], list[str]] = {}

    # Module-level variable type registry. Maps (module_dotted, var_name)
    # to a type chain inferred from a module-level constructor call:
    # `profiler = Profiler()` → ('utility_functions', 'profiler') → ['Profiler'].
    # When another file imports `profiler` and calls `profiler.X()`, the
    # main walk consults this map to attach receiver_type.
    module_var_types: dict[tuple[str, str], list[str]] = {}

    # Inter-procedural inference: populated by a POST-pass that looks at
    # actual call sites of each function and propagates known argument
    # types back to the callee's unannotated parameters. Keyed by
    # (function_node_id, param_name) → type_chain.
    inferred_param_types: dict[tuple[str, str], list[str]] = {}

    def __init__(self, file_path: Path, module_dotted: str) -> None:
        self.file_path = str(file_path.resolve())
        self.is_package = file_path.name == "__init__.py"
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

        # Per-function parameter names (positional + kw-only, in order),
        # keyed by function node_id. Used by inter-procedural inference to
        # map call-site arg positions to param names.
        self.func_params: dict[str, list[str]] = {}
        # Call sites seen in this file: (caller_func_id, target_call_chain,
        # positional_arg_types, keyword_arg_types). arg_types are
        # type_chains or None. caller_func_id is the function-node-id
        # containing the call (so we know the scope for ref updates).
        self.call_sites: list[tuple[str, list[str], list[list[str] | None], dict[str, list[str]]]] = []
        self.wildcard_imports: list[tuple[str, int]] = []
        # class node id → list of (base_chain, lineno) for B6 inherits resolution.
        self.class_base_chains: dict[str, list[tuple[list[str], int]]] = {}
        # Stack of {local_name → type_chain} maps — pushed when entering a
        # function/method scope, popped on exit. Populated from parameter
        # annotations and AnnAssign at function scope. Used by visit_Call /
        # visit_Attribute to attach `receiver_type` to refs so the stitcher
        # can resolve `obj.method()` through C++ class bindings.
        self.local_type_stack: list[dict[str, list[str]]] = []
        # Stack of {local_name → alias_chain} maps. For patterns like
        # `g = ttnn.get_golden_function(ttnn.gcd)` we record
        # `g → ['ttnn','gcd','golden_function']` so later `g(...)` calls
        # expand to that chain and resolve via the @attach_golden_function
        # PyRegistrations (Flaw 9).
        self.local_alias_stack: list[dict[str, list[str]]] = []
        # Per-class field type map. Populated by scanning __init__ for
        # `self.X = <typed_local>` assignments. Keyed by (class_qualname,
        # field_name) → type_chain. Used in method bodies to resolve
        # `self.X.method()` via the receiver_type mechanism.
        self.class_field_types: dict[tuple[str, str], list[str]] = {}
        # Stack of current class qualname (or "" when not in a class scope).
        self.class_qn_stack: list[str] = []

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
        # Build the local-type table from parameter annotations BEFORE walking
        # the body, so visit_Call / visit_Attribute can tag refs with
        # receiver_type.
        local_types: dict[str, list[str]] = {}
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if arg.annotation is not None:
                rt = _peel_receiver_type(arg.annotation)
                if rt:
                    local_types[arg.arg] = rt
        if args.vararg and args.vararg.annotation is not None:
            rt = _peel_receiver_type(args.vararg.annotation)
            if rt:
                local_types[args.vararg.arg] = rt
        if args.kwarg and args.kwarg.annotation is not None:
            rt = _peel_receiver_type(args.kwarg.annotation)
            if rt:
                local_types[args.kwarg.arg] = rt
        # For methods, `self` resolves to the enclosing class.
        if parent_is_class and args.args:
            # Heuristic: first arg of a method is `self` (or `cls`).
            first = args.args[0].arg
            if first in ("self", "cls"):
                # Use the class's bare name as the receiver type. The stitcher
                # will look it up in class_methods_global by python class name.
                cls_node = self.nodes[self.scope_stack[-1]]
                local_types[first] = [cls_node.name]
        # Pytype fallback: for parameters with no source annotation, use the
        # type pytype inferred. Source annotations always win — only fills
        # gaps. Keyed by (module_dotted, function_qualname).
        if FileIndexer.inferred_types:
            inferred_params = FileIndexer.inferred_types.get((self.module_dotted, qualname), {})
            for param_name, inferred_type in inferred_params.items():
                if param_name not in local_types:
                    local_types[param_name] = inferred_type
        # Type propagator: scan top-level body assignments and infer the
        # type of the LHS variable from the RHS expression.
        #   x = Foo()              → x: Foo            (constructor call)
        #   x = some_func(...)     → x: return-type    (if known)
        # Only top-level statements scanned — assignments in conditionals
        # are skipped for simplicity. Existing param annotations win.
        self._infer_assignment_types(node.body, local_types)
        # Apply inter-procedural-inferred param types when no source
        # annotation or pytype stub typed the param. Inferred from call
        # sites in a post-pass (see resolve_interprocedural_param_types()).
        if FileIndexer.inferred_param_types:
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.arg in local_types:
                    continue
                inferred = FileIndexer.inferred_param_types.get((nid, arg.arg))
                if inferred:
                    local_types[arg.arg] = inferred
        # Save param names so the inter-procedural inference pass can map
        # caller-arg positions back to parameter names.
        param_names: list[str] = []
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            param_names.append(arg.arg)
        self.func_params[nid] = param_names
        self.local_type_stack.append(local_types)
        # Pre-scan body for `var = ttnn...get_golden_function(<op>)`-style
        # alias assignments. This must happen before walking the body so
        # subsequent `var(...)` calls see the alias.
        local_aliases: dict[str, list[str]] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if isinstance(stmt.value, ast.Call):
                    fn_chain = attr_chain(stmt.value.func)
                    if fn_chain and fn_chain[-1] == "get_golden_function" and stmt.value.args:
                        op_chain = attr_chain(stmt.value.args[0])
                        if op_chain:
                            local_aliases[stmt.targets[0].id] = op_chain + ["golden_function"]
        self.local_alias_stack.append(local_aliases)
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
        self.local_type_stack.pop()
        self.local_alias_stack.pop()

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
        # Track current class qualname for __init__ scanning of `self.X` fields.
        class_qn = f"{self.module_dotted}.{qualname}"
        self.class_qn_stack.append(class_qn)
        # Pre-pass: locate __init__ and scan it for `self.X = typed_local`
        # assignments so OTHER methods in the same class can use the captured
        # field types regardless of class-body ordering.
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                init_local_types: dict[str, list[str]] = {}
                args = stmt.args
                for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                    if arg.annotation is not None:
                        rt = _peel_receiver_type(arg.annotation)
                        if rt:
                            init_local_types[arg.arg] = rt
                for body_stmt in stmt.body:
                    if isinstance(body_stmt, ast.Assign) and len(body_stmt.targets) == 1:
                        tgt = body_stmt.targets[0]
                        if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"):
                            if isinstance(body_stmt.value, ast.Name):
                                t = init_local_types.get(body_stmt.value.id)
                                if t:
                                    self.class_field_types[(class_qn, tgt.attr)] = t
                    elif isinstance(body_stmt, ast.AnnAssign) and body_stmt.target is not None:
                        tgt = body_stmt.target
                        if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"):
                            rt = _peel_receiver_type(body_stmt.annotation)
                            if rt:
                                self.class_field_types[(class_qn, tgt.attr)] = rt
                break  # only one __init__ per class
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
        self.class_qn_stack.pop()
        self.qualname_stack.pop()
        self.scope_stack.pop()

    # ─ expressions: calls and attribute accesses ─

    def _emit_ref(
        self, src: str, chain: list[str], site_line: int, kind: str, site_col: int = 0,
    ) -> None:
        # Local alias expansion: `g = ttnn.get_golden_function(ttnn.gcd)` then
        # `g(...)` — expand `chain=['g', ...]` to the recorded alias chain.
        if self.local_alias_stack and chain:
            alias = self.local_alias_stack[-1].get(chain[0])
            if alias is not None:
                chain = alias + chain[1:]
        # Field-type expansion: `self.X.Y(...)` where X is a tracked field of
        # the enclosing class. Drop the 'self' prefix and use the field's
        # type as receiver_type so the stitcher resolves Y as a method on
        # that type. Only fires when chain has at least 3 components.
        receiver_type: list[str] | None = None
        if (self.class_qn_stack and self.class_qn_stack[-1]
                and len(chain) >= 3 and chain[0] == "self"):
            field_t = self.class_field_types.get((self.class_qn_stack[-1], chain[1]))
            if field_t:
                chain = chain[1:]  # drop 'self'; chain[0] is now the field name
                receiver_type = field_t
        # Dedup on (src, chain, line, kind) — avoids the common case where an
        # attribute appears both as a Call's func and as part of an arg walk.
        sig = (src, tuple(chain), site_line, kind)
        if sig in self._seen_refs:
            return
        self._seen_refs.add(sig)
        # Receiver-type tagging (Option 2 typed-method resolver). If chain[0]
        # is a known annotated local in the current function scope, record
        # the receiver's declared type so the stitcher can resolve chain[1]
        # against the class's method/field bindings.
        if receiver_type is None and chain and self.local_type_stack:
            # Cover BOTH the standard case (chain length >= 2: x.method()) AND
            # the __call__ case (chain length 1, kind=='call': model(x) where
            # model is a typed local). The stitcher routes the length-1 case
            # to <type>.__call__ or <type>.forward.
            if len(chain) >= 2 or (len(chain) == 1 and kind == "call"):
                receiver_type = self.local_type_stack[-1].get(chain[0])
        # Module-var fallback: when chain[0] isn't a local but IS a name
        # imported from a module whose top-level vars we pre-scanned —
        # e.g. `from utility_functions import profiler` + `profiler.start()` —
        # look up the imported name in the source module's module_var_types.
        if receiver_type is None and len(chain) >= 2 and FileIndexer.module_var_types:
            # Same-module top-level var.
            t = FileIndexer.module_var_types.get((self.module_dotted, chain[0]))
            if t:
                receiver_type = t
            else:
                # Imported `from X import name`: look up in source module.
                imp = self.imports.get(chain[0])
                if imp is not None and not imp.is_module and imp.target_name is not None:
                    t = FileIndexer.module_var_types.get((imp.target_module, imp.target_name))
                    if t:
                        receiver_type = t
        self.refs.append(PyRef(
            src=src,
            target_chain=chain,
            site_file=self.file_path,
            site_line=site_line,
            site_col=site_col,
            kind=kind,
            receiver_type=receiver_type,
        ))
        if len(chain) >= 2 and chain[0] in ("ttnn", "tt_metal"):
            self.nodes[src].is_binding_caller = True

    def visit_Call(self, node: ast.Call) -> None:
        chain = attr_chain(node.func)
        src = self.scope_stack[-1]
        # Inter-procedural inference: record this call site with the types
        # of its positional / keyword arguments. The post-pass will use
        # these to infer the callee's unannotated parameter types.
        # Only meaningful when we're inside a function (have a local scope
        # to look types up in) AND the call's target is a chain we can
        # later resolve.
        if chain and self.local_type_stack:
            pos_types: list[list[str] | None] = []
            for arg in node.args:
                pos_types.append(self._infer_expr_type(arg))
            kw_types: dict[str, list[str]] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                t = self._infer_expr_type(kw.value)
                if t:
                    kw_types[kw.arg] = t
            self.call_sites.append((src, chain, pos_types, kw_types))
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
        # Flaw 9: `attach_golden_function(op, golden_function=impl)` attaches
        # `impl` as `op.golden_function` at runtime. Capture as a PyRegistration
        # under python_name="<op-chain>.golden_function" so later calls to
        # `op.golden_function(...)` resolve via the existing pyreg path.
        if chain and chain[-1] == "attach_golden_function" and len(node.args) >= 1:
            self._capture_attach_golden_function(node)
        # Recurse normally into all children (func + args + keywords). visit_Attribute
        # will dedup the func-as-attribute against the call ref above.
        self.generic_visit(node)

    def _infer_expr_type(self, expr: ast.expr) -> list[str] | None:
        """Best-effort type chain for an expression.

        Used during call-site arg type collection (inter-procedural pass).
        Only handles the easy cases:
          - bare Name → local_types lookup
          - Name with module_var_types lookup (cross-file imports)
          - Call to a known class → that class's type
          - Call to a function with annotated return → return type

        Returns None when we can't determine a useful type.
        """
        if isinstance(expr, ast.Name):
            if self.local_type_stack:
                t = self.local_type_stack[-1].get(expr.id)
                if t:
                    return t
            # Same-module module var.
            t = FileIndexer.module_var_types.get((self.module_dotted, expr.id))
            if t:
                return t
            # Imported `from X import Y`.
            imp = self.imports.get(expr.id)
            if imp is not None and not imp.is_module and imp.target_name is not None:
                t = FileIndexer.module_var_types.get((imp.target_module, imp.target_name))
                if t:
                    return t
            return None
        if isinstance(expr, ast.Call):
            fn_chain = attr_chain(expr.func)
            if fn_chain:
                # Constructor.
                if fn_chain[-1] in FileIndexer.classes_by_name:
                    return fn_chain
                # Known return type.
                rt = self._lookup_func_return_type(fn_chain)
                if rt:
                    return rt
            return None
        return None

    def _infer_assignment_types(
        self, body: list[ast.stmt], local_types: dict[str, list[str]]
    ) -> None:
        """Scan a function body for top-level assignments that we can type-infer.

        Two patterns handled (no flow-sensitivity — last assignment wins
        per name, but we walk in source order so this generally matches
        what a reader would expect):

        1. Constructor calls: `x = Foo(...)` or `x = ttnn.Tensor(...)` —
           if the call's func chain ends in a known class name, infer x
           as that class.

        2. Function calls with known return type: `x = make_thing(...)`
           where `make_thing` has a `-> SomeType` annotation, or it's
           imported from another file whose annotation we collected
           pre-pass. Look up via `func_returns`.

        We deliberately DO NOT recurse into if/else, for, with, try, etc.
        Flow-sensitive analysis is a whole project on its own; this
        catches the common pattern of `obj = Thing(...); obj.use(...)`
        at the top of a function body.
        """
        for stmt in body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            var_name = stmt.targets[0].id
            # Skip if already typed (param annotation wins).
            if var_name in local_types:
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            func_chain = attr_chain(value.func)
            if not func_chain:
                continue
            # (1) Constructor call: chain ends in a known class name.
            tail = func_chain[-1]
            cls_qns = FileIndexer.classes_by_name.get(tail)
            if cls_qns:
                local_types[var_name] = func_chain
                continue
            # (2) Function call with known return type.
            #     Resolve func_chain → (module_dotted, function_qualname),
            #     then look up its return type.
            ret_type = self._lookup_func_return_type(func_chain)
            if ret_type:
                local_types[var_name] = ret_type

    def _lookup_func_return_type(self, func_chain: list[str]) -> list[str] | None:
        """Look up the return type for a function call chain in the current scope.

        Resolves through self.imports and module_defs to find the function's
        (module_dotted, qualname), then queries FileIndexer.func_returns.
        Returns None when the chain isn't resolvable or the function has
        no return annotation.
        """
        if not func_chain:
            return None
        # Case A: bare name — local module function.
        if len(func_chain) == 1:
            name = func_chain[0]
            # Try imports first (cross-module).
            imp = self.imports.get(name)
            if imp is not None and not imp.is_module and imp.target_name is not None:
                return FileIndexer.func_returns.get((imp.target_module, imp.target_name))
            # Else try same-module qualname.
            return FileIndexer.func_returns.get((self.module_dotted, name))
        # Case B: dotted chain — first segment is an imported module or alias.
        head = func_chain[0]
        imp = self.imports.get(head)
        if imp is None:
            return None
        if imp.is_module:
            qualname = ".".join(func_chain[1:])
            return FileIndexer.func_returns.get((imp.target_module, qualname))
        # `from X import Y` then `Y.method(...)` — skip for now.
        return None

    def _capture_attach_golden_function(self, call: ast.Call) -> None:
        # Positional: attach_golden_function(operation, golden_function=...)
        # OR attach_golden_function(operation, golden_function_positional)
        op_node = call.args[0]
        op_chain = attr_chain(op_node)
        if not op_chain:
            return
        # Find the golden_function value (kwarg or 2nd positional arg).
        gf_expr = None
        for kw in call.keywords:
            if kw.arg == "golden_function":
                gf_expr = kw.value
                break
        if gf_expr is None and len(call.args) >= 2:
            gf_expr = call.args[1]
        if gf_expr is None:
            return
        impl_chain = attr_chain(gf_expr)
        impl_id: str | None = None
        if impl_chain and len(impl_chain) == 1:
            target_name = impl_chain[0]
            for nid, n in self.nodes.items():
                if n.kind in ("function", "method") and n.name == target_name:
                    impl_id = nid
                    break
        # Register under "<op-chain>.golden_function" so any caller of
        # `<op>.golden_function(...)` resolves through by_pyname_pyimpl.
        python_name = ".".join(op_chain) + ".golden_function"
        self.registrations.append(PyRegistration(
            python_name=python_name,
            impl_node_id=impl_id,
            impl_chain=impl_chain if impl_id is None else None,
            site_file=self.file_path,
            site_line=call.lineno,
            decorator_label="@attach_golden_function",
        ))

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
            target_module = _resolve_relative_module(
                self.module_dotted, node.level, node.module, is_package=self.is_package
            )
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
        if len(self.scope_stack) == 1:
            # Capture two kinds of module-level Assign:
            #   (1) `__all__ = [...]` — drives wildcard-import expansion.
            #   (2) `X = expr` — module-level variable / re-binding (e.g. `TILE_LAYOUT = Layout.TILE`,
            #       `bfloat16 = DataType.BFLOAT16`). Without these, refs to
            #       `ttnn.TILE_LAYOUT` are unresolved despite the value being
            #       a well-defined module attribute. Record as a `module_var`
            #       node so the bare-name + re-export resolver can find them.
            for tgt in node.targets:
                # __all__ assignment
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.append(elt.value)
                        self.dunder_all = names
                # Regular module-level variable assignment.
                if isinstance(tgt, ast.Name) and tgt.id != "__all__":
                    var_name = tgt.id
                    nid = f"py:module_var:{self.module_dotted}.{var_name}:{node.lineno}"
                    if nid not in self.nodes:
                        self.nodes[nid] = Node(
                            id=nid,
                            language="python",
                            kind="module_var",
                            name=var_name,
                            qualified_name=f"{self.module_dotted}.{var_name}",
                            file=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                        )
                    # Last-binding-wins for resolver lookups.
                    self.module_defs[var_name] = nid
                # Tuple-unpacking `a, b = ...` is uncommon at module scope for
                # exported constants; ignore for now.
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Annotated module-level assignment: `X: T = ...` or `X: T` (no value).

        Treated the same as a plain `X = ...` for module-var node recording.
        """
        if len(self.scope_stack) == 1 and isinstance(node.target, ast.Name):
            var_name = node.target.id
            if var_name != "__all__":
                nid = f"py:module_var:{self.module_dotted}.{var_name}:{node.lineno}"
                if nid not in self.nodes:
                    self.nodes[nid] = Node(
                        id=nid,
                        language="python",
                        kind="module_var",
                        name=var_name,
                        qualified_name=f"{self.module_dotted}.{var_name}",
                        file=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                    )
                self.module_defs[var_name] = nid
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
                n.kind in ("function", "class", "module_var")
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
            # Skip transient / generated dirs only. tests/ is now in scope.
            sp = str(p)
            if "/__pycache__/" in sp or "/.tox/" in sp or "/build/" in sp:
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
    ap.add_argument(
        "--pytype-stubs",
        default=None,
        help="Root of pytype-generated stubs (the directory containing pyi/). "
             "Used to fill in receiver_type for unannotated parameters.",
    )
    args = ap.parse_args()

    files = _gather_files(args)
    if not files:
        print("error: at least one --file or --dir is required", file=sys.stderr)
        sys.exit(2)
    roots = [Path(r) for r in args.module_root] or [Path(".")]

    if args.pytype_stubs:
        stubs_path = Path(args.pytype_stubs)
        FileIndexer.inferred_types = load_pytype_stubs(stubs_path)
        print(f"[py_index] loaded {len(FileIndexer.inferred_types)} pytype-inferred function signatures",
              file=sys.stderr)

    # Type-propagator pre-pass: scan all files for class definitions and
    # function return annotations. Cheap (one ast.parse per file) and
    # populates the globals that `_infer_assignment_types` consults during
    # the main walk.
    classes_by_name: dict[str, list[str]] = {}
    func_returns: dict[tuple[str, str], list[str]] = {}
    module_var_types: dict[tuple[str, str], list[str]] = {}
    for fp in files:
        cls_records, fn_records, mv_records = quick_scan_for_types(fp, roots)
        for module, qualname, name in cls_records:
            classes_by_name.setdefault(name, []).append(f"{module}.{qualname}")
        for module, qualname, ret_type in fn_records:
            func_returns[(module, qualname)] = ret_type
        for module, var_name, type_chain in mv_records:
            # Heuristic: only keep if the type_chain ends in a known class —
            # otherwise we'd record `parser = argparse.ArgumentParser()`-style
            # entries whose type is external/non-resolvable.
            if type_chain[-1] in classes_by_name or len(type_chain) >= 2:
                module_var_types[(module, var_name)] = type_chain
    # Second pass: filter module_var_types to only keep known classes.
    module_var_types = {
        k: v for k, v in module_var_types.items()
        if v[-1] in classes_by_name
    }
    FileIndexer.classes_by_name = classes_by_name
    FileIndexer.func_returns = func_returns
    FileIndexer.module_var_types = module_var_types
    print(
        f"[py_index] type-propagator: {len(classes_by_name)} class names, "
        f"{len(func_returns)} func return-types, "
        f"{len(module_var_types)} module-var types pre-collected",
        file=sys.stderr,
    )

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
            if n.kind in ("function", "class", "module_var"):
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

    # Pass 6: shallow inter-procedural parameter inference.
    # Look at every call site we collected, resolve the target to a
    # function node id, and aggregate caller arg types per parameter
    # position. When a parameter has consistent caller-arg types
    # across all (visible) call sites, infer that as the param's type.
    # Then update existing refs whose chain[0] is the inferred-typed
    # parameter to attach receiver_type for the stitcher.
    fid_to_fi: dict[str, "FileIndexer"] = {}
    for fi in indexers:
        for fid in fi.func_params:
            fid_to_fi[fid] = fi
    # Aggregate caller-arg types per (func_id, param_name).
    type_collector: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for fi in indexers:
        for caller_id, target_chain, pos_types, kw_types in fi.call_sites:
            target_id = _resolve_chain_to_node(fi, target_chain)
            if not target_id:
                continue
            target_fi = fid_to_fi.get(target_id)
            if target_fi is None:
                continue
            params = target_fi.func_params.get(target_id, [])
            if not params:
                continue
            # Skip the implicit 'self' / 'cls' parameter at position 0
            # for methods — caller positions start at the next param.
            if params and params[0] in ("self", "cls"):
                effective_params = params[1:]
            else:
                effective_params = params
            for i, t in enumerate(pos_types):
                if t and i < len(effective_params):
                    type_collector[(target_id, effective_params[i])].add(tuple(t))
            for kname, t in kw_types.items():
                if t and kname in params:
                    type_collector[(target_id, kname)].add(tuple(t))
    # Keep only parameters where ALL call sites passed the same type.
    inferred_params_global: dict[tuple[str, str], list[str]] = {}
    for (fid, pname), types in type_collector.items():
        if len(types) == 1:
            inferred_params_global[(fid, pname)] = list(next(iter(types)))
    FileIndexer.inferred_param_types = inferred_params_global

    # Apply to existing refs: for refs whose chain[0] is an inferred-typed
    # parameter of their enclosing function, set receiver_type. Don't
    # overwrite already-set receiver_types.
    interproc_refs_updated = 0
    for fi in indexers:
        for r in fi.refs:
            if r.receiver_type:
                continue
            if not r.target_chain:
                continue
            recv = r.target_chain[0]
            params = fi.func_params.get(r.src)
            if not params or recv not in params:
                continue
            inferred = inferred_params_global.get((r.src, recv))
            if inferred:
                r.receiver_type = inferred
                interproc_refs_updated += 1

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
        f"{b4_edges} via Jedi type-resolver, "
        f"{interproc_refs_updated} refs receiver-type-tagged via inter-procedural inference"
        f" [{len(inferred_params_global)} params inferred]), "
        f"{len(out['refs'])} refs, {len(out['registrations'])} registrations, "
        f"{len(out['diagnostics'])} diags -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

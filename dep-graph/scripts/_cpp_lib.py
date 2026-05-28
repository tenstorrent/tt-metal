"""Shared C++ indexing utilities.

Imported by `cpp_index.py` (legacy monolithic CLI), `cpp_index_worker.py`
(one-TU shard writer), and `cpp_index_merger.py` (shard folder).
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

from clang import cindex

# Pin to the system libclang-20 so the parser matches the compiler used in the build.
SYSTEM_LIBCLANG = "/usr/lib/x86_64-linux-gnu/libclang-20.so.1"
if Path(SYSTEM_LIBCLANG).exists():
    cindex.Config.set_library_file(SYSTEM_LIBCLANG)


# ─── libclang bindings patch (Tier 1 A1) ───────────────────────────────────
#
# libclang 18.1.1 (the latest on PyPI as of writing) doesn't know about cursor
# kinds emitted by clang-20. The default `BaseEnumeration.from_id` raises
# `ValueError("Unknown template argument kind N")` on first access to such a
# cursor, which poisoned whole TUs before we wrapped accesses in `_safe_kind`.
#
# The patch below intercepts `from_id` at module import: unknown IDs get a
# sentinel registered on the fly and named `_UNKNOWN_<id>`. After registration,
# subsequent `from_id` calls follow the normal fast path. The sentinel still
# isn't in any of our FUNCTION_KINDS / kind comparison sets, so the walker
# naturally treats those cursors as opaque (descends through them but does
# not record anything).
def _wrap_missing_libclang_apis() -> None:
    """Register the libclang C functions cindex.py 18.1.1 didn't bind.

    Two functions matter for us:
      - `clang_getOverriddenCursors(cursor, out_array, out_count)` — given a
        CXX_METHOD cursor, returns the parent methods it overrides. Used by
        B2 to emit `kind=overrides` edges.
      - `clang_getSpecializedCursorTemplate(cursor)` — already in cindex's
        function list, but with no Python wrapper method. Returns the primary
        template a specialization specializes. Used by B3 to emit
        `kind=instantiates` edges.
    """
    import ctypes
    from ctypes import POINTER, c_uint, byref

    lib = cindex.conf.lib

    # clang_getOverriddenCursors / clang_disposeOverriddenCursors. Not in
    # cindex's functionList in 18.1.1; register manually.
    try:
        lib.clang_getOverriddenCursors.argtypes = [
            cindex.Cursor, POINTER(POINTER(cindex.Cursor)), POINTER(c_uint),
        ]
        lib.clang_getOverriddenCursors.restype = None
        lib.clang_disposeOverriddenCursors.argtypes = [POINTER(cindex.Cursor)]
        lib.clang_disposeOverriddenCursors.restype = None
    except AttributeError:
        # libclang.so doesn't have these symbols — skip silently.
        pass


def get_overridden_cursors(cursor: cindex.Cursor) -> list[cindex.Cursor]:
    """Return the parent methods that `cursor` overrides (empty for non-virtual
    or top-of-hierarchy methods).

    The returned list contains byte-copies of the underlying CXCursor structs
    so they survive `clang_disposeOverriddenCursors`. The TU reference is
    propagated so the AST is kept alive while these cursors are in use.
    """
    import ctypes
    from ctypes import POINTER, c_uint, byref
    lib = cindex.conf.lib
    fn = getattr(lib, "clang_getOverriddenCursors", None)
    dispose = getattr(lib, "clang_disposeOverriddenCursors", None)
    if fn is None or dispose is None:
        return []
    out = POINTER(cindex.Cursor)()
    num = c_uint(0)
    try:
        fn(cursor, byref(out), byref(num))
    except Exception:
        return []
    tu = getattr(cursor, "translation_unit", None) or getattr(cursor, "_tu", None)
    results: list[cindex.Cursor] = []
    for i in range(num.value):
        src = out[i]
        # Byte-copy the Cursor struct so it survives the dispose call.
        copy = cindex.Cursor()
        ctypes.memmove(ctypes.addressof(copy), ctypes.addressof(src), ctypes.sizeof(cindex.Cursor))
        if tu is not None:
            copy._tu = tu
        results.append(copy)
    try:
        dispose(out)
    except Exception:
        pass
    return results


def get_specialized_template(cursor: cindex.Cursor) -> cindex.Cursor | None:
    """Return the primary template a specialization specializes, or None.

    `clang_getSpecializedCursorTemplate` is already registered in cindex.py's
    functionList with `Cursor.from_cursor_result` as its errcheck (which
    returns None for null cursors). So this is a simple call.
    """
    try:
        result = cindex.conf.lib.clang_getSpecializedCursorTemplate(cursor)
    except Exception:
        return None
    # from_cursor_result already returns None for invalid; defend against errors.
    return result


def _patch_libclang_unknown_kinds() -> None:
    base = cindex.BaseEnumeration
    orig = base.from_id.__func__  # unwrap classmethod

    def patched_from_id(cls, id):
        try:
            return orig(cls, id)
        except ValueError:
            # Lazy registration: instantiating cls(id) appends the sentinel
            # to cls._kinds (see BaseEnumeration.__init__). Give it a name so
            # downstream `.name` accesses don't blow up.
            sentinel = cls(id)
            setattr(cls, f"_UNKNOWN_{id}", sentinel)
            cls._name_map = None  # invalidate the cached name map
            return sentinel

    base.from_id = classmethod(patched_from_id)


_patch_libclang_unknown_kinds()
_wrap_missing_libclang_apis()


# ─── argv preprocessing ────────────────────────────────────────────────────

COMPILER_BASENAMES = (
    "clang++", "clang++-20", "clang", "clang-20", "gcc", "g++", "g++-12", "cc", "c++",
)


def strip_compiler(argv: list[str]) -> list[str]:
    if argv and os.path.basename(argv[0]) in COMPILER_BASENAMES:
        return argv[1:]
    return argv


def prune_args(argv: list[str], source_path: str) -> list[str]:
    """Strip args libclang shouldn't see or can't consume.

    See docstring on probe_parse.py for the full list. Key items:
      - PCH includes (libclang 18 can't read clang-20 PCHs)
      - -Werror (we don't want spurious style-warning escalations)
      - -o <file>, -c, the source file itself
    """
    src_norm = os.path.normpath(source_path)
    pruned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if os.path.normpath(a) == src_norm:
            i += 1
            continue
        if a == "-o":
            i += 2
            continue
        if a.startswith("-o") and a != "-o":
            i += 1
            continue
        if a == "-c":
            i += 1
            continue
        if a in ("-Werror", "-pedantic-errors", "-Winvalid-pch"):
            i += 1
            continue
        if a == "-Xclang" and i + 1 < len(argv) and argv[i + 1] == "-include-pch":
            i += 4
            continue
        if (
            a == "-Xclang"
            and i + 3 < len(argv)
            and argv[i + 1] == "-include"
            and argv[i + 2] == "-Xclang"
            and argv[i + 3].endswith((".hxx", ".hpp", ".h"))
            and "pch" in argv[i + 3].lower()
        ):
            i += 4
            continue
        pruned.append(a)
        i += 1
    return pruned


# ─── compile_commands.json ─────────────────────────────────────────────────

@dataclass
class TUEntry:
    file: str
    directory: str
    argv: list[str]


def load_db(db_path: Path) -> dict[str, TUEntry]:
    raw = json.loads(db_path.read_text())
    out: dict[str, TUEntry] = {}
    for entry in raw:
        argv = entry.get("arguments")
        if argv is None:
            argv = shlex.split(entry["command"])
        out[os.path.normpath(entry["file"])] = TUEntry(
            file=os.path.normpath(entry["file"]),
            directory=entry["directory"],
            argv=list(argv),
        )
    return out


# ─── scope filter ──────────────────────────────────────────────────────────

# Per §1 of opus-instructions.md. `tt_metal/impl/` and `tt_metal/jit_build/`
# are included automatically because they live under the `/workspace/tt_metal/`
# prefix and are not explicitly excluded.
DEFAULT_IN_SCOPE_PREFIXES = (
    "/workspace/ttnn/cpp/",
    "/workspace/ttnn/api/",   # public ttnn headers (e.g. ttnn::Tensor alias)
    "/workspace/ttnn/core/",  # ttnn impl + bindings (distributed_nanobind, graph_nanobind, tensor.cpp, …)
    "/workspace/ttnn/examples/",  # ttnn host-program labs (add.cpp, lab_eltwise_binary.cpp, …)
    "/workspace/tt_metal/",
    "/workspace/tt_stl/",
    "/workspace/tests/",      # tests now in scope (per user direction)
)
DEFAULT_OUT_OF_SCOPE_PREFIXES = (
    "/workspace/.cpmcache/",
    "/workspace/build_Release/",
    "/workspace/build/",
    "/workspace/tt_metal/third_party/",
    "/workspace/tt_metal/hw/",
    "/workspace/runtime/sfpi/",
    "/workspace/.github/",
    # Note: /workspace/tests/ used to be here but is now IN scope.
)


def in_scope(path: str | None) -> bool:
    if not path:
        return False
    p = os.path.normpath(path)
    if any(p.startswith(o) for o in DEFAULT_OUT_OF_SCOPE_PREFIXES):
        return False
    return any(p.startswith(s) for s in DEFAULT_IN_SCOPE_PREFIXES)


# ─── cache hash ────────────────────────────────────────────────────────────


def canonical_argv(argv: list[str]) -> str:
    """A stable string representation of an argv suitable for hashing."""
    return "\x1f".join(argv)


def tu_cache_key(tu_path: str, argv: list[str]) -> str:
    """Short hash uniquely identifying a TU + its compile flags.

    Mtime is *not* part of the key — it's checked separately when the shard
    manifest is consulted. This means the cache directory stays small as a TU
    is edited repeatedly (one hash, one shard dir).
    """
    blob = f"{os.path.normpath(tu_path)}\n{canonical_argv(argv)}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ─── AST helpers ───────────────────────────────────────────────────────────

FUNCTION_KINDS = {
    cindex.CursorKind.FUNCTION_DECL,
    cindex.CursorKind.CXX_METHOD,
    cindex.CursorKind.CONSTRUCTOR,
    cindex.CursorKind.DESTRUCTOR,
    cindex.CursorKind.CONVERSION_FUNCTION,
    cindex.CursorKind.FUNCTION_TEMPLATE,
}

# Kinds that can be the *target* of a nanobind binding. Functions are the
# common case for `.def`/`.def_static`/`.def_prop_*`; fields are the target
# for `.def_ro` / `.def_rw` (read-only / read-write attribute bindings).
BINDABLE_KINDS = FUNCTION_KINDS | {
    cindex.CursorKind.FIELD_DECL,
    cindex.CursorKind.VAR_DECL,            # static fields appear as VAR_DECL
    cindex.CursorKind.ENUM_CONSTANT_DECL,  # nb::enum_<E>(...).value("X", E::X) target
}

# Class-like declarations. B2 records nodes for these and follows their base
# specifiers to emit `inherits` edges. ENUM_DECL is included so `nb::enum_<E>`
# class bindings have a real node to point at.
CLASS_KINDS = {
    cindex.CursorKind.CLASS_DECL,
    cindex.CursorKind.STRUCT_DECL,
    cindex.CursorKind.CLASS_TEMPLATE,
    cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    cindex.CursorKind.ENUM_DECL,
}


def qualified_name(c: cindex.Cursor) -> str:
    parts: list[str] = []
    cur = c
    while cur is not None and cur.kind != cindex.CursorKind.TRANSLATION_UNIT:
        spelling = cur.spelling
        if spelling:
            parts.append(spelling)
        cur = cur.semantic_parent
    return "::".join(reversed(parts))


def signature_str(c: cindex.Cursor) -> str:
    try:
        ret = c.result_type.spelling if c.result_type and c.result_type.spelling else "void"
    except Exception:
        ret = "?"
    try:
        params = ", ".join(p.type.spelling for p in c.get_arguments())
    except Exception:
        params = "?"
    return f"{ret} {qualified_name(c)}({params})"


def node_id(c: cindex.Cursor) -> str:
    canonical = c.canonical
    usr = canonical.get_usr() or c.get_usr()
    return f"cpp:{usr}" if usr else f"cpp:{qualified_name(c)}"


def cursor_loc(c: cindex.Cursor) -> tuple[str | None, int, int]:
    if c.extent and c.extent.start.file:
        return (c.extent.start.file.name, c.extent.start.line, c.extent.end.line)
    if c.location and c.location.file:
        return (c.location.file.name, c.location.line, c.location.line)
    return (None, 0, 0)


# ─── output records ────────────────────────────────────────────────────────


@dataclass
class Node:
    id: str
    language: str
    kind: str
    name: str
    qualified_name: str
    file: str
    line_start: int
    line_end: int
    signature: str
    is_definition: bool
    is_binding_target: bool = False
    is_template: bool = False
    discovered_in: list[str] = field(default_factory=list)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    site_file: str
    site_line: int
    via: str | None = None
    crosses_language: bool = False
    # B2: dispatch attribute. Set to "virtual" on call edges whose callee is a
    # virtual method (the call may dispatch to any override at runtime), and
    # left None for static / direct calls. Used by query.py to fan out
    # virtual-call edges when computing blast radius.
    via_dispatch: str | None = None


@dataclass
class Binding:
    python_name: str
    cpp_node_id: str
    cpp_qualified_name: str
    site_file: str
    site_line: int
    helper: str


# ─── indexer ───────────────────────────────────────────────────────────────


# D1: host-side functions that launch a Tensix/RISC-V kernel by passing a
# string-literal path to a `.cpp` / `.hpp` file. The kernel itself is compiled
# by the SFPI/RISC-V toolchain at JIT time and is NOT in our compile_commands;
# we represent each kernel as an opaque `kind=kernel_file` node identified by
# the path, and emit `kind=launches` edges from the calling host function.
# Method names invoked on a `class_<T>` instance — chained off the constructor.
# They share the constructor's return type but are method bindings, not class
# bindings. Excluded from the class-binding detection in `_handle_call`.
_CLASS_DEF_METHODS = {
    "def", "def_ro", "def_rw", "def_prop_ro", "def_prop_rw",
    "def_static", "def_submodule", "def_readonly", "def_readwrite",
    "value",  # nb::enum_<E>(...).value("NAME", E::NAME)
}


KERNEL_LAUNCHERS = {
    "CreateKernel",
    "CreateComputeKernel",
    "CreateDataMovementKernel",
    "CreateReadKernel",
    "CreateWriteKernel",
    "CreateKernelFromPrecompiled",
    "CreateEthernetKernel",   # speculative; harmless if absent
}

BINDING_HELPERS = {
    "bind_function",
    "bind_binary_operation",
    "bind_binary_inplace_operation",
    "bind_binary_gcd_lcm_operation",
    "bind_binary_unary_max_operation",
    "bind_binary_unary_operation",
    "bind_binary_with_float_param",
    "bind_binary_composite",
    "bind_binary_composite_with_rtol_atol",
    "bind_binary_composite_overload",
    "bind_binary_operation_with_fast_approx",
    "bind_binary_overload_operation",
    # nanobind / pybind binding methods (all share the same string-literal-first,
    # callable-arg shape, so `_extract_binding` handles them identically).
    "def",            # method/free function    cls.def("name", &T::method)
    "def_ro",         # read-only field         cls.def_ro("field", &T::field)
    "def_rw",         # read-write field        cls.def_rw("field", &T::field)
    "def_prop_ro",    # read-only property      cls.def_prop_ro("name", &T::getter)
    "def_prop_rw",    # read-write property     cls.def_prop_rw("name", &T::getter, &T::setter)
    "def_static",     # static method           cls.def_static("name", &T::static_fn)
    "value",          # enum value              nb::enum_<E>(...).value("NAME", E::NAME)
}


class Indexer:
    def __init__(self, db: dict[str, TUEntry]) -> None:
        self.db = db
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.bindings: list[Binding] = []
        self.diagnostics: list[dict] = []
        self.included_files: list[str] = []
        self.index = cindex.Index.create()

    def parse_tu(self, tu_path: str) -> cindex.TranslationUnit | None:
        key = os.path.normpath(tu_path)
        if key not in self.db:
            self.diagnostics.append({"tu": tu_path, "error": "not in compile_commands.json"})
            return None
        entry = self.db[key]
        argv = strip_compiler(entry.argv)
        argv = prune_args(argv, entry.file)
        saved_cwd = os.getcwd()
        os.chdir(entry.directory)
        try:
            tu = self.index.parse(
                entry.file,
                args=argv,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except cindex.TranslationUnitLoadError as e:
            self.diagnostics.append({"tu": tu_path, "error": f"parse failed: {e}"})
            return None
        finally:
            os.chdir(saved_cwd)
        for d in tu.diagnostics:
            if d.severity >= cindex.Diagnostic.Error:
                self.diagnostics.append({
                    "tu": tu_path,
                    "severity": d.severity,
                    "message": d.spelling,
                    "location": str(d.location),
                })
        # Capture include list for cache freshness checks.
        for inc in tu.get_includes():
            if inc.source and getattr(inc.source, "name", None):
                self.included_files.append(os.path.normpath(inc.source.name))
        return tu

    def index_tu(self, tu_path: str) -> None:
        tu = self.parse_tu(tu_path)
        if tu is None:
            return
        self._walk(tu.cursor, current_function=None, tu_path=tu_path)

    def _safe_kind(self, cursor: cindex.Cursor) -> cindex.CursorKind | None:
        """Return cursor.kind, or None if libclang returned a kind value the
        Python bindings (18.1.1) don't recognize. Newer clang (20) emits cursor
        kinds the binding's enum table doesn't cover; raising would abort the
        whole TU. We treat such cursors as opaque and skip them.
        """
        try:
            return cursor.kind
        except ValueError as e:
            self.diagnostics.append({
                "tu": getattr(cursor.location.file, "name", "?") if cursor.location and cursor.location.file else "?",
                "severity": "warning",
                "message": f"unknown cursor kind: {e}",
            })
            return None

    def _iter_children(self, cursor: cindex.Cursor):
        try:
            yield from cursor.get_children()
        except ValueError as e:
            self.diagnostics.append({"severity": "warning", "message": f"get_children() failed: {e}"})

    def _walk(
        self,
        cursor: cindex.Cursor,
        current_function: cindex.Cursor | None,
        tu_path: str,
    ) -> None:
        kind = self._safe_kind(cursor)
        if kind is None:
            # Walk children defensively; we lose info from this cursor but the
            # subtree may still contain analyzable nodes.
            for child in self._iter_children(cursor):
                self._walk(child, current_function, tu_path)
            return

        if kind in FUNCTION_KINDS:
            f, _, _ = cursor_loc(cursor)
            if in_scope(f):
                self._record_function(cursor, tu_path)
            for child in self._iter_children(cursor):
                self._walk(child, current_function=cursor if cursor.is_definition() else current_function, tu_path=tu_path)
            return

        if kind in CLASS_KINDS:
            f, _, _ = cursor_loc(cursor)
            if in_scope(f):
                self._record_class(cursor, tu_path)
            # Class bodies can contain methods (FUNCTION_KINDS), nested classes,
            # and call expressions in initializers. Recurse normally.
            # For ENUM_DECL, also record each ENUM_CONSTANT_DECL child so that
            # `nb::enum_<E>(...).value("X", E::X)` bindings have a real node to
            # point at on the C++ side.
            if kind == cindex.CursorKind.ENUM_DECL and in_scope(f):
                for child in self._iter_children(cursor):
                    try:
                        ck = child.kind
                    except ValueError:
                        continue
                    if ck == cindex.CursorKind.ENUM_CONSTANT_DECL:
                        self._record_enum_constant(child, tu_path)
            for child in self._iter_children(cursor):
                self._walk(child, current_function, tu_path)
            return

        if kind == cindex.CursorKind.CALL_EXPR:
            self._handle_call(cursor, current_function, tu_path)

        for child in self._iter_children(cursor):
            self._walk(child, current_function, tu_path)

    def _record_function(self, c: cindex.Cursor, tu_path: str) -> None:
        try:
            f, l0, l1 = cursor_loc(c)
        except Exception:
            return
        nid = node_id(c)
        node = self.nodes.get(nid)
        if node is None:
            node = Node(
                id=nid,
                language="cpp",
                kind=str(c.kind).removeprefix("CursorKind."),
                name=c.spelling,
                qualified_name=qualified_name(c),
                file=f or "",
                line_start=l0,
                line_end=l1,
                signature=signature_str(c),
                is_definition=c.is_definition(),
                is_template=c.kind == cindex.CursorKind.FUNCTION_TEMPLATE,
                discovered_in=[tu_path],
            )
            self.nodes[nid] = node
        else:
            if c.is_definition() and not node.is_definition:
                node.is_definition = True
                node.file = f or node.file
                node.line_start = l0
                node.line_end = l1
            if tu_path not in node.discovered_in:
                node.discovered_in.append(tu_path)

        # B2 overrides: virtual methods get explicit `kind=overrides` edges
        # to each parent method they override. ctypes-wrapped at module load.
        if c.kind == cindex.CursorKind.CXX_METHOD:
            for parent in get_overridden_cursors(c):
                try:
                    parent_id = node_id(parent)
                except Exception:
                    continue
                if parent_id == nid:
                    continue
                self.edges.append(Edge(
                    src=nid, dst=parent_id, kind="overrides",
                    site_file=f or "", site_line=l0,
                ))

        # B3 instantiates: a function-template specialization links to its
        # primary template. Most useful for the `bind_function<"name">`
        # instantiations (one per ttnn op) and the `*DeviceOperation`
        # struct-template family.
        try:
            primary = get_specialized_template(c)
        except Exception:
            primary = None
        if primary is not None:
            try:
                primary_id = node_id(primary)
            except Exception:
                primary_id = None
            if primary_id and primary_id != nid:
                self.edges.append(Edge(
                    src=nid, dst=primary_id, kind="instantiates",
                    site_file=f or "", site_line=l0,
                ))

    def _record_enum_constant(self, c: cindex.Cursor, tu_path: str) -> None:
        """Record an ENUM_CONSTANT_DECL as a node so `nb::enum_<>.value(...)`
        bindings have a valid cross-language edge target.
        """
        try:
            f, l0, l1 = cursor_loc(c)
        except Exception:
            return
        nid = node_id(c)
        if nid not in self.nodes:
            self.nodes[nid] = Node(
                id=nid,
                language="cpp",
                kind="ENUM_CONSTANT_DECL",
                name=c.spelling,
                qualified_name=qualified_name(c),
                file=f or "",
                line_start=l0,
                line_end=l1,
                signature=qualified_name(c),
                is_definition=True,
                is_template=False,
                discovered_in=[tu_path],
            )
        elif tu_path not in self.nodes[nid].discovered_in:
            self.nodes[nid].discovered_in.append(tu_path)

    def _record_class(self, c: cindex.Cursor, tu_path: str) -> None:
        """B2: record a class/struct/class-template node, and emit `inherits`
        edges to its base classes via CXX_BASE_SPECIFIER children.
        """
        try:
            f, l0, l1 = cursor_loc(c)
        except Exception:
            return
        nid = node_id(c)
        node_kind = "enum" if c.kind == cindex.CursorKind.ENUM_DECL else "class"
        if nid not in self.nodes:
            self.nodes[nid] = Node(
                id=nid,
                language="cpp",
                kind=node_kind,
                name=c.spelling,
                qualified_name=qualified_name(c),
                file=f or "",
                line_start=l0,
                line_end=l1,
                signature=qualified_name(c),
                is_definition=c.is_definition(),
                is_template=c.kind == cindex.CursorKind.CLASS_TEMPLATE,
                discovered_in=[tu_path],
            )
        elif tu_path not in self.nodes[nid].discovered_in:
            self.nodes[nid].discovered_in.append(tu_path)

        # ENUM_DECL doesn't have base classes / template instantiations.
        if c.kind == cindex.CursorKind.ENUM_DECL:
            return

        # B3 instantiates for class-template specializations.
        try:
            primary = get_specialized_template(c)
        except Exception:
            primary = None
        if primary is not None:
            try:
                primary_id = node_id(primary)
            except Exception:
                primary_id = None
            if primary_id and primary_id != nid:
                self.edges.append(Edge(
                    src=nid, dst=primary_id, kind="instantiates",
                    site_file=f or "", site_line=l0,
                ))

        for child in self._iter_children(c):
            try:
                ck = child.kind
            except ValueError:
                continue
            if ck != cindex.CursorKind.CXX_BASE_SPECIFIER:
                continue
            # `child.referenced` is the base class declaration cursor.
            base = child.referenced
            if base is None:
                continue
            try:
                base_kind = base.kind
            except ValueError:
                continue
            if base_kind not in CLASS_KINDS:
                continue
            base_id = node_id(base)
            if base_id == nid:
                continue
            site_line = child.location.line if child.location else l0
            self.edges.append(Edge(
                src=nid,
                dst=base_id,
                kind="inherits",
                site_file=f or "",
                site_line=site_line,
            ))

    def _handle_call(
        self,
        call: cindex.Cursor,
        current_function: cindex.Cursor | None,
        tu_path: str,
    ) -> None:
        callee = call.referenced or (call.get_definition() if hasattr(call, "get_definition") else None)
        callee_name = call.spelling or (callee.spelling if callee else "")
        if callee_name in BINDING_HELPERS:
            self._extract_binding(call, callee_name, tu_path)
        if callee_name in KERNEL_LAUNCHERS:
            self._extract_kernel_launch(call, current_function, callee_name, tu_path)
        # D1b: newer KernelDescriptor pattern. `desc.kernel_source = "path.cpp"`
        # is an `operator=` call (because kernel_source is std::string_view /
        # std::string with overloaded =). Detect by callee name + LHS member name.
        if callee_name == "operator=":
            self._maybe_extract_kernel_source(call, current_function, tu_path)
        # Tier 1 (3): class bindings via `nb::class_<T>(m, "Name")` AND via
        # tt-metal's `tt_serializable_class<T>(m, "Name", ...)` helper. Both
        # return a `class_<T>` (or `nb::class_<T>`) — detect structurally.
        # Excludes chained `.def_*` calls on the resulting class_ instance,
        # which share the same return type but are method bindings already
        # handled by `_extract_binding`.
        if (
            callee_name
            and callee_name not in _CLASS_DEF_METHODS
            and self._call_returns_class_(call)
        ):
            self._extract_class_binding(call, tu_path)
        if current_function is None or callee is None:
            return
        if callee.kind not in FUNCTION_KINDS:
            return
        # Template instantiations: libclang resolves a call like `ttnn::full(...)`
        # to a FUNCTION_DECL whose USR includes the concrete template arguments
        # (e.g. `c:@N@ttnn@F@full<#f>#...`), but `_record_function` only records
        # the primary template node (USR has no concrete args). Redirect the
        # callee to the primary template so the edge dst matches an indexed node.
        primary = get_specialized_template(callee)
        if primary is not None:
            try:
                if primary.kind in FUNCTION_KINDS:
                    callee = primary
            except ValueError:
                pass
        src_file, _, _ = cursor_loc(current_function)
        callee_file, _, _ = cursor_loc(callee)
        if not (in_scope(src_file) and in_scope(callee_file)):
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        # B2: tag virtual method calls so consumers can fan out through
        # `overrides` edges (every method that overrides this one is a
        # potential runtime target).
        via_dispatch: str | None = None
        try:
            if callee.kind == cindex.CursorKind.CXX_METHOD and callee.is_virtual_method():
                via_dispatch = "virtual"
        except (ValueError, AttributeError):
            pass
        self.edges.append(Edge(
            src=node_id(current_function),
            dst=node_id(callee),
            kind="calls",
            site_file=site_file,
            site_line=site_line,
            via_dispatch=via_dispatch,
        ))

    # ─── D1: kernel-launch extraction ──────────────────────────────────

    _KERNEL_PATH_SUFFIXES = (".cpp", ".hpp", ".h", ".cc", ".cxx")

    # Fields on the descriptor types that bind a kernel source path. Add to
    # this set if new descriptor variants surface. Survey result:
    #   - `kernel_source` (std::string)         — KernelDescriptor (Program / ProgramDescriptor flow)
    #   - `kernel_path`   (std::string)         — used by fabric / ccl edm kernels
    _KERNEL_PATH_FIELDS = {"kernel_source", "kernel_path"}

    @classmethod
    def _resolve_string_value(cls, cursor: cindex.Cursor, depth: int = 0) -> str | None:
        """Resolve a cursor to its effective string-literal value, if any.

        Single-value version. For multi-value resolution (covering
        variables assigned across multiple branches), use
        `_resolve_string_values_all`.
        """
        vals = cls._resolve_string_values_all(cursor, scope=None, depth=depth)
        return vals[0] if vals else None

    @classmethod
    def _resolve_string_values_all(
        cls,
        cursor: cindex.Cursor,
        scope: cindex.Cursor | None,
        depth: int = 0,
        _visited_usrs: set | None = None,
    ) -> list[str]:
        """Resolve a cursor to ALL possible string-literal values it can take.

        Handles:
          - direct STRING_LITERAL
          - UNEXPOSED_EXPR / cast wrappers — descend
          - DECL_REF_EXPR → VAR_DECL with string-literal initializer
          - DECL_REF_EXPR → VAR_DECL with NO initializer but assigned
            via `var = "literal"` later in the enclosing function (Flaw 13:
            kernel_source through variable assignment). Requires `scope`
            (the enclosing FUNCTION_DECL cursor) to find those assignments.
          - Ternary / conditional operators — both branches harvested.
        """
        if depth > 12:
            return []
        if _visited_usrs is None:
            _visited_usrs = set()
        try:
            kind = cursor.kind
        except ValueError:
            return []
        if kind == cindex.CursorKind.STRING_LITERAL:
            sp = cursor.spelling
            if sp and sp.startswith('"') and sp.endswith('"'):
                return [sp[1:-1]]
            return [sp] if sp else []
        if kind == cindex.CursorKind.DECL_REF_EXPR:
            ref = cursor.referenced
            if ref is None:
                return []
            try:
                rk = ref.kind
            except ValueError:
                rk = None
            if rk != cindex.CursorKind.VAR_DECL:
                return []
            usr = ref.get_usr()
            if usr in _visited_usrs:
                return []
            _visited_usrs = _visited_usrs | {usr}
            # First: try the initializer (constexpr / const cases).
            results: list[str] = []
            try:
                children = list(ref.get_children())
            except ValueError:
                children = []
            for ch in children:
                results.extend(cls._resolve_string_values_all(ch, scope, depth + 1, _visited_usrs))
            # Fallback: scan the enclosing function's body for assignments
            # to this variable (`var = expr`). Catches the multi-branch
            # pattern where `compute_kernel` is declared empty and then
            # set in different control-flow paths.
            if scope is not None:
                for rhs in cls._find_assignments_to(ref, scope, usr):
                    results.extend(cls._resolve_string_values_all(rhs, scope, depth + 1, _visited_usrs))
            # Dedup while preserving order.
            seen = set()
            uniq: list[str] = []
            for v in results:
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            return uniq
        if kind == cindex.CursorKind.LAMBDA_EXPR:
            return []
        # Default: walk children, collect from any child that has values.
        try:
            children = list(cursor.get_children())
        except ValueError:
            return []
        out: list[str] = []
        for ch in children:
            out.extend(cls._resolve_string_values_all(ch, scope, depth + 1, _visited_usrs))
        return out

    @staticmethod
    def _find_assignments_to(
        var_decl: cindex.Cursor,
        scope: cindex.Cursor,
        var_usr: str,
    ) -> list[cindex.Cursor]:
        """Find all `var = expr` operator= calls in `scope`'s subtree where
        the LHS is `var_decl`. Returns the RHS cursors.
        """
        if not var_usr:
            return []
        results: list[cindex.Cursor] = []

        def walk(c: cindex.Cursor) -> None:
            try:
                ck = c.kind
            except ValueError:
                return
            if ck == cindex.CursorKind.CALL_EXPR and c.spelling == "operator=":
                try:
                    args = list(c.get_arguments())
                except Exception:
                    args = []
                if len(args) >= 2:
                    lhs = args[0]
                    try:
                        if lhs.kind == cindex.CursorKind.DECL_REF_EXPR:
                            ref = lhs.referenced
                            if ref is not None and ref.get_usr() == var_usr:
                                results.append(args[1])
                    except (ValueError, AttributeError):
                        pass
            try:
                for ch in c.get_children():
                    walk(ch)
            except ValueError:
                pass

        try:
            walk(scope)
        except Exception:
            pass
        return results

    def _maybe_extract_kernel_source(
        self,
        call: cindex.Cursor,
        current_function: cindex.Cursor | None,
        tu_path: str,
    ) -> None:
        """Detect `descriptor.kernel_source = "path.cpp"` assignment patterns
        (newer style, used by ProgramDescriptor-based program factories).

        The cursor is a `CALL_EXPR` to `operator=` whose:
          - arg 0 is a MEMBER_REF_EXPR with spelling in `_KERNEL_PATH_FIELDS`
          - arg 1 is the RHS — either a STRING_LITERAL directly, or a
            DECL_REF_EXPR to a constexpr `const char*` whose initializer is one.
        """
        if current_function is None:
            return
        try:
            args = list(call.get_arguments())
        except Exception:
            return
        if len(args) < 2:
            return
        lhs, rhs = args[0], args[1]
        try:
            if lhs.kind != cindex.CursorKind.MEMBER_REF_EXPR:
                return
        except ValueError:
            return
        if lhs.spelling not in self._KERNEL_PATH_FIELDS:
            return
        # Multi-value resolution: catches the Flaw 13 pattern where the
        # RHS is a local variable that gets assigned different kernel
        # paths in different control-flow branches. e.g.
        #   std::string compute_kernel;
        #   if (...) compute_kernel = "a.cpp"; else compute_kernel = b ? "x.cpp" : "y.cpp";
        #   desc.kernel_source = compute_kernel;
        paths = self._resolve_string_values_all(rhs, scope=current_function)
        # Filter to actual kernel-file paths.
        paths = [p for p in paths if p and any(p.endswith(s) for s in self._KERNEL_PATH_SUFFIXES)]
        if not paths:
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        for path in paths:
            kernel_id = f"kernel:{path}"
            if kernel_id not in self.nodes:
                self.nodes[kernel_id] = Node(
                    id=kernel_id,
                    language="cpp",
                    kind="kernel_file",
                    name=path.rsplit("/", 1)[-1],
                    qualified_name=path,
                    file=path,
                    line_start=0,
                    line_end=0,
                    signature=path,
                    is_definition=False,
                    discovered_in=[tu_path],
                )
            elif tu_path not in self.nodes[kernel_id].discovered_in:
                self.nodes[kernel_id].discovered_in.append(tu_path)
            self.edges.append(Edge(
                src=node_id(current_function),
                dst=kernel_id,
                kind="launches",
                site_file=site_file,
                site_line=site_line,
                via="KernelDescriptor.kernel_source",
            ))

    def _extract_kernel_launch(
        self,
        call: cindex.Cursor,
        current_function: cindex.Cursor | None,
        launcher: str,
        tu_path: str,
    ) -> None:
        """Find the first string-literal arg that looks like a kernel source
        path, synthesize a `kind=kernel_file` node identified by that path,
        and emit a `kind=launches` edge from `current_function`.
        """
        if current_function is None:
            return
        try:
            args = list(call.get_arguments())
        except Exception:
            return
        path: str | None = None
        for arg in args:
            stack = [arg]
            while stack and path is None:
                c = stack.pop()
                try:
                    kind = c.kind
                except ValueError:
                    continue
                if kind == cindex.CursorKind.STRING_LITERAL:
                    sp = c.spelling
                    if sp and sp.startswith('"') and sp.endswith('"'):
                        sp = sp[1:-1]
                    if sp and any(sp.endswith(suf) for suf in self._KERNEL_PATH_SUFFIXES):
                        path = sp
                        break
                    # not a path-like literal: keep looking through later args
                if kind == cindex.CursorKind.LAMBDA_EXPR:
                    continue
                try:
                    stack.extend(reversed(list(c.get_children())))
                except ValueError:
                    pass
            if path is not None:
                break
        if path is None:
            return

        # Synthesize a stable node for the kernel file. ID is `kernel:` + the
        # path verbatim so the merger dedups across TUs that launch the same
        # kernel.
        kernel_id = f"kernel:{path}"
        if kernel_id not in self.nodes:
            self.nodes[kernel_id] = Node(
                id=kernel_id,
                language="cpp",
                kind="kernel_file",
                name=path.rsplit("/", 1)[-1],
                qualified_name=path,
                file=path,
                line_start=0,
                line_end=0,
                signature=path,
                is_definition=False,
                discovered_in=[tu_path],
            )
        elif tu_path not in self.nodes[kernel_id].discovered_in:
            self.nodes[kernel_id].discovered_in.append(tu_path)

        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        self.edges.append(Edge(
            src=node_id(current_function),
            dst=kernel_id,
            kind="launches",
            site_file=site_file,
            site_line=site_line,
            via=launcher,
        ))

    @staticmethod
    def _call_returns_class_(call: cindex.Cursor) -> bool:
        try:
            t = call.type
            if t is None:
                return False
            spelling = t.spelling or ""
        except Exception:
            return False
        # We don't want to match Type<class_<T>> or class_ as part of an unrelated
        # type name. Require it to be the outer type or namespace-qualified.
        return (
            spelling.startswith("class_<")
            or spelling.startswith("nb::class_<")
            or spelling.startswith("nanobind::class_<")
            or spelling.startswith("pybind11::class_<")
            or spelling.startswith("enum_<")
            or spelling.startswith("nb::enum_<")
            or spelling.startswith("nanobind::enum_<")
            or spelling.startswith("pybind11::enum_<")
        )

    def _extract_class_binding(self, call: cindex.Cursor, tu_path: str) -> None:
        """Handle `nb::class_<T>(module, "Name")` and tt-metal helpers like
        `tt_serializable_class<T>(module, "Name", ...)` — the Python class
        name bound to a C++ type T.

        Differs from .def-style bindings in TWO ways:
          1. The bound symbol is a TYPE, not a function pointer. We extract
             it from the call's RETURN TYPE's first template argument.
          2. The python name is whichever arg is a STRING_LITERAL —
             typically the SECOND arg (after the module), not the first.
        """
        # Walk every arg in order looking for the first string literal —
        # `_find_python_name` only looks at arg[0] which doesn't fit here.
        python_name: str | None = None
        try:
            args = list(call.get_arguments())
        except Exception:
            args = []
        for arg in args:
            stack = [arg]
            while stack:
                c = stack.pop()
                try:
                    kind = c.kind
                except ValueError:
                    continue
                if kind == cindex.CursorKind.STRING_LITERAL:
                    sp = c.spelling
                    if sp and sp.startswith('"') and sp.endswith('"'):
                        python_name = sp[1:-1]
                    else:
                        python_name = sp
                    break
                if kind == cindex.CursorKind.LAMBDA_EXPR:
                    continue
                try:
                    stack.extend(reversed(list(c.get_children())))
                except ValueError:
                    pass
            if python_name is not None:
                break
        if not python_name:
            return
        # Pull T out of the call's type. `nb::class_<T>(...)` has type
        # `nb::class_<T>`, and its first template argument is the bound type.
        # T might be a typedef / using alias (e.g. `using Tensor = tt::tt_metal::Tensor`)
        # — canonicalize to get to the real class.
        try:
            call_type = call.type
            if call_type is None:
                return
            if call_type.get_num_template_arguments() < 1:
                return
            t_type = call_type.get_template_argument_type(0)
            # Strip typedef / using to the underlying class.
            t_type_canon = t_type.get_canonical()
            t_cursor = t_type_canon.get_declaration()
        except Exception:
            return
        if t_cursor is None:
            return
        try:
            t_kind = t_cursor.kind
        except ValueError:
            return
        if t_kind not in CLASS_KINDS:
            return
        # Make sure the class is recorded as a node (so the binding has a
        # resolvable cpp_node_id even if no other TU saw the class definition).
        f, _, _ = cursor_loc(t_cursor)
        if in_scope(f):
            self._record_class(t_cursor, tu_path)
        bound_id = node_id(t_cursor)
        # Bindings dedup by (python_name, cpp_node_id, site_file, site_line);
        # we record this one explicitly.
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        self.bindings.append(Binding(
            python_name=python_name,
            cpp_node_id=bound_id,
            cpp_qualified_name=qualified_name(t_cursor),
            site_file=site_file,
            site_line=site_line,
            helper="class_",
        ))
        node = self.nodes.get(bound_id)
        if node is not None:
            node.is_binding_target = True

    def _extract_binding(self, call: cindex.Cursor, helper: str, tu_path: str) -> None:
        python_name = self._find_python_name(call, helper)
        if not python_name:
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        callee_cursor = call.referenced
        helper_usr = callee_cursor.get_usr() if callee_cursor else None
        for fn in self._find_referenced_decls(call, helper=helper, skip_usr=helper_usr):
            self.bindings.append(Binding(
                python_name=python_name,
                cpp_node_id=node_id(fn),
                cpp_qualified_name=qualified_name(fn),
                site_file=site_file,
                site_line=site_line,
                helper=helper,
            ))
            self._record_function(fn, tu_path)
            n = self.nodes.get(node_id(fn))
            if n is not None:
                n.is_binding_target = True

    @staticmethod
    def _find_python_name(call: cindex.Cursor, helper: str) -> str | None:
        """Extract the Python-name string literal from a binding call.

        Critically, we ONLY look at the call's direct arguments — not its full
        subtree. For chained calls like
            `nb::class_<T>(m, "Name").def_ro("field", &T::field)`
        walking the full subtree of the `.def_ro(...)` call would descend
        through its receiver into the outer constructor and pick up "Name"
        instead of "field".
        """
        try:
            args = list(call.get_arguments())
        except Exception:
            args = []
        for arg in args:
            # First string-literal in argument position is the Python name.
            # Descend through implicit casts and the like — but stop at lambdas.
            stack = [arg]
            while stack:
                c = stack.pop()
                try:
                    kind = c.kind
                except ValueError:
                    continue
                if kind == cindex.CursorKind.STRING_LITERAL:
                    sp = c.spelling
                    if sp and sp.startswith('"') and sp.endswith('"'):
                        return sp[1:-1]
                    return sp
                if kind == cindex.CursorKind.LAMBDA_EXPR:
                    continue
                try:
                    stack.extend(reversed(list(c.get_children())))
                except ValueError:
                    pass
            # Stop after the first non-string arg; the Python name is always first.
            break

        # Templated helpers like `bind_function<"name">(...)`: the literal sits
        # between angle brackets in the source. libclang does not expose this
        # cleanly as a child cursor, so fall back to token scanning bounded by
        # the call's source range.
        try:
            tokens = list(call.get_tokens())
        except Exception:
            tokens = []
        seen_helper = False
        depth = 0
        for tok in tokens:
            sp = tok.spelling
            if not seen_helper and sp == helper:
                seen_helper = True
                continue
            if not seen_helper:
                continue
            if sp == "<":
                depth += 1
                continue
            if sp == ">":
                depth -= 1
                if depth <= 0:
                    break
                continue
            if depth >= 1 and tok.kind == cindex.TokenKind.LITERAL and sp.startswith('"') and sp.endswith('"'):
                return sp[1:-1]
        return None

    @staticmethod
    def _find_referenced_decls(
        call: cindex.Cursor, helper: str | None = None, skip_usr: str | None = None
    ) -> Iterator[cindex.Cursor]:
        """Yield C++ declaration cursors that look like bound symbols.

        Covers both functions (for `.def`/`.def_static`/`.def_prop_*` bindings)
        and fields/static-vars (for `.def_ro` / `.def_rw` attribute bindings).

        Walks the call's subtree, BUT does not descend into lambda bodies.
        The bound symbol of `cls.def("name", &SomeFn)` is `SomeFn`, but the
        bound symbol of `cls.def("name", [](Args){ ... body ... })` is the
        lambda itself — the function references inside the body are
        implementation calls, not bound symbols.

        Filters applied:
          - binding helper itself (e.g. `bind_function`),
          - nanobind / pybind utility functions (`nb::arg::operator=`, etc.),
          - stdlib helpers (`std::`),
          - operator overloads,
          - anything that's the target of a method-call or function-call
            INSIDE a lambda or compound statement (skipped via no-descend).
        """
        FRAMEWORK_NAMESPACE_PREFIXES = (
            "nanobind::", "nb::",
            "pybind11::", "py::",
            "std::",
            "__builtin_",
        )

        def is_acceptable_ref(ref: cindex.Cursor) -> bool:
            try:
                ref_kind = ref.kind
            except ValueError:
                return False
            if (
                ref_kind not in BINDABLE_KINDS
                or ref.spelling in BINDING_HELPERS
                or ref.spelling.startswith("operator")
                or ref.spelling.startswith("__builtin_")
            ):
                return False
            # VAR_DECLs can be (a) static class members — legit binding
            # targets — or (b) function-local variables like `const auto doc`
            # that nanobind helpers pass through. Class-scoped vars stay.
            if ref_kind == cindex.CursorKind.VAR_DECL:
                try:
                    parent_kind = ref.semantic_parent.kind if ref.semantic_parent else None
                except ValueError:
                    parent_kind = None
                if parent_kind not in CLASS_KINDS:
                    return False
            # ENUM_CONSTANT_DECLs are only valid binding targets for `.value(...)`
            # calls on `nb::enum_<T>` instances. In other contexts they appear
            # as default-value arguments (`nb::arg("x") = SomeEnum::VAL`) and
            # must not be treated as bind targets.
            if ref_kind == cindex.CursorKind.ENUM_CONSTANT_DECL and helper != "value":
                return False
            qn = qualified_name(ref)
            if any(qn.startswith(p) for p in FRAMEWORK_NAMESPACE_PREFIXES):
                return False
            return True

        # Iterative pre-order walk starting from each direct argument — NOT
        # the entire call subtree, which for chained calls
        # `obj.def(...).def_ro(...)` would descend through the receiver and
        # incorrectly attribute references from an unrelated outer call.
        # Lambda subtrees are still skipped inside arg walks.
        seen: set[str] = set()
        lambda_subtrees: list[cindex.Cursor] = []  # collected for the fallback pass
        try:
            args = list(call.get_arguments())
        except Exception:
            args = []
        stack: list[cindex.Cursor] = list(reversed(args))
        first = True
        yielded = False
        while stack:
            c = stack.pop()
            try:
                kind = c.kind
            except ValueError:
                continue
            if not first and kind == cindex.CursorKind.LAMBDA_EXPR:
                lambda_subtrees.append(c)
                continue   # don't recurse into lambda body on the primary pass
            first = False
            if kind == cindex.CursorKind.DECL_REF_EXPR:
                ref = c.referenced
                if ref is not None and is_acceptable_ref(ref):
                    qn = qualified_name(ref)
                    key = ref.get_usr() or qn
                    if key != skip_usr and key not in seen:
                        seen.add(key)
                        yielded = True
                        yield ref
            # Push children for further inspection.
            try:
                children = list(c.get_children())
            except ValueError:
                continue
            stack.extend(reversed(children))

        # Flaw 8 fallback: if no direct function-pointer was found in the
        # call args, the binding is likely a thin lambda wrapper:
        #   mod.def("name", [args](...) { return target(...); })
        # Walk lambda bodies, find the first CALL_EXPR whose callee is an
        # acceptable bind target, and yield it. Only fires when the primary
        # pass yielded nothing — otherwise lambdas inside argument
        # expressions (call_guard etc.) would over-fire.
        if not yielded and lambda_subtrees:
            for lam in lambda_subtrees:
                # Collect ALL acceptable call refs inside the lambda body, then
                # pick the best one. Implicit-return-value constructors (`return
                # self.method()` creates a hidden Shape::Shape constructor wrapping
                # the actual `method()` call) appear FIRST in preorder, so taking
                # the first match yields the wrong target (Flaw 12). Prefer the
                # last non-CONSTRUCTOR CALL_EXPR, which is the innermost / real
                # call. Falls back to constructors if no other call exists.
                candidates: list[tuple[bool, cindex.Cursor]] = []  # (is_ctor, ref)
                stack2: list[cindex.Cursor] = [lam]
                while stack2:
                    c = stack2.pop()
                    try:
                        ck = c.kind
                    except ValueError:
                        continue
                    if ck == cindex.CursorKind.CALL_EXPR:
                        ref = c.referenced
                        if ref is not None and is_acceptable_ref(ref):
                            try:
                                is_ctor = ref.kind in (
                                    cindex.CursorKind.CONSTRUCTOR,
                                    cindex.CursorKind.DESTRUCTOR,
                                )
                            except ValueError:
                                is_ctor = False
                            candidates.append((is_ctor, ref))
                    try:
                        stack2.extend(reversed(list(c.get_children())))
                    except ValueError:
                        continue
                # Pick the first non-constructor candidate (innermost call after
                # skipping implicit return-value wrappers). If only constructors,
                # take the first one.
                chosen: cindex.Cursor | None = None
                for is_ctor, ref in candidates:
                    if not is_ctor:
                        chosen = ref
                        break
                if chosen is None and candidates:
                    chosen = candidates[0][1]
                if chosen is not None:
                    qn = qualified_name(chosen)
                    key = chosen.get_usr() or qn
                    if key != skip_usr and key not in seen:
                        seen.add(key)
                        yield chosen


# ─── JSONL helpers ─────────────────────────────────────────────────────────


def write_jsonl(path: Path, records) -> int:
    """Write an iterable of dataclasses (or dicts) as JSONL. Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            if hasattr(r, "__dataclass_fields__"):
                f.write(json.dumps(asdict(r), separators=(",", ":")))
            else:
                f.write(json.dumps(r, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

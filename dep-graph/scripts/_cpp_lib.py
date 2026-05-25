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
    "/workspace/tt_metal/",
    "/workspace/tt_stl/",
)
DEFAULT_OUT_OF_SCOPE_PREFIXES = (
    "/workspace/.cpmcache/",
    "/workspace/build_Release/",
    "/workspace/build/",
    "/workspace/tt_metal/third_party/",
    "/workspace/tt_metal/hw/",
    "/workspace/runtime/sfpi/",
    "/workspace/tests/",
    "/workspace/.github/",
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


@dataclass
class Binding:
    python_name: str
    cpp_node_id: str
    cpp_qualified_name: str
    site_file: str
    site_line: int
    helper: str


# ─── indexer ───────────────────────────────────────────────────────────────


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
    "def",
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
        if current_function is None or callee is None:
            return
        if callee.kind not in FUNCTION_KINDS:
            return
        src_file, _, _ = cursor_loc(current_function)
        callee_file, _, _ = cursor_loc(callee)
        if not (in_scope(src_file) and in_scope(callee_file)):
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        self.edges.append(Edge(
            src=node_id(current_function),
            dst=node_id(callee),
            kind="calls",
            site_file=site_file,
            site_line=site_line,
        ))

    def _extract_binding(self, call: cindex.Cursor, helper: str, tu_path: str) -> None:
        python_name = self._find_python_name(call, helper)
        if not python_name:
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        callee_cursor = call.referenced
        helper_usr = callee_cursor.get_usr() if callee_cursor else None
        for fn in self._find_referenced_functions(call, skip_usr=helper_usr):
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
        for c in call.walk_preorder():
            if c == call:
                continue
            if c.kind == cindex.CursorKind.STRING_LITERAL:
                spelling = c.spelling
                if spelling and spelling.startswith('"') and spelling.endswith('"'):
                    return spelling[1:-1]
                return spelling
            if c.kind == cindex.CursorKind.CALL_EXPR and c != call:
                continue

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
    def _find_referenced_functions(
        call: cindex.Cursor, skip_usr: str | None = None
    ) -> Iterator[cindex.Cursor]:
        """Yield C++ function cursors that look like bound symbols.

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
        # Iterative pre-order walk, skipping LAMBDA_EXPR subtrees.
        seen: set[str] = set()
        stack = [call]
        first = True
        while stack:
            c = stack.pop()
            try:
                kind = c.kind
            except ValueError:
                continue
            if not first and kind == cindex.CursorKind.LAMBDA_EXPR:
                continue   # don't recurse into lambda body
            first = False
            if kind == cindex.CursorKind.DECL_REF_EXPR:
                ref = c.referenced
                if ref is not None:
                    try:
                        ref_kind = ref.kind
                    except ValueError:
                        ref_kind = None
                    if (
                        ref_kind in FUNCTION_KINDS
                        and ref.spelling not in BINDING_HELPERS
                        and not ref.spelling.startswith("operator")
                        and not ref.spelling.startswith("__builtin_")
                    ):
                        qn = qualified_name(ref)
                        if not any(qn.startswith(p) for p in FRAMEWORK_NAMESPACE_PREFIXES):
                            key = ref.get_usr() or qn
                            if key != skip_usr and key not in seen:
                                seen.add(key)
                                yield ref
            # Push children for further inspection.
            try:
                children = list(c.get_children())
            except ValueError:
                continue
            stack.extend(reversed(children))


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

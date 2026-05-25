"""C++ AST indexer for the tt-metal dependency graph.

Runs inside the docker container. Given a compile_commands.json and one or
more translation units, walks each TU with libclang and emits:

    - nodes:    function/method declarations & definitions in scope
    - edges:    intra-C++ call edges (caller -> callee)
    - bindings: nanobind / pybind .def(...) mappings (python_name -> cpp symbol)

This is the vertical-slice version: deliberately minimal. The TU-arg stripping
logic (PCH, -Werror, -c, -o) was validated by probe_parse.py.

Usage:
    python cpp_index.py --db /workspace/build_Release/compile_commands.json \
        --tu /workspace/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp \
        --tu /workspace/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp \
        --out /workspace/dep-graph/cache/cpp_index.json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

from clang import cindex

# Match the compiler used in the build.
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
    """Drop args that libclang shouldn't see or can't consume.

    Removed:
      -o <out>          : not relevant to parsing
      -c                : implied
      -Werror, -pedantic-errors : avoid spurious "errors" from style warnings
      -Winvalid-pch     : meaningless without PCH
      -Xclang -include-pch -Xclang <path.pch>    : libclang 18 can't read clang-20 PCH
      -Xclang -include -Xclang <cmake_pch.hxx>   : the PCH header re-include
      the source file itself (so libclang receives it via the index.parse() arg)
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
            i += 4  # -Xclang -include-pch -Xclang <path>
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
    "/workspace/tt_metal/hostdevcommon/",  # mostly device-side
    "/workspace/runtime/sfpi/",
    "/workspace/tests/",
)


def in_scope(path: str | None) -> bool:
    if not path:
        return False
    p = os.path.normpath(path)
    if any(p.startswith(o) for o in DEFAULT_OUT_OF_SCOPE_PREFIXES):
        return False
    return any(p.startswith(s) for s in DEFAULT_IN_SCOPE_PREFIXES)


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
    """A reasonably canonical, human-readable signature."""
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
    """Unique id for a function/method/template.

    Uses the canonical cursor's USR. USR is the libclang-stable unique symbol
    string — robust across overloads and templates.
    """
    canonical = c.canonical
    usr = canonical.get_usr() or c.get_usr()
    return f"cpp:{usr}" if usr else f"cpp:{qualified_name(c)}"


def cursor_loc(c: cindex.Cursor) -> tuple[str | None, int, int]:
    if c.extent and c.extent.start.file:
        return (c.extent.start.file.name, c.extent.start.line, c.extent.end.line)
    if c.location and c.location.file:
        return (c.location.file.name, c.location.file.line if False else c.location.line, c.location.line)
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
    kind: str  # "calls" | "binds"
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
    helper: str  # which binding helper (e.g. "bind_binary_operation", "cls.def")


# ─── indexer ───────────────────────────────────────────────────────────────

BINDING_HELPERS = {
    # Generic ttnn helpers
    "bind_function",
    # eltwise/binary helpers seen in binary_nanobind.cpp
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
    # nanobind / pybind raw
    "def",  # cls.def("name", &sym) or mod.def("name", &sym)
}


class Indexer:
    def __init__(self, db: dict[str, TUEntry]) -> None:
        self.db = db
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.bindings: list[Binding] = []
        self.diagnostics: list[dict] = []
        self.index = cindex.Index.create()

    # ─── tu parsing ────────────────────────────────────────────────────

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
        return tu

    # ─── walking ───────────────────────────────────────────────────────

    def index_tu(self, tu_path: str) -> None:
        tu = self.parse_tu(tu_path)
        if tu is None:
            return
        self._walk(tu.cursor, current_function=None, tu_path=tu_path)

    def _walk(
        self,
        cursor: cindex.Cursor,
        current_function: cindex.Cursor | None,
        tu_path: str,
    ) -> None:
        # Function/method declaration or definition
        if cursor.kind in FUNCTION_KINDS:
            f, l0, l1 = cursor_loc(cursor)
            if in_scope(f):
                self._record_function(cursor, tu_path)
            # Recurse into the body with this as current function
            for child in cursor.get_children():
                self._walk(child, current_function=cursor if cursor.is_definition() else current_function, tu_path=tu_path)
            return

        # Call expression: emit a calls-edge from current_function -> callee
        if cursor.kind == cindex.CursorKind.CALL_EXPR:
            self._handle_call(cursor, current_function, tu_path)
            # fall through to recurse (calls can contain calls)

        for child in cursor.get_children():
            self._walk(child, current_function, tu_path)

    # ─── handlers ──────────────────────────────────────────────────────

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
            # Upgrade to definition if a later TU sees the body, and record TU
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
        # 1) Binding-helper detection
        if callee_name in BINDING_HELPERS:
            self._extract_binding(call, callee_name, tu_path)
        # 2) Plain call edge — emit only if we have both endpoints in scope
        if current_function is None or callee is None:
            return
        if callee.kind not in FUNCTION_KINDS:
            return
        src_file, _, _ = cursor_loc(current_function)
        callee_file, _, _ = cursor_loc(callee)
        # Both endpoints must be in scope; otherwise we get dangling edges
        # to nodes we deliberately don't record (std::*, framework internals).
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

    # ─── binding extraction ────────────────────────────────────────────

    def _extract_binding(self, call: cindex.Cursor, helper: str, tu_path: str) -> None:
        """Extract (python_name, cpp_symbol) from a binding helper call.

        Strategy:
          - Walk the call's child cursors. Among them are:
              * arguments (possibly nested in implicit casts / static_cast)
              * for templated helpers, the template argument may appear as the
                callee's referenced declaration's spelling. To recover the
                string literal we walk tokens between the helper name and '('.
          - Find string literals in arg position 0 (for `.def("name", ...)`) OR
            in the template arg list (for `bind_function<"name">`).
          - Find every DECL_REF_EXPR whose referenced cursor is a function;
            those are the bound C++ symbols.
        """
        python_name = self._find_python_name(call, helper)
        if not python_name:
            return
        site_file = call.location.file.name if call.location and call.location.file else ""
        site_line = call.location.line if call.location else 0
        # Skip the helper itself — its own DECL_REF_EXPR shows up in walk_preorder.
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
            # Mark the target node as a binding target
            self._record_function(fn, tu_path)
            n = self.nodes.get(node_id(fn))
            if n is not None:
                n.is_binding_target = True

    @staticmethod
    def _find_python_name(call: cindex.Cursor, helper: str) -> str | None:
        # 1) First StringLiteral child of the call (the .def("name", ...) case).
        for c in call.walk_preorder():
            if c == call:
                continue
            if c.kind == cindex.CursorKind.STRING_LITERAL:
                spelling = c.spelling
                if spelling and spelling.startswith('"') and spelling.endswith('"'):
                    return spelling[1:-1]
                return spelling
            # Don't descend into nested calls/lambdas — first literal wins
            if c.kind == cindex.CursorKind.CALL_EXPR and c != call:
                continue

        # 2) Template-argument case: walk the call's tokens between the helper
        #    identifier and the '(' for a string literal. Used by
        #    bind_function<"name"> / bind_binary_operation<"name">.
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
        seen: set[str] = set()
        for c in call.walk_preorder():
            if c.kind == cindex.CursorKind.DECL_REF_EXPR:
                ref = c.referenced
                if ref is not None and ref.kind in FUNCTION_KINDS:
                    # Skip the binding helper itself — it shows up as a DECL_REF
                    # both as the call's callee and (for template instantiations)
                    # as a child cursor whose USR may differ from call.referenced.
                    if ref.spelling in BINDING_HELPERS:
                        continue
                    key = ref.get_usr() or qualified_name(ref)
                    if key == skip_usr or key in seen:
                        continue
                    seen.add(key)
                    yield ref


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="compile_commands.json path")
    ap.add_argument("--tu", action="append", default=[], help="TU to index (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    db = load_db(Path(args.db))
    indexer = Indexer(db)

    tus = args.tu or sorted(db.keys())
    for tu in tus:
        print(f"[cpp_index] parsing {tu}", file=sys.stderr)
        indexer.index_tu(tu)

    out = {
        "nodes": [asdict(n) for n in indexer.nodes.values()],
        "edges": [asdict(e) for e in indexer.edges],
        "bindings": [asdict(b) for b in indexer.bindings],
        "diagnostics": indexer.diagnostics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"[cpp_index] {len(out['nodes'])} nodes, {len(out['edges'])} edges, "
        f"{len(out['bindings'])} bindings, {len(out['diagnostics'])} diags -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

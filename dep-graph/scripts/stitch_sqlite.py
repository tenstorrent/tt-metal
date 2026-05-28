"""Stitcher (SQLite edition).

Reads the merged C++ index (JSONL streams) and a Python index JSON, resolves
cross-language refs against the C++ binding table, and writes a SQLite DB
queryable by `query.py`.

Inputs:
    --cpp-dir   /workspace/dep-graph/cache/cpp_index        # merger output
    --py        /workspace/dep-graph/cache/py_index.json    # py_index output
    --out       /workspace/dep-graph/out/dep-graph.sqlite

Schema mirrors §11 P2-B in opus-instructions.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cpp_lib import read_jsonl  # noqa: E402


SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    language    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file        TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    signature   TEXT,
    is_definition INTEGER NOT NULL DEFAULT 0,
    is_binding_target INTEGER NOT NULL DEFAULT 0,
    is_binding_caller INTEGER NOT NULL DEFAULT 0,
    attrs_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_name  ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_file  ON nodes(file);
CREATE INDEX IF NOT EXISTS idx_nodes_lang  ON nodes(language);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    kind        TEXT NOT NULL,
    site_file   TEXT NOT NULL,
    site_line   INTEGER NOT NULL,
    crosses_language INTEGER NOT NULL DEFAULT 0,
    via_decorator TEXT,
    via_helper    TEXT,
    via_dispatch  TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
CREATE INDEX IF NOT EXISTS idx_edges_dispatch ON edges(via_dispatch);

CREATE TABLE IF NOT EXISTS bindings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    python_name TEXT NOT NULL,
    cpp_node_id TEXT NOT NULL,
    cpp_qualified_name TEXT NOT NULL,
    site_file   TEXT NOT NULL,
    site_line   INTEGER NOT NULL,
    helper      TEXT
);
CREATE INDEX IF NOT EXISTS idx_bindings_pyname ON bindings(python_name);
CREATE INDEX IF NOT EXISTS idx_bindings_cppid  ON bindings(cpp_node_id);

CREATE TABLE IF NOT EXISTS py_registrations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    python_name TEXT NOT NULL,             -- e.g. "ttnn.from_torch"
    impl_node_id TEXT,                     -- Python node when impl is a local def
    impl_chain   TEXT,                     -- JSON list when impl is an attr chain
    site_file    TEXT NOT NULL,
    site_line    INTEGER NOT NULL,
    decorator_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_pyreg_name ON py_registrations(python_name);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS cpp_diagnostics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tu        TEXT,
    severity  TEXT,
    message   TEXT,
    location  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cppdiag_severity ON cpp_diagnostics(severity);
CREATE INDEX IF NOT EXISTS idx_cppdiag_message  ON cpp_diagnostics(message);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    con.executescript(SCHEMA)
    return con


def insert_cpp_nodes(con: sqlite3.Connection, cpp_dir: Path) -> int:
    rows = []
    n = 0
    for node in read_jsonl(cpp_dir / "nodes.jsonl"):
        rows.append((
            node["id"], "cpp", node.get("kind", ""), node.get("name", ""),
            node.get("qualified_name", ""), node.get("file", ""),
            int(node.get("line_start", 0)), int(node.get("line_end", 0)),
            node.get("signature", ""), int(bool(node.get("is_definition"))),
            int(bool(node.get("is_binding_target"))), 0,
            json.dumps({
                "is_template": bool(node.get("is_template")),
                "discovered_in": node.get("discovered_in", []),
            }),
        ))
        if len(rows) >= 5000:
            con.executemany(
                "INSERT OR IGNORE INTO nodes "
                "(id, language, kind, name, qualified_name, file, line_start, line_end, "
                " signature, is_definition, is_binding_target, is_binding_caller, attrs_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            n += len(rows)
            rows.clear()
    if rows:
        con.executemany(
            "INSERT OR IGNORE INTO nodes "
            "(id, language, kind, name, qualified_name, file, line_start, line_end, "
            " signature, is_definition, is_binding_target, is_binding_caller, attrs_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        n += len(rows)
    return n


def insert_cpp_edges(con: sqlite3.Connection, cpp_dir: Path) -> int:
    rows = []
    n = 0
    for e in read_jsonl(cpp_dir / "edges.jsonl"):
        rows.append((
            e["src"], e["dst"], e.get("kind", "calls"),
            e.get("site_file", ""), int(e.get("site_line", 0)),
            int(bool(e.get("crosses_language"))), None,
            e.get("via"),  # via_helper — used by D1 launches edges (launcher name)
            e.get("via_dispatch"),
        ))
        if len(rows) >= 5000:
            con.executemany(
                "INSERT INTO edges (src, dst, kind, site_file, site_line, "
                "crosses_language, via_decorator, via_helper, via_dispatch) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            n += len(rows)
            rows.clear()
    if rows:
        con.executemany(
            "INSERT INTO edges (src, dst, kind, site_file, site_line, "
            "crosses_language, via_decorator, via_helper, via_dispatch) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        n += len(rows)
    return n


def insert_cpp_bindings(con: sqlite3.Connection, cpp_dir: Path) -> int:
    rows = []
    n = 0
    for b in read_jsonl(cpp_dir / "bindings.jsonl"):
        rows.append((
            b["python_name"], b["cpp_node_id"], b["cpp_qualified_name"],
            b.get("site_file", ""), int(b.get("site_line", 0)),
            b.get("helper"),
        ))
        if len(rows) >= 5000:
            con.executemany(
                "INSERT INTO bindings (python_name, cpp_node_id, cpp_qualified_name, "
                "site_file, site_line, helper) VALUES (?,?,?,?,?,?)",
                rows,
            )
            n += len(rows)
            rows.clear()
    if rows:
        con.executemany(
            "INSERT INTO bindings (python_name, cpp_node_id, cpp_qualified_name, "
            "site_file, site_line, helper) VALUES (?,?,?,?,?,?)",
            rows,
        )
        n += len(rows)
    return n


def insert_py_nodes_and_refs(con: sqlite3.Connection, py_json: Path) -> dict:
    py = json.loads(py_json.read_text())
    # nodes
    nrows = []
    for n in py["nodes"]:
        nrows.append((
            n["id"], "python", n.get("kind", ""), n.get("name", ""),
            n.get("qualified_name", ""), n.get("file", ""),
            int(n.get("line_start", 0)), int(n.get("line_end", 0)),
            None, 1,
            0, int(bool(n.get("is_binding_caller"))),
            json.dumps({"decorators": n.get("decorators", [])}),
        ))
    con.executemany(
        "INSERT OR IGNORE INTO nodes (id, language, kind, name, qualified_name, file, "
        "line_start, line_end, signature, is_definition, is_binding_target, "
        "is_binding_caller, attrs_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        nrows,
    )
    py_n = len(nrows)

    # intra-Python edges (calls / binds / imports / inherits).
    py_e = 0
    if py.get("edges"):
        erows = [
            (e["src"], e["dst"], e.get("kind", "calls"),
             e.get("site_file", ""), int(e.get("site_line", 0)),
             0, None, None, None)
            for e in py["edges"]
        ]
        con.executemany(
            "INSERT INTO edges (src, dst, kind, site_file, site_line, "
            "crosses_language, via_decorator, via_helper, via_dispatch) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            erows,
        )
        py_e = len(erows)

    # Python registrations (decorator + call form).
    reg_rows = []
    for r in py.get("registrations", []):
        reg_rows.append((
            r["python_name"], r.get("impl_node_id"),
            json.dumps(r["impl_chain"]) if r.get("impl_chain") else None,
            r["site_file"], int(r["site_line"]), r.get("decorator_label"),
        ))
    if reg_rows:
        con.executemany(
            "INSERT INTO py_registrations (python_name, impl_node_id, impl_chain, "
            "site_file, site_line, decorator_label) VALUES (?,?,?,?,?,?)",
            reg_rows,
        )

    # Build resolution maps.
    by_pyname_cpp: dict[str, list[tuple[str, str]]] = {}
    by_pyname_cpp_full: dict[str, list[tuple[str, str, str]]] = {}  # python_name → [(cpp_id, cpp_qn, helper), ...]
    for row in con.execute("SELECT python_name, cpp_node_id, helper, cpp_qualified_name FROM bindings"):
        by_pyname_cpp.setdefault(row[0], []).append((row[1], row[2]))
        by_pyname_cpp_full.setdefault(row[0], []).append((row[1], row[3], row[2]))

    # ─── Typed-method resolver (Option 2) ──────────────────────────────
    # For each class binding (helper='class_'), collect its methods/fields
    # keyed by python class name. Then for a ref like `["obj", "method"]`
    # with receiver_type=["ttnn","MemoryConfig"], we can look up "MemoryConfig"
    # → methods → find "method" → emit cross-language edge.
    class_cpp_by_pyname: dict[str, list[str]] = {}  # MemoryConfig → [tt::tt_metal::MemoryConfig, ...]
    for r in con.execute("SELECT python_name, cpp_qualified_name FROM bindings WHERE helper='class_'"):
        class_cpp_by_pyname.setdefault(r[0], []).append(r[1])
    # All non-class bindings keyed by (cpp_class_prefix, python_member_name) →
    # list of cpp_node_id targets. Lookup: given a cpp class name + member name
    # → list of bound method/field node ids.
    cpp_class_members: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in con.execute(
        "SELECT python_name, cpp_node_id, cpp_qualified_name, helper FROM bindings "
        "WHERE helper != 'class_'"
    ):
        py_member_name, member_node_id, member_cpp_qn, helper = r
        # Find the longest class-cpp-prefix among class bindings that matches.
        # Cache for speed: build once mapping cpp_qn → class_cpp_qn.
        for cpp_class_qn in class_cpp_by_pyname.values():
            # iterate flat list of class cpp qns
            pass
    # Build the cpp_class → member map cleanly.
    all_class_cpps = {cpp for cpps in class_cpp_by_pyname.values() for cpp in cpps}
    # Sort by length desc so longest prefix wins.
    sorted_class_cpps = sorted(all_class_cpps, key=len, reverse=True)
    cpp_class_members.clear()
    # First: explicit bindings (highest signal).
    for r in con.execute(
        "SELECT python_name, cpp_node_id, cpp_qualified_name FROM bindings WHERE helper != 'class_'"
    ):
        py_member_name, member_node_id, member_cpp_qn = r
        for class_cpp in sorted_class_cpps:
            prefix = class_cpp + "::"
            if member_cpp_qn.startswith(prefix):
                cpp_class_members.setdefault((class_cpp, py_member_name), []).append((member_node_id, "bound"))
                break
    # Second: raw cpp methods/fields. Many ttnn classes bind methods via
    # `bind_function<"name">` template machinery our extractor doesn't unwrap,
    # so we fall back to "method exists on the class" → match by prefix +
    # trailing identifier. Tagged 'raw' so consumers can filter.
    bound_pairs = set(cpp_class_members.keys())
    for class_cpp in sorted_class_cpps:
        prefix = class_cpp + "::"
        cursor = con.execute(
            "SELECT id, qualified_name FROM nodes "
            "WHERE language='cpp' AND kind IN ('CXX_METHOD','FUNCTION_DECL','FIELD_DECL','VAR_DECL') "
            "AND qualified_name LIKE ?",
            (prefix + "%",),
        )
        for node_id, qn in cursor:
            tail = qn[len(prefix):]
            # Skip nested classes / parameterised tails — only direct members
            # (no further `::` and no `<` template noise).
            if "::" in tail or "<" in tail or "(" in tail:
                continue
            # Strip trailing overload signature (parens) — already filtered.
            member_name = tail
            if not member_name:
                continue
            key = (class_cpp, member_name)
            if key in bound_pairs:
                continue  # bound entry wins
            cpp_class_members.setdefault(key, []).append((node_id, "raw"))

    # Build a Python-only class-methods table for receiver_type resolution
    # of types that have NO cpp binding (BenchmarkProfiler, ModelArgs, etc.).
    # Keyed by (bare_class_name, method_name) → list of method node_ids.
    # We collect from `qualified_name LIKE '%.<ClassName>.<MethodName>'`
    # patterns in the python nodes table.
    py_class_methods: dict[tuple[str, str], list[str]] = {}
    py_classes_by_name: dict[str, set[str]] = {}  # bare_name → set of class qns
    for row in con.execute(
        "SELECT id, qualified_name, name, kind FROM nodes WHERE language='python' AND kind='class'"
    ):
        node_id, qn, name, kind = row
        py_classes_by_name.setdefault(name, set()).add(qn)
    # Now find each python method/function whose qualified_name is
    # `<class_qn>.<method_name>` and bucket by (class_name, method_name).
    for row in con.execute(
        "SELECT id, qualified_name, name, kind FROM nodes "
        "WHERE language='python' AND kind IN ('method', 'function')"
    ):
        node_id, qn, name, kind = row
        # method qualname structure: <module>.<ClassName>.<method_name>
        # Strip trailing .<method_name> to get the candidate class qn.
        if "." not in qn:
            continue
        candidate_class_qn, method_name = qn.rsplit(".", 1)
        if method_name != name:
            continue
        # Is candidate_class_qn an actual class node we recorded?
        class_bare_name = candidate_class_qn.rsplit(".", 1)[-1]
        if candidate_class_qn in py_classes_by_name.get(class_bare_name, set()):
            py_class_methods.setdefault(
                (class_bare_name, method_name), []
            ).append(node_id)

    # Set of "module name" tokens we should reject as receiver_type[0] —
    # things that look like external/stdlib namespaces.
    EXTERNAL_PREFIXES = {
        "torch", "numpy", "np", "pandas", "pd", "matplotlib", "plt", "PIL",
        "scipy", "sklearn", "transformers", "tokenizers", "huggingface_hub",
        "torchvision", "torchaudio",
    }

    # Resolution helper bound to the maps above.
    def resolve_via_receiver_type(receiver_type, member_name):
        """Given a receiver_type chain and a method name, return list of
        (target_node_id, helper_label, crosses_language) edge targets.

        Two resolution paths:
          - cpp binding: look up receiver_type[-1] in class_cpp_by_pyname →
            check cpp_class_members for the method.
          - py class fallback: when the cpp path returns nothing, look up
            receiver_type[-1] in py_class_methods. Resolves to a Python
            method node (no cross-language edge).

        Rejects receiver_types whose leading namespace is an external
        library (torch, numpy, etc.) — those wouldn't match anyway and
        risk false positives if a tt-metal class shares a method name.
        """
        if not receiver_type:
            return []
        if receiver_type[0] in EXTERNAL_PREFIXES:
            return []
        py_class_name = receiver_type[-1]
        out: list[tuple[str, str, int]] = []

        # 1. C++ binding resolution (cpp class members).
        cpp_classes = class_cpp_by_pyname.get(py_class_name, [])
        if cpp_classes:
            for cpp_class in cpp_classes:
                hits = cpp_class_members.get((cpp_class, member_name), [])
                for node_id, src in hits:
                    label = f"{py_class_name}/{src}" if src else py_class_name
                    out.append((node_id, label, 1))
            if out:
                return out

        # 2. Python-only class resolution (no cpp binding for this type).
        py_hits = py_class_methods.get((py_class_name, member_name), [])
        for node_id in py_hits:
            out.append((node_id, f"py:{py_class_name}", 0))
        return out

    by_pyname_pyimpl: dict[str, list[tuple[str | None, list[str] | None, str | None]]] = {}
    for row in con.execute(
        "SELECT python_name, impl_node_id, impl_chain, decorator_label FROM py_registrations"
    ):
        by_pyname_pyimpl.setdefault(row[0], []).append((row[1], json.loads(row[2]) if row[2] else None, row[3]))

    cross_rows = []
    unresolved = 0
    resolved_via_pyreg = 0
    resolved_via_binding = 0
    resolved_via_receiver_type = 0
    from collections import Counter
    unresolved_chains: Counter = Counter()
    for r in py["refs"]:
        chain = r["target_chain"]
        kind = "binds" if r["kind"] == "attr_access" else "calls"

        # 0. Typed-method resolution (Option 2). Fires for ANY ref with a
        # known receiver type — chain[0] doesn't need to be `ttnn`. e.g.
        # `memory_config.is_sharded()` where memory_config: ttnn.MemoryConfig.
        receiver_type = r.get("receiver_type")
        if receiver_type and chain:
            type_hits: list[tuple[str, str, int]] = []
            if len(chain) >= 2:
                # `obj.method(...)`: chain[1] is the method name.
                type_hits = resolve_via_receiver_type(receiver_type, chain[1])
            elif len(chain) == 1 and kind == "calls":
                # `model(args)` where model is a typed local. In Python this
                # invokes `model.__call__`. Try __call__ first, then forward
                # (PyTorch-style; many tt-metal model classes have forward).
                for candidate in ("__call__", "forward"):
                    type_hits = resolve_via_receiver_type(receiver_type, candidate)
                    if type_hits:
                        break
            if type_hits:
                for tgt_id, label, crosses in type_hits:
                    cross_rows.append((
                        r["src"], tgt_id, kind,
                        r["site_file"], int(r["site_line"]),
                        crosses, None, f"receiver_type:{label}", None,
                    ))
                    resolved_via_receiver_type += 1
                continue  # this ref handled; don't fall through

        if not chain or chain[0] != "ttnn" or len(chain) < 2:
            unresolved += 1
            if chain:
                unresolved_chains[".".join(chain[:3])] += 1
            continue

        # 1. Prefer py_registrations matched on the FULL dotted name OR any
        # progressively-shorter prefix ending at chain[-1]. This catches
        # cases like `ttnn.operations.core.from_torch` where the registered
        # name is `ttnn.from_torch` (a re-export through __init__.py).
        full_name = ".".join(chain)
        pyreg_hits = by_pyname_pyimpl.get(full_name, [])
        # Also try the "head.tail" 2-component form which dominates the
        # registration table (`ttnn.X`).
        if not pyreg_hits and len(chain) >= 2:
            short_name = f"{chain[0]}.{chain[-1]}"
            if short_name != full_name:
                pyreg_hits = by_pyname_pyimpl.get(short_name, [])
        matched_here = False
        for impl_id, impl_chain, label in pyreg_hits:
            if impl_id:
                cross_rows.append((
                    r["src"], impl_id, kind, r["site_file"], int(r["site_line"]),
                    0,  # intra-Python edge — same language
                    label, None, None,
                ))
                resolved_via_pyreg += 1
                matched_here = True
            elif impl_chain:
                # Pass-through registration to a C++ binding (e.g. ttnn._ttnn.X).
                # The impl_chain like ['ttnn','_ttnn','operations','core','deallocate']
                # corresponds to cpp namespace `operations::core::deallocate`. Filter
                # the same-named bindings to those whose cpp_qualified_name *ends*
                # with the impl_chain (modulo the leading `ttnn`/`_ttnn` python
                # module wrappers). Without this, `deallocate` would fan out to
                # 3 different cpp methods (Flaw 11).
                candidate = impl_chain[-1]
                all_matches = by_pyname_cpp_full.get(candidate, [])
                # Build expected cpp suffix by stripping python-side wrappers from
                # the head and joining with `::`.
                stripped = list(impl_chain)
                while stripped and stripped[0] in ("ttnn", "_ttnn", "tt_metal"):
                    stripped = stripped[1:]
                suffix = "::".join(stripped) if stripped else candidate
                filtered: list[tuple[str, str]] = []
                for cpp_id, cpp_qn, helper in all_matches:
                    if cpp_qn == suffix or cpp_qn.endswith("::" + suffix):
                        filtered.append((cpp_id, helper))
                # Fallback: if suffix matching found nothing (e.g. the impl_chain
                # is too short or doesn't align), keep the old behaviour to
                # preserve recall.
                if not filtered:
                    filtered = [(c, h) for c, _, h in all_matches]
                for cpp_id, helper in filtered:
                    cross_rows.append((
                        r["src"], cpp_id, kind, r["site_file"], int(r["site_line"]),
                        1, label, helper, None,
                    ))
                    resolved_via_pyreg += 1
                    matched_here = True
        if matched_here:
            continue

        # 2. Fall back: try last-component match against C++ bindings.
        candidate = chain[-1]
        all_matches_full = by_pyname_cpp_full.get(candidate, [])
        if not all_matches_full:
            unresolved += 1
            unresolved_chains[".".join(chain[:3])] += 1
            continue
        # When chain starts with `ttnn`, prefer bindings whose cpp_qn is
        # in the `ttnn::` namespace — exclude class methods of unrelated
        # classes (Tensor::X, MeshDevice::X) which were causing fan-out
        # (Flaw 11 path 2). If filtering kills all matches, fall back to
        # the unfiltered set to preserve recall.
        matches: list[tuple[str, str]]
        if chain[0] == "ttnn":
            filtered = [(c, h) for c, qn, h in all_matches_full if qn.startswith("ttnn::")]
            matches = filtered if filtered else [(c, h) for c, _, h in all_matches_full]
        else:
            matches = [(c, h) for c, _, h in all_matches_full]
        for cpp_id, helper in matches:
            cross_rows.append((
                r["src"], cpp_id, kind, r["site_file"], int(r["site_line"]),
                1, None, helper, None,
            ))
            resolved_via_binding += 1

    if cross_rows:
        con.executemany(
            "INSERT INTO edges (src, dst, kind, site_file, site_line, "
            "crosses_language, via_decorator, via_helper, via_dispatch) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            cross_rows,
        )

    # Persist top unresolved chain prefixes so the report can read them
    # without re-counting against the (misleading) full ref list.
    con.execute("CREATE TABLE IF NOT EXISTS unresolved_top_refs (prefix TEXT PRIMARY KEY, count INTEGER)")
    con.execute("DELETE FROM unresolved_top_refs")
    if unresolved_chains:
        con.executemany(
            "INSERT INTO unresolved_top_refs (prefix, count) VALUES (?, ?)",
            unresolved_chains.most_common(50),
        )

    return {
        "py_n": py_n, "py_e": py_e,
        "registrations": len(reg_rows),
        "cross_edges": len(cross_rows),
        "resolved_via_pyreg": resolved_via_pyreg,
        "resolved_via_binding": resolved_via_binding,
        "resolved_via_receiver_type": resolved_via_receiver_type,
        "unresolved": unresolved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-dir", default="/workspace/dep-graph/cache/cpp_index")
    ap.add_argument("--py", default="/workspace/dep-graph/cache/py_index.json")
    ap.add_argument("--out", default="/workspace/dep-graph/out/dep-graph.sqlite")
    args = ap.parse_args()

    start = time.time()
    con = open_db(Path(args.out))
    cpp_dir = Path(args.cpp_dir)

    print(f"[stitch_sqlite] loading C++ index from {cpp_dir}", file=sys.stderr)
    nn = insert_cpp_nodes(con, cpp_dir)
    ne = insert_cpp_edges(con, cpp_dir)
    nb = insert_cpp_bindings(con, cpp_dir)
    # Stream cpp diagnostics into the DB for regression chains to query.
    dn = 0
    for d in read_jsonl(cpp_dir / "diagnostics.jsonl"):
        con.execute(
            "INSERT INTO cpp_diagnostics (tu, severity, message, location) VALUES (?,?,?,?)",
            (d.get("tu"), str(d.get("severity", "")), d.get("message", ""), d.get("location")),
        )
        dn += 1
    con.commit()
    print(f"  cpp: {nn} nodes, {ne} edges, {nb} bindings, {dn} diagnostics", file=sys.stderr)

    print(f"[stitch_sqlite] loading Python index from {args.py}", file=sys.stderr)
    pyres = insert_py_nodes_and_refs(con, Path(args.py))
    con.commit()
    print(
        f"  py:  {pyres['py_n']} nodes, {pyres['py_e']} edges, "
        f"{pyres['registrations']} registrations, {pyres['cross_edges']} edges from refs "
        f"(via_pyreg={pyres['resolved_via_pyreg']}, "
        f"via_binding={pyres['resolved_via_binding']}, "
        f"via_receiver_type={pyres['resolved_via_receiver_type']}), "
        f"{pyres['unresolved']} unresolved",
        file=sys.stderr,
    )

    con.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", "0.2"),
    )
    con.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("built_at", str(int(time.time()))),
    )
    con.commit()
    con.execute("ANALYZE")
    con.close()
    elapsed = time.time() - start
    print(f"[stitch_sqlite] done in {elapsed:.1f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

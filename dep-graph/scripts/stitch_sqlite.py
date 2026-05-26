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
    for row in con.execute("SELECT python_name, cpp_node_id, helper FROM bindings"):
        by_pyname_cpp.setdefault(row[0], []).append((row[1], row[2]))

    by_pyname_pyimpl: dict[str, list[tuple[str | None, list[str] | None, str | None]]] = {}
    for row in con.execute(
        "SELECT python_name, impl_node_id, impl_chain, decorator_label FROM py_registrations"
    ):
        by_pyname_pyimpl.setdefault(row[0], []).append((row[1], json.loads(row[2]) if row[2] else None, row[3]))

    cross_rows = []
    unresolved = 0
    resolved_via_pyreg = 0
    resolved_via_binding = 0
    for r in py["refs"]:
        chain = r["target_chain"]
        if not chain or chain[0] != "ttnn" or len(chain) < 2:
            unresolved += 1
            continue
        kind = "binds" if r["kind"] == "attr_access" else "calls"

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
                # Try to resolve via bindings.
                candidate = impl_chain[-1]
                for cpp_id, helper in by_pyname_cpp.get(candidate, []):
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
        matches = by_pyname_cpp.get(candidate, [])
        if not matches:
            unresolved += 1
            continue
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

    return {
        "py_n": py_n, "py_e": py_e,
        "registrations": len(reg_rows),
        "cross_edges": len(cross_rows),
        "resolved_via_pyreg": resolved_via_pyreg,
        "resolved_via_binding": resolved_via_binding,
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
        f"(via_pyreg={pyres['resolved_via_pyreg']}, via_binding={pyres['resolved_via_binding']}), "
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

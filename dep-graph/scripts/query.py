"""Interactive query CLI for dep-graph.sqlite.

Subcommands:
    callers     SYMBOL                  who calls this?
    callees     SYMBOL                  what does this call?
    blast       SYMBOL --depth N        BFS in both directions
    by-file     PATH                    every node defined in a file
    crosses     SYMBOL                  every cross-language edge touching SYMBOL
    bind        PYTHON_NAME             list C++ targets bound under a Python name
    find        PATTERN                 grep nodes by name/qualified_name/file

SYMBOL accepts either a node id (`cpp:c:@N@...` / `py:...`) or a unique
qualified name (`ttnn::add`). When the qualified name resolves to multiple
nodes, the query is applied to each.

Output goes to stdout in a compact, line-per-result form. Add --json for
machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = "/workspace/dep-graph/out/dep-graph.sqlite"


def resolve_symbol(con: sqlite3.Connection, symbol: str) -> list[str]:
    """Return one or more node ids that match `symbol`."""
    rows = con.execute(
        "SELECT id FROM nodes WHERE id = ? OR qualified_name = ?",
        (symbol, symbol),
    ).fetchall()
    return [r[0] for r in rows]


def fmt_node(row: sqlite3.Row | tuple) -> str:
    # (id, qualified_name, kind, language, file, line_start, signature)
    nid, qname, kind, lang, file, ln, sig = row
    where = file.replace("/workspace/", "") if file else ""
    return f"  [{lang}/{kind}] {qname}  @ {where}:{ln}"


def cmd_callers(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    targets = resolve_symbol(con, args.symbol)
    if not targets:
        print(f"no node matches {args.symbol!r}", file=sys.stderr)
        sys.exit(1)
    out = []
    for tid in targets:
        rows = con.execute(
            """
            SELECT n.id, n.qualified_name, n.kind, n.language, n.file, n.line_start, n.signature,
                   e.site_file, e.site_line, e.kind, e.crosses_language, e.via_helper
            FROM edges e JOIN nodes n ON n.id = e.src
            WHERE e.dst = ?
            ORDER BY n.language, n.qualified_name, e.site_line
            """,
            (tid,),
        ).fetchall()
        out.append({"target": tid, "callers": [
            {"id": r[0], "qualified_name": r[1], "kind": r[2], "language": r[3],
             "file": r[4], "line": r[5], "signature": r[6],
             "edge": {"site_file": r[7], "site_line": r[8], "kind": r[9],
                      "crosses_language": bool(r[10]), "via_helper": r[11]}}
            for r in rows
        ]})
    if args.json:
        print(json.dumps(out, indent=2))
        return
    for item in out:
        print(f"callers of {item['target']}: {len(item['callers'])}")
        for c in item["callers"][:args.limit or 200]:
            xl = " [cross-lang]" if c["edge"]["crosses_language"] else ""
            print(
                f"  {c['language']}/{c['kind']:13s} {c['qualified_name']:60s} "
                f"@ {c['file'].replace('/workspace/','')}:{c['line']}"
                f"  via {c['edge']['kind']}{xl}"
            )
        if args.limit and len(item['callers']) > args.limit:
            print(f"  … and {len(item['callers']) - args.limit} more")


def cmd_callees(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    targets = resolve_symbol(con, args.symbol)
    if not targets:
        print(f"no node matches {args.symbol!r}", file=sys.stderr)
        sys.exit(1)
    out = []
    for sid in targets:
        rows = con.execute(
            """
            SELECT n.id, n.qualified_name, n.kind, n.language, n.file, n.line_start, n.signature,
                   e.site_file, e.site_line, e.kind, e.crosses_language, e.via_helper
            FROM edges e JOIN nodes n ON n.id = e.dst
            WHERE e.src = ?
            ORDER BY n.language, n.qualified_name, e.site_line
            """,
            (sid,),
        ).fetchall()
        out.append({"source": sid, "callees": [
            {"id": r[0], "qualified_name": r[1], "kind": r[2], "language": r[3],
             "file": r[4], "line": r[5], "signature": r[6],
             "edge": {"site_file": r[7], "site_line": r[8], "kind": r[9],
                      "crosses_language": bool(r[10]), "via_helper": r[11]}}
            for r in rows
        ]})
    if args.json:
        print(json.dumps(out, indent=2))
        return
    for item in out:
        print(f"callees of {item['source']}: {len(item['callees'])}")
        for c in item["callees"][:args.limit or 200]:
            xl = " [cross-lang]" if c["edge"]["crosses_language"] else ""
            print(
                f"  {c['language']}/{c['kind']:13s} {c['qualified_name']:60s} "
                f"@ {c['file'].replace('/workspace/','')}:{c['line']}"
                f"  via {c['edge']['kind']}{xl}"
            )


def cmd_blast(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    seeds = resolve_symbol(con, args.symbol)
    if not seeds:
        print(f"no node matches {args.symbol!r}", file=sys.stderr)
        sys.exit(1)
    upstream: dict[str, int] = {s: 0 for s in seeds}
    downstream: dict[str, int] = {s: 0 for s in seeds}
    frontier_u = list(seeds)
    frontier_d = list(seeds)
    for depth in range(1, args.depth + 1):
        if not frontier_u and not frontier_d:
            break
        if frontier_u:
            placeholders = ",".join("?" * len(frontier_u))
            rows = con.execute(
                f"SELECT DISTINCT src FROM edges WHERE dst IN ({placeholders})",
                frontier_u,
            ).fetchall()
            next_u = [r[0] for r in rows if r[0] not in upstream]
            for n in next_u:
                upstream[n] = depth
            frontier_u = next_u
        if frontier_d:
            placeholders = ",".join("?" * len(frontier_d))
            rows = con.execute(
                f"SELECT DISTINCT dst FROM edges WHERE src IN ({placeholders})",
                frontier_d,
            ).fetchall()
            next_d = [r[0] for r in rows if r[0] not in downstream]
            for n in next_d:
                downstream[n] = depth
            frontier_d = next_d

    summary = {
        "seeds": seeds,
        "depth": args.depth,
        "upstream_count": len(upstream) - len(seeds),
        "downstream_count": len(downstream) - len(seeds),
    }
    if args.json:
        # Decorate with node info
        def info(node_ids):
            if not node_ids:
                return []
            ph = ",".join("?" * len(node_ids))
            return [
                dict(zip(("id","qualified_name","kind","language","file","line_start"), r))
                for r in con.execute(
                    f"SELECT id, qualified_name, kind, language, file, line_start FROM nodes WHERE id IN ({ph})",
                    list(node_ids),
                )
            ]
        summary["upstream"] = info([k for k in upstream if k not in seeds])
        summary["downstream"] = info([k for k in downstream if k not in seeds])
        print(json.dumps(summary, indent=2))
        return
    print(f"blast radius from {len(seeds)} seed(s), depth={args.depth}")
    print(f"  upstream  (callers, transitively): {summary['upstream_count']}")
    print(f"  downstream (callees, transitively): {summary['downstream_count']}")


def cmd_by_file(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    pat = args.path if args.path.startswith("/") else f"%{args.path}%"
    rows = con.execute(
        "SELECT id, qualified_name, kind, language, file, line_start, signature "
        "FROM nodes WHERE file LIKE ? ORDER BY file, line_start",
        (pat,),
    ).fetchall()
    if args.json:
        print(json.dumps([
            dict(zip(("id","qualified_name","kind","language","file","line_start","signature"), r))
            for r in rows
        ], indent=2))
        return
    for r in rows:
        print(fmt_node(r))


def cmd_crosses(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    targets = resolve_symbol(con, args.symbol)
    if not targets:
        print(f"no node matches {args.symbol!r}", file=sys.stderr)
        sys.exit(1)
    out_rows = []
    for t in targets:
        rows = con.execute(
            """
            SELECT e.src, e.dst, e.kind, e.site_file, e.site_line, e.via_helper,
                   ns.qualified_name, ns.language, nd.qualified_name, nd.language
            FROM edges e
            LEFT JOIN nodes ns ON ns.id = e.src
            LEFT JOIN nodes nd ON nd.id = e.dst
            WHERE e.crosses_language = 1 AND (e.src = ? OR e.dst = ?)
            ORDER BY e.site_file, e.site_line
            """,
            (t, t),
        ).fetchall()
        out_rows.extend(rows)
    if args.json:
        print(json.dumps([
            dict(zip(("src","dst","kind","site_file","site_line","via_helper",
                      "src_qname","src_lang","dst_qname","dst_lang"), r))
            for r in out_rows
        ], indent=2))
        return
    for r in out_rows:
        src, dst, kind, sf, sl, via, sq, slng, dq, dlng = r
        print(f"  {slng or '?'}:{sq or src}  -[{kind}/{via}]->  {dlng or '?'}:{dq or dst}"
              f"  @ {sf.replace('/workspace/','')}:{sl}")


def cmd_bind(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = con.execute(
        "SELECT python_name, cpp_qualified_name, cpp_node_id, helper, site_file, site_line "
        "FROM bindings WHERE python_name = ? ORDER BY site_file, site_line",
        (args.python_name,),
    ).fetchall()
    if args.json:
        print(json.dumps([
            dict(zip(("python_name","cpp_qualified_name","cpp_node_id","helper","site_file","site_line"), r))
            for r in rows
        ], indent=2))
        return
    for r in rows:
        pn, qn, nid, h, sf, sl = r
        print(f"  {pn!r} -> {qn}  via {h}  @ {sf.replace('/workspace/','')}:{sl}")


def cmd_find(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    pat = f"%{args.pattern}%"
    rows = con.execute(
        "SELECT id, qualified_name, kind, language, file, line_start, signature "
        "FROM nodes WHERE name LIKE ? OR qualified_name LIKE ? OR file LIKE ? "
        "ORDER BY language, qualified_name LIMIT ?",
        (pat, pat, pat, args.limit or 200),
    ).fetchall()
    if args.json:
        print(json.dumps([
            dict(zip(("id","qualified_name","kind","language","file","line_start","signature"), r))
            for r in rows
        ], indent=2))
        return
    for r in rows:
        print(fmt_node(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_callers = sub.add_parser("callers"); p_callers.add_argument("symbol"); p_callers.add_argument("--limit", type=int, default=200); p_callers.set_defaults(func=cmd_callers)
    p_callees = sub.add_parser("callees"); p_callees.add_argument("symbol"); p_callees.add_argument("--limit", type=int, default=200); p_callees.set_defaults(func=cmd_callees)
    p_blast = sub.add_parser("blast"); p_blast.add_argument("symbol"); p_blast.add_argument("--depth", type=int, default=2); p_blast.set_defaults(func=cmd_blast)
    p_byfile = sub.add_parser("by-file"); p_byfile.add_argument("path"); p_byfile.set_defaults(func=cmd_by_file)
    p_crosses = sub.add_parser("crosses"); p_crosses.add_argument("symbol"); p_crosses.set_defaults(func=cmd_crosses)
    p_bind = sub.add_parser("bind"); p_bind.add_argument("python_name"); p_bind.set_defaults(func=cmd_bind)
    p_find = sub.add_parser("find"); p_find.add_argument("pattern"); p_find.add_argument("--limit", type=int, default=200); p_find.set_defaults(func=cmd_find)

    args = ap.parse_args()
    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(2)
    con = sqlite3.connect(args.db)
    args.func(con, args)


if __name__ == "__main__":
    main()

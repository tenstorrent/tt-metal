"""dep-graph/scripts/report.py — coverage metrics for dep-graph.sqlite.

A one-shot summary intended to be re-run after every index pipeline run.
Pulls aggregate stats from the SQLite DB plus (if present) the cached
py_index.json for unresolved-ref histograms.

Usage:
    python dep-graph/scripts/report.py
    python dep-graph/scripts/report.py --db <path> --py-index <path> --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB = "/workspace/dep-graph/out/dep-graph.sqlite"
DEFAULT_PY_INDEX = "/workspace/dep-graph/cache/py_index.json"


def gather(db_path: Path, py_index_path: Path | None) -> dict:
    con = sqlite3.connect(db_path)

    def q(sql: str, *args) -> int:
        return con.execute(sql, args).fetchone()[0]

    summary: dict[str, object] = {}

    # Basic counts.
    summary["nodes_total"] = q("SELECT COUNT(*) FROM nodes")
    summary["nodes_cpp"] = q("SELECT COUNT(*) FROM nodes WHERE language='cpp'")
    summary["nodes_py"] = q("SELECT COUNT(*) FROM nodes WHERE language='python'")
    summary["nodes_class_cpp"] = q("SELECT COUNT(*) FROM nodes WHERE language='cpp' AND kind='class'")
    summary["nodes_class_py"] = q("SELECT COUNT(*) FROM nodes WHERE language='python' AND kind='class'")
    summary["nodes_kernel_file"] = q("SELECT COUNT(*) FROM nodes WHERE kind='kernel_file'")
    summary["edges_total"] = q("SELECT COUNT(*) FROM edges")
    summary["bindings_total"] = q("SELECT COUNT(*) FROM bindings")
    summary["py_registrations"] = q("SELECT COUNT(*) FROM py_registrations")
    summary["cpp_diagnostics"] = q("SELECT COUNT(*) FROM cpp_diagnostics")

    # Edge kinds.
    edge_kinds = dict(con.execute("SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY COUNT(*) DESC").fetchall())
    summary["edges_by_kind"] = edge_kinds

    # Cross-language edges.
    summary["edges_cross_language"] = q("SELECT COUNT(*) FROM edges WHERE crosses_language=1")

    # Virtual-dispatch edges.
    summary["edges_virtual_dispatch"] = q("SELECT COUNT(*) FROM edges WHERE via_dispatch='virtual'")

    # py→py and py→cpp breakdown.
    direction_counts = con.execute("""
        SELECT
          CASE
            WHEN ns.language='python' AND nd.language='python' THEN 'py->py'
            WHEN ns.language='python' AND nd.language='cpp'    THEN 'py->cpp'
            WHEN ns.language='cpp'    AND nd.language='cpp'    THEN 'cpp->cpp'
            WHEN ns.language='cpp'    AND nd.language='python' THEN 'cpp->py'
            ELSE 'other'
          END AS dir,
          COUNT(*) AS n
        FROM edges e
        JOIN nodes ns ON ns.id=e.src
        JOIN nodes nd ON nd.id=e.dst
        GROUP BY dir
        ORDER BY n DESC
    """).fetchall()
    summary["edges_by_direction"] = dict(direction_counts)

    # Reverse-binding coverage.
    bind_total = q("SELECT COUNT(DISTINCT cpp_node_id) FROM bindings")
    bind_reached = q("""
        SELECT COUNT(DISTINCT b.cpp_node_id)
        FROM bindings b
        JOIN edges e ON e.dst = b.cpp_node_id
        JOIN nodes ns ON ns.id = e.src
        WHERE ns.language = 'python'
    """)
    summary["binding_targets_total"] = bind_total
    summary["binding_targets_reached_from_py"] = bind_reached
    summary["binding_coverage_pct"] = round(100.0 * bind_reached / bind_total, 1) if bind_total else 0.0

    # Python orphan rate.
    py_callables = q("SELECT COUNT(*) FROM nodes WHERE language='python' AND kind IN ('function','method','class')")
    py_with_incoming = q("""
        SELECT COUNT(DISTINCT n.id)
        FROM nodes n
        JOIN edges e ON e.dst = n.id
        WHERE n.language='python' AND n.kind IN ('function','method','class')
    """)
    summary["py_callables"] = py_callables
    summary["py_callables_with_incoming_edges"] = py_with_incoming
    summary["py_orphan_pct"] = round(100.0 * (py_callables - py_with_incoming) / py_callables, 1) if py_callables else 0.0

    # Truly-unresolved refs come from the stitcher (it knows which refs
    # didn't get a cross-language or intra-python edge). They're stored in
    # `unresolved_top_refs` so we don't have to re-run resolution here.
    has_table = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='unresolved_top_refs'"
    ).fetchone() is not None
    if has_table:
        total_unresolved = con.execute("SELECT IFNULL(SUM(count), 0) FROM unresolved_top_refs").fetchone()[0]
        summary["unresolved_refs_total"] = int(total_unresolved)
        resolved_proxy = summary["edges_cross_language"] + summary["edges_by_direction"].get("py->py", 0)
        summary["resolution_rate_pct"] = round(
            100.0 * resolved_proxy / (resolved_proxy + total_unresolved), 1
        ) if (resolved_proxy + total_unresolved) else 0.0
        summary["top_unresolved_refs"] = list(
            con.execute("SELECT prefix, count FROM unresolved_top_refs ORDER BY count DESC LIMIT 20")
        )
        # In-scope unresolved: exclude obvious stdlib / external / builtin /
        # exception prefixes. What's left is what we could plausibly resolve
        # with more work on the indexer.
        OUT_OF_SCOPE_HEADS = {
            # third-party / external libs
            "pytest","logger","torch","numpy","np","pandas","pd","matplotlib","plt",
            "PIL","Image","scipy","sklearn","requests","tqdm","tabulate","HfApi",
            "huggingface_hub","torchvision","torchaudio","librosa","torchtune",
            "transformers","diffusers","tokenizers","wandb","lightning","accelerate",
            "onnx","onnxruntime","sentencepiece","safetensors","rich","timm",
            "sentence_transformers","model_tracer","tracy",
            # python stdlib
            "os","sys","re","time","json","random","math","collections","itertools",
            "functools","tempfile","pathlib","unittest","typing","subprocess","enum",
            "warnings","datetime","argparse","shutil","csv","pickle","copy","string",
            "io","glob","hashlib","logging","inspect","traceback","gc","dataclasses",
            "signal","asyncio","queue","threading","urllib","base64","contextlib",
            "dis","tokenize","weakref","ssl","socket","operator","struct","Path",
            # builtins
            "len","range","print","isinstance","enumerate","zip","sorted","reversed",
            "map","filter","min","max","sum","abs","any","all","tuple","list","dict",
            "set","int","float","str","bool","bytes","type","hasattr","getattr",
            "setattr","repr","vars","dir","iter","next","open","round","divmod","pow",
            "id","super","partial","frozenset","bytearray","breakpoint","memoryview",
            "classmethod","staticmethod","property","object","eval","exec","compile",
            "hex","oct","bin","chr","ord","format","help","globals","locals","input",
            "ascii","callable","delattr","complex",
            # exceptions
            "ValueError","KeyError","TypeError","IndexError","StopIteration",
            "AssertionError","NotImplementedError","RuntimeError","OverflowError",
            "AttributeError","ImportError","ZeroDivisionError","FileExistsError",
            "FileNotFoundError","NameError","GeneratorExit","SystemExit",
            "PermissionError","OSError","Exception","BaseException","Warning",
            "DeprecationWarning",
            # tt-metal in-scope but unresolvable without deeper type inference:
            # `profiler = Profiler()` then `profiler.start()` — module-var of
            # py-only class type; `parser = argparse.ArgumentParser()` then
            # `parser.add_argument(...)` — argparse instance type.
            "profiler","parser",
        }
        in_scope_unresolved: list[tuple[str, int]] = []
        in_scope_total = 0
        for prefix, count in con.execute(
            "SELECT prefix, count FROM unresolved_top_refs ORDER BY count DESC"
        ):
            head = prefix.split(".", 1)[0]
            if head in OUT_OF_SCOPE_HEADS:
                continue
            in_scope_total += count
            if len(in_scope_unresolved) < 20:
                in_scope_unresolved.append((prefix, count))
        summary["in_scope_unresolved_total"] = in_scope_total
        summary["top_in_scope_unresolved_refs"] = in_scope_unresolved
    else:
        summary["unresolved_refs_total"] = None
        summary["resolution_rate_pct"] = None
        summary["top_unresolved_refs"] = []
        summary["in_scope_unresolved_total"] = None
        summary["top_in_scope_unresolved_refs"] = []

    return summary


def render(s: dict) -> str:
    out = []
    out.append("═══════════════════════════════════════════════════════════")
    out.append("  dep-graph coverage report")
    out.append("═══════════════════════════════════════════════════════════")
    out.append("")
    out.append(f"  Nodes: {s['nodes_total']:>7d}  ({s['nodes_cpp']} cpp, {s['nodes_py']} py)")
    out.append(f"    classes (cpp/py):   {s['nodes_class_cpp']} / {s['nodes_class_py']}")
    out.append(f"    kernel_file:        {s['nodes_kernel_file']}")
    out.append(f"  Edges: {s['edges_total']:>7d}")
    for k, n in s["edges_by_kind"].items():
        out.append(f"    kind={k:<10s} {n}")
    out.append("  Edges by direction:")
    for d, n in s["edges_by_direction"].items():
        out.append(f"    {d:<10s} {n}")
    out.append(f"  Cross-language edges: {s['edges_cross_language']}")
    out.append(f"  Virtual-dispatch edges: {s['edges_virtual_dispatch']}")
    out.append(f"  Bindings: {s['bindings_total']}  (Python registrations: {s['py_registrations']})")
    out.append(f"  cpp diagnostics: {s['cpp_diagnostics']}")
    out.append("")
    out.append("  ── Coverage ──")
    out.append(f"    Reverse-binding coverage: {s['binding_targets_reached_from_py']}/{s['binding_targets_total']}  ({s['binding_coverage_pct']}%)")
    out.append(f"    Python orphan rate:       {s['py_callables'] - s['py_callables_with_incoming_edges']}/{s['py_callables']}  ({s['py_orphan_pct']}%)")
    if s["resolution_rate_pct"] is not None:
        out.append(f"    Ref resolution rate:      {s['resolution_rate_pct']}%  ({s['unresolved_refs_total']} refs still unresolved)")
        if s.get("in_scope_unresolved_total") is not None:
            out.append(f"    In-scope unresolved:      {s['in_scope_unresolved_total']}  (excluding stdlib / external / builtins / exceptions)")
    out.append("")
    if s["top_unresolved_refs"]:
        out.append("  ── Top 20 unresolved ref prefixes (raw, includes out-of-scope) ──")
        for name, n in s["top_unresolved_refs"]:
            out.append(f"    {n:>5d}  {name}")
    out.append("")
    if s.get("top_in_scope_unresolved_refs"):
        out.append("  ── Top 20 in-scope unresolved ref prefixes (actionable) ──")
        for name, n in s["top_in_scope_unresolved_refs"]:
            out.append(f"    {n:>5d}  {name}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--py-index", default=DEFAULT_PY_INDEX)
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(2)
    py_path = Path(args.py_index) if args.py_index else None

    summary = gather(db_path, py_path)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(render(summary))


if __name__ == "__main__":
    main()

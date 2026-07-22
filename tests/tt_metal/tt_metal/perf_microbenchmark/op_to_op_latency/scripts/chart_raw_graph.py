#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render a device-program dependency graph (RAW edges) from a ttnn graph capture, via graphviz.

Nodes = device programs in program order (top->bottom). Black arrows = RAW data dependencies
(producer buffer -> consumer, matched on buffer address). Node coloring:
  - GREEN border  = the adjacency boundary BEFORE this op is REORDERABLE (op N-1's output is not read
                    by this op) -> the op-boundary barrier there could be relaxed.
  - GRAY dashed   = in-place op that records NO output tensor -> its consumers are UNRESOLVABLE from the
                    capture (conservative: not counted as reorderable).
  - white         = ordinary RAW-chained boundary (barrier needed).
Faint gray edges chain the program order. Title carries the summary stats.

Usage: chart_raw_graph.py <capture.json> --out <base> [--title "..."]  -> <base>.png and <base>.svg
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_hazard_analyzer import parse_device_ops, analyze, apply_resolutions, infer_attention_edges  # noqa: E402


def short(name):
    n = name.split("::")[-1].replace("DeviceOperation", "").replace("Operation", "")
    return n[:26]


def build(capture_json, out_base, title, resolve_inplace=False, add_edges=(), node_range=None):
    import graphviz

    nodes = json.load(open(capture_json))
    ops = parse_device_ops(nodes)
    # inferred edges (drawn dashed-orange): explicit --add-edge plus auto create-heads->SDPA when resolving
    inferred = set(add_edges) | (set(infer_attention_edges(ops)) if resolve_inplace else set())
    apply_resolutions(ops, resolve_inplace, add_edges)
    n = len(ops)
    m = analyze(ops)
    # optional slice for legibility on huge graphs: draw only ops [lo, hi) and edges within
    lo, hi = node_range if node_range else (0, n)
    in_win = lambda i: lo <= i < hi

    def producer_before(addr, i):
        p = None
        for k in range(i):
            if addr in ops[k]["out_ids"]:
                p = k
        return p

    raw_edges = set()
    for i, op in enumerate(ops):
        for a in op["in_ids"]:
            p = producer_before(a, i)
            if p is not None:
                raw_edges.add((p, i))

    # Classify the boundary BEFORE op i (between i-1 and i):
    #   'raw'         : op i reads op i-1's output -> barrier needed
    #   'reorderable' : op i does NOT read it AND op i-1 recorded an output -> barrier relaxable (confirmed)
    #   'unresolved'  : op i-1 is in-place (recorded NO output) -> dependency can't be seen -> can't tell
    boundary = [None] * n
    for i in range(1, n):
        if ops[i - 1]["out_ids"] & ops[i]["in_ids"]:
            boundary[i] = "raw"
        elif not ops[i - 1]["out_ids"]:
            boundary[i] = "unresolved"
        else:
            boundary[i] = "reorderable"

    g = graphviz.Digraph("raw", format="png")
    g.attr(rankdir="TB", nodesep="0.25", ranksep="0.35", bgcolor="white")
    g.attr("node", shape="box", style="filled,rounded", fillcolor="white", fontname="Helvetica", fontsize="10")

    BORDER = {
        "reorderable": ("#2e7d32", "3"),
        "unresolved": ("#ef6c00", "3"),
        "raw": ("#444444", "1"),
        None: ("#444444", "1"),
    }
    for i, op in enumerate(ops):
        if not in_win(i):
            continue
        in_place = not op["out_ids"]
        label = f"{i}: {short(op['name'])}" + ("\\n(in-place op)" if in_place else "")
        col, pw = BORDER[boundary[i]]
        style = "filled,rounded,dashed" if in_place else "filled,rounded"
        fill = "#fff3e0" if boundary[i] == "unresolved" else ("#eeeeee" if in_place else "white")
        g.node(str(i), label=label, color=col, penwidth=pw, style=style, fillcolor=fill)

    # faint program-order chain
    for i in range(n - 1):
        if in_win(i) and in_win(i + 1):
            g.edge(str(i), str(i + 1), style="dotted", color="#cccccc", arrowhead="none", weight="10")
    # RAW dependency edges (black solid = seen in capture; dashed orange = inferred/injected semantic edge)
    for p, c in sorted(raw_edges):
        if not (in_win(p) and in_win(c)):
            continue
        if (p, c) in inferred:
            g.edge(
                str(p),
                str(c),
                color="#ef6c00",
                penwidth="1.6",
                style="dashed",
                label="inferred",
                fontsize="8",
                fontcolor="#ef6c00",
            )
        else:
            g.edge(str(p), str(c), color="#111111", penwidth="1.4", constraint="false" if c - p > 1 else "true")

    reorder_cnt = boundary.count("reorderable")
    unresolved_cnt = boundary.count("unresolved")
    raw_cnt = boundary.count("raw")
    g.attr(
        label=(
            f"{title}\\l"
            f"{n} device programs · boundaries: {raw_cnt} RAW / {reorder_cnt} reorderable / {unresolved_cnt} unresolved "
            f"· critical path {m['critical_path']}/{n}\\l"
            f"GREEN border = barrier relaxable (confirmed) · ORANGE = unresolved (predecessor in-place) · "
            f"gray dashed = in-place op · black arrow = RAW dep\\l"
        ),
        labelloc="t",
        fontsize="12",
        fontname="Helvetica",
    )

    png = g.render(out_base, cleanup=True)
    g.format = "svg"
    svg = g.render(out_base, cleanup=True)
    print(
        f"nodes={n} raw_edges={len(raw_edges)} boundaries: raw={raw_cnt} reorderable={reorder_cnt} unresolved={unresolved_cnt}"
    )
    print(f"-> {png}\n-> {svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_json")
    ap.add_argument("--out", default="/tmp/raw_graph")
    ap.add_argument("--title", default="Llama-3.2 decoder layer — device-program RAW dependency graph")
    ap.add_argument("--resolve-inplace", action="store_true", help="treat in-place ops as RMW (reconnect through them)")
    ap.add_argument("--add-edge", action="append", default=[], help="inject a known semantic edge P:C (repeatable)")
    ap.add_argument("--range", default=None, help="draw only ops [START:END) for legibility on huge graphs")
    a = ap.parse_args()
    add_edges = [tuple(int(x) for x in e.split(":")) for e in a.add_edge]
    node_range = tuple(int(x) for x in a.range.split(":")) if a.range else None
    build(a.capture_json, a.out, a.title, a.resolve_inplace, add_edges, node_range)


if __name__ == "__main__":
    raise SystemExit(main())

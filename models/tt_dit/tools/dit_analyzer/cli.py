# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Command line entry point.

    ditcheck analyze example:sd35_block_double_gather --states
    ditcheck analyze captured_graph.json --top 5 --json findings.json
    ditcheck dump example:sd35_block > sd35_block.json
    ditcheck examples
    ditcheck ops
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import analyze_graph, load_graph
from .examples import DESCRIPTIONS, EXAMPLES
from .report import render_report, render_states
from .semantics import describe_registry


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("graph", help="path to a graph JSON dump, or 'example:<name>'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ditcheck", description="Collective-redundancy static analyzer for DiT graphs")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="run the full analysis and print ranked findings")
    _add_common(a)
    a.add_argument("--top", type=int, default=10, help="how many findings to print (default 10)")
    a.add_argument("--states", action="store_true", help="also print per-device state tables at every collective")
    a.add_argument("--no-proof", action="store_true", help="omit proof objects from the text output")
    a.add_argument(
        "--link-bw",
        type=float,
        default=None,
        metavar="GBPS",
        help="effective link bandwidth for a rough latency estimate",
    )
    a.add_argument("--json", dest="json_out", default=None, metavar="PATH", help="write findings + proofs as JSON")
    a.add_argument(
        "--fail-on",
        default=None,
        choices=["provable", "likely", "suspicious"],
        help="exit 1 if a finding at or above this confidence exists (for CI)",
    )

    s = sub.add_parser("states", help="print per-device tensor state around collectives only")
    _add_common(s)
    s.add_argument("--node", default=None, help="only show collectives whose id/label contains this string")

    d = sub.add_parser("dump", help="print the graph as JSON")
    _add_common(d)

    sub.add_parser("examples", help="list built-in example graphs")
    sub.add_parser("ops", help="list registered op semantics")
    return p


def _confidence_at_least(level: str, threshold: str) -> bool:
    order = {"provable": 0, "likely": 1, "suspicious": 2}
    return order.get(level, 3) <= order[threshold]


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "examples":
        for name in sorted(EXAMPLES):
            print("  %-28s %s" % (name, DESCRIPTIONS.get(name, "")))
        print("\nuse as:  ditcheck analyze example:<name>")
        return 0

    if args.cmd == "ops":
        print(describe_registry())
        return 0

    graph = load_graph(args.graph)

    if args.cmd == "dump":
        print(graph.to_json())
        return 0

    report = analyze_graph(graph)

    if args.cmd == "states":
        print(render_states(report, node_filter=args.node))
        return 0

    print(
        render_report(
            report,
            top=args.top,
            states=args.states,
            link_bw_gbs=args.link_bw,
            proof=not args.no_proof,
        )
    )

    if args.json_out:
        payload = {
            "graph": graph.name,
            "mesh": list(graph.mesh.shape),
            "findings": [f.to_dict() for f in report.findings],
            "diagnostics": [{"node": d.node, "code": d.code, "message": d.message} for d in report.diagnostics],
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("\nwrote %s" % args.json_out)

    if args.fail_on:
        hits = [f for f in report.findings if _confidence_at_least(f.confidence, args.fail_on)]
        if hits:
            print("\n%d finding(s) at confidence >= %s" % (len(hits), args.fail_on))
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Command line entry point.

    ditcheck dryrun ltx_block --preset bh_4x8 --out ltx.graph.json --analyze
    ditcheck analyze example:sd35_block_double_gather --states
    ditcheck analyze captured_graph.json --top 5 --json findings.json
    ditcheck dump example:sd35_block > sd35_block.json
    ditcheck examples
    ditcheck ops [--missing GRAPH | --check GRAPH]
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

    r = sub.add_parser("dryrun", help="run a model target against the metadata-only ttnn and emit its graph")
    r.add_argument("target", nargs="?", default=None, help="target name; omit to list the available ones")
    r.add_argument("--preset", default=None, help="mesh configuration (see the target listing)")
    r.add_argument("--out", default=None, metavar="PATH", help="write the graph JSON here")
    r.add_argument("--analyze", action="store_true", help="analyze the graph straight away")
    r.add_argument(
        "--check-oracle",
        action="store_true",
        help="diff collectives and findings against the preset's hand-written examples/ graph (drift test)",
    )
    r.add_argument("--top", type=int, default=10, help="findings to print with --analyze")
    r.add_argument(
        "--fail-on",
        default=None,
        choices=["provable", "likely", "suspicious"],
        help="with --analyze, exit 1 if a finding at or above this confidence exists",
    )

    sub.add_parser("examples", help="list built-in example graphs")
    o = sub.add_parser("ops", help="list registered op semantics, or a graph's coverage gaps")
    o.add_argument("--missing", default=None, metavar="GRAPH", help="list the ops a graph uses that have no spec")
    o.add_argument("--check", default=None, metavar="GRAPH", help="exit 1 if a graph contains an uncovered op")
    o.add_argument(
        "--stub",
        action="store_true",
        help="with --missing: print a copy-paste registration skeleton (shim rule + analyzer OpSpec) per op",
    )
    return p


def _confidence_at_least(level: str, threshold: str) -> bool:
    order = {"provable": 0, "likely": 1, "suspicious": 2}
    return order.get(level, 3) <= order[threshold]


def _dryrun(args) -> int:
    """Build a target's graph from source. Shadows `ttnn`, so run it in its own process."""
    from .dryrun import provenance
    from .dryrun.targets import TARGETS, build, describe

    if not args.target:
        print("dry-run targets:\n%s" % describe())
        print("\nuse as:  ditcheck dryrun <target> --preset <preset>")
        return 0
    if args.target not in TARGETS:
        raise SystemExit("unknown target '%s' (have: %s)" % (args.target, ", ".join(sorted(TARGETS))))

    graph, preset = build(args.target, args.preset)
    info = provenance()
    print(
        "dry run %s/%s: %d nodes, %d symbols, mesh %s %s"
        % (args.target, preset.name, len(graph.nodes), len(graph.symbols), tuple(graph.mesh.shape), graph.mesh.topology)
    )
    print("torch: %s" % info["torch"])
    for note in info["substitutions"]:
        print("  substituted %s" % note)
    if graph.meta.get("checkpoint"):
        flags = graph.meta.get("checkpoint_flags", {})
        print(
            "checkpoint: %s -> %s" % (graph.meta["checkpoint"], ", ".join("%s=%s" % (k, v) for k, v in flags.items()))
        )
    if info["unregistered_ops"]:
        print(
            "\nunregistered ops (%d distinct) -- `ditcheck ops --missing` on the graph for detail:"
            % len(info["unregistered_ops"])
        )
        for entry in info["unregistered_ops"]:
            print("  %-58s x%-4d %s" % (entry["call"], entry["count"], entry["loc"] or ""))
    else:
        print("unregistered ops: none")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(graph.to_json())
        print("\nwrote %s" % args.out)

    if args.check_oracle:
        from .dryrun.verify import compare_with_oracle, render

        if not preset.oracle:
            raise SystemExit("preset '%s' has no oracle to check against" % preset.name)
        diff = compare_with_oracle(graph, preset.oracle, unregistered={e["call"]: e for e in info["unregistered_ops"]})
        print()
        print(render(diff))
        if diff["failures"]:
            return 1

    if not args.analyze:
        return 0

    report = analyze_graph(graph)
    print()
    print(render_report(report, top=args.top))
    if args.fail_on:
        hits = [f for f in report.findings if _confidence_at_least(f.confidence, args.fail_on)]
        if hits:
            print("\n%d finding(s) at confidence >= %s" % (len(hits), args.fail_on))
            return 1
    return 0


def _registration_stub(call: str, arity: int) -> str:
    """A copy-paste registration skeleton for one unregistered op.

    Emits both halves a new op needs -- the shim shape rule in ``dryrun/ops.py``
    (build the IR node) and the analyzer ``OpSpec`` in ``semantics.py`` (reason
    through it) -- with ``shim`` / ``apply`` / ``demand`` left as ``TODO``. Until
    the phase-8 one-registration merge lands, these live in two files, so the stub
    names both (roadmap "op registration", ergonomic point 3).
    """
    canon = call.split(".")[-1]
    n = max(1, int(arity))
    params = ", ".join(["x"] + ["a%d" % i for i in range(1, n)])
    ins = ", ".join(["x"] + ["a%d" % i for i in range(1, n)])
    return (
        "  # --- shim shape rule -> dryrun/ops.py (build the IR node) ------------\n"
        "  def %s(%s, **k):\n"
        '      # TODO(shim): return recorder.emit("%s", [%s], <out_logical>, <out_dist>, base="%s")\n'
        '      raise NotImplementedError("shim shape rule for %s")\n'
        "  # register in the matching table: TENSOR_OPS / EXPERIMENTAL_OPS / TRANSFORMER_OPS\n"
        '  #     "%s": %s,\n'
        "\n"
        "  # --- analyzer semantics -> semantics.py (reason through it) ----------\n"
        "  def _%s_apply(c):    # forward availability\n"
        "      ...  # TODO(apply): c.define(0, dist, regions, value_id, tainted)\n"
        "  def _%s_demand(c):   # backward necessity\n"
        "      ...  # TODO(demand): c.need(c.node.inputs[i], dev, region)\n"
        '  register(OpSpec("%s", COMPUTE, _%s_apply, _%s_demand), aliases=("%s",))\n'
        % (canon, params, canon, ins, canon, call, canon, canon, canon, canon, canon, canon, canon, call)
    )


def _ops_coverage(graph, fail: bool, stub: bool = False) -> int:
    """Which ops in this graph have no semantics, and what that costs the analysis."""
    from .ir import UNREGISTERED_OP

    missing: dict = {}
    for node in graph.nodes:
        if node.op != UNREGISTERED_OP:
            continue
        call = node.attrs.get("call", node.op)
        entry = missing.setdefault(call, {"count": 0, "locs": [], "arity": node.attrs.get("arity", 0)})
        entry["count"] += 1
        if node.call_site and node.call_site not in entry["locs"]:
            entry["locs"].append(node.call_site)

    if not missing:
        print("op coverage: every op in '%s' has registered semantics." % graph.name)
        return 0

    report = analyze_graph(graph)
    print("op coverage: %d call(s) in '%s' have no semantics.\n" % (len(missing), graph.name))
    for call, entry in sorted(missing.items(), key=lambda kv: -kv[1]["count"]):
        print("  %s  x%d  (%d tensor args)" % (call, entry["count"], entry["arity"]))
        for where in entry["locs"][:3]:
            print("      at %s" % where)
    blocked = [w for w in report.withheld if any(op in missing for op in w.ops)]
    print(
        "\n%d finding(s) are withheld because their proof passes through these ops." % len(blocked)
        + ("" if blocked else " Nothing is currently blocked on them.")
    )
    if stub:
        print("\nregistration skeletons (fill the TODOs, then re-run):\n")
        for call, entry in sorted(missing.items(), key=lambda kv: -kv[1]["count"]):
            print("%s:" % call)
            print(_registration_stub(call, entry["arity"]))
    else:
        print("\nre-run with --stub for a copy-paste registration skeleton per op.")
    return 1 if fail else 0


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "examples":
        for name in sorted(EXAMPLES):
            print("  %-28s %s" % (name, DESCRIPTIONS.get(name, "")))
        print("\nuse as:  ditcheck analyze example:<name>")
        return 0

    if args.cmd == "dryrun":
        return _dryrun(args)

    if args.cmd == "ops":
        if args.missing or args.check:
            return _ops_coverage(
                load_graph(args.missing or args.check), fail=bool(args.check), stub=getattr(args, "stub", False)
            )
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

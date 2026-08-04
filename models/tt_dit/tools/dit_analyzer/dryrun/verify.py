# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Diffing a dry run against the hand-written graph for the same block.

``examples/`` states by hand what the dry run derives from source. When the two
disagree, either the model changed and the example is stale, or a shim shape rule
drifted -- and the second is the roadmap's top risk (blocker 36, and the shim
bugs the spike found), because a wrong shape does not perturb the graph, it
invents redundancy. So the examples stay, as oracles rather than as scaffolding.

Four criteria, all of them assertions:

1. the forward completed, with no op left unregistered;
2. the analyzer raised no diagnostics (its own shape/axis consistency checks);
3. the collectives match the oracle as a multiset of (op, mesh axis, gathered
   extent, logical shape);
4. the findings match by (rule, confidence, bytes per call, calls).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from ..ir import Graph
from ..semantics import lookup


def collectives(graph: Graph) -> List[Tuple[Any, ...]]:
    """(op, mesh axis, gathered extent, non-unit logical extents).

    Rank-normalised: the hand model writes ``[1, N, D]`` where the real code uses
    ``[1, B, N, D]``, so compare the extents that carry information.
    """
    out = []
    for n in graph.nodes:
        if not lookup(n.op).is_collective:
            continue
        shape = graph.symbol(n.outputs[0]).shape
        dim = n.attrs.get("dim")
        out.append((n.op, n.mesh_axis, shape[dim] if dim is not None else None, tuple(d for d in shape if d != 1)))
    return out


def findings_signature(report) -> List[Tuple[Any, ...]]:
    return sorted((f.rule, f.confidence, f.bytes_per_call, f.calls) for f in report.findings)


def compare_with_oracle(graph: Graph, oracle_name: str, unregistered: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run both analyses and return a diff plus a list of failed criteria."""
    from .. import analyze_graph
    from ..examples import load

    oracle = load(oracle_name)
    dry_collectives, oracle_collectives = collectives(graph), collectives(oracle)
    counted_dry, counted_oracle = Counter(dry_collectives), Counter(oracle_collectives)
    only_dry = sorted((counted_dry - counted_oracle).elements())
    only_oracle = sorted((counted_oracle - counted_dry).elements())

    report, oracle_report = analyze_graph(graph), analyze_graph(oracle)

    failures: List[str] = []
    if not graph.nodes:
        failures.append("1: the dry run produced no nodes")
    if unregistered:
        failures.append("1: unregistered ops: %s" % ", ".join(sorted(unregistered)))
    if report.diagnostics:
        failures.append("2: analyzer diagnostics: %s" % ", ".join(sorted({d.code for d in report.diagnostics})))
    if only_dry or only_oracle:
        failures.append("3: collectives differ from the oracle (+%d / -%d)" % (len(only_dry), len(only_oracle)))
    if findings_signature(report) != findings_signature(oracle_report):
        failures.append(
            "4: findings differ from the oracle\n     dry run: %s\n     oracle:  %s"
            % (findings_signature(report), findings_signature(oracle_report))
        )
    if report.withheld:
        failures.append(
            "4: %d finding(s) withheld pending op registration: %s" % (len(report.withheld), report.missing_ops)
        )

    return {
        "oracle": oracle_name,
        "counts": (len(dry_collectives), len(oracle_collectives)),
        "only_dry": only_dry,
        "only_oracle": only_oracle,
        "report": report,
        "oracle_report": oracle_report,
        "failures": failures,
    }


def render(diff: Dict[str, Any]) -> str:
    """The oracle diff as text, whether it passed or not."""
    dry, ref = diff["counts"]
    lines = ["oracle: example:%s" % diff["oracle"], "collectives: dry run %d, oracle %d" % (dry, ref)]
    if diff["only_dry"] or diff["only_oracle"]:
        lines.append("  only in the dry run (%d):" % len(diff["only_dry"]))
        lines += ["    + %s" % (x,) for x in diff["only_dry"][:12]]
        lines.append("  only in the oracle (%d):" % len(diff["only_oracle"]))
        lines += ["    - %s" % (x,) for x in diff["only_oracle"][:12]]
    else:
        lines.append("  identical multiset of (op, mesh axis, gathered extent, shape)")

    report = diff["report"]
    rules = Counter(f.rule for f in report.findings)
    lines.append("findings: %s" % (dict(rules) or "none"))
    for f in report.findings:
        lines.append("  [%s/%s] %s" % (f.severity, f.confidence, f.title))
        for i, frame in enumerate(f.source_chain):
            lines.append("        %s%s" % ("" if i == 0 else "via ", frame))
    if report.diagnostics:
        lines.append("diagnostics: %s" % dict(Counter(d.code for d in report.diagnostics)))

    if diff["failures"]:
        lines.append("")
        lines.append("FAIL (%d criteria):" % len(diff["failures"]))
        lines += ["  - %s" % f for f in diff["failures"]]
    else:
        lines.append("")
        lines.append("PASS: the dry run matches the hand-written oracle on all four criteria")
    return "\n".join(lines)

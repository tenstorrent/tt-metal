# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Text rendering: per-device state tables, ranked findings, proof traces."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .ir import Graph
from .rules import Finding, Report
from .semantics import lookup

RULE_ORDER = [
    "dead_collective",
    "replicated_stage",
    "unused_gather",
    "duplicate_gather",
    "overwide_gather",
    "participant_shrink",
    "invariant_collective",
    "mergeable_collectives",
]


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % int(n)
        n /= 1024.0
    return "%.1f TiB" % n


def _us(bytes_moved: float, link_bw_gbs: float) -> str:
    return "%.1f us" % (bytes_moved / (link_bw_gbs * 1e9) * 1e6)


def render_trust(graph: Graph) -> str:
    """One line stating how far the graph's shapes can be trusted.

    A standing requirement (``DitStaticAnalyzerPlan.md`` §"Honesty rules"): the
    report must say, every time, when a finding rests on shapes the metadata-only
    ttnn shim *computed* rather than on real ttnn -- "the shim believes" -- so a
    shim-derived finding is never mistaken for a device-verified one.
    """
    prov = getattr(graph, "provenance", "unknown")
    if prov == "dry-run":
        return (
            "trust: THE SHIM BELIEVES -- shapes were computed by the metadata-only ttnn shim, not "
            "verified against real ttnn. Corroborate with `--check-oracle` and with per-op "
            "conformance on a device (conform.py / phase 11) before acting on a finding."
        )
    if prov == "hand-written":
        return (
            "trust: hand-transcribed graph -- findings rest on this transcription's fidelity to the "
            "model source, not on a device."
        )
    if prov == "captured":
        return "trust: captured from a device trace -- shapes are ground truth (entry placements may be declared)."
    return "trust: provenance unrecorded -- treat every finding as unverified."


def render_header(graph: Graph) -> str:
    mesh = graph.mesh
    lines = [
        "=" * 100,
        "graph: %s" % graph.name,
        "mesh:  %dx%d %s  axes=(%s)  topology=%s"
        % (mesh.shape[0], mesh.shape[1], mesh.arch, ", ".join(mesh.axis_names), mesh.topology),
        "nodes: %d   symbols: %d   denoise steps: %d" % (len(graph.nodes), len(graph.symbols), graph.steps),
    ]
    for k, v in sorted(graph.meta.items()):
        lines.append("%-6s %s" % (k + ":", v))
    n_seg = len(graph.segments())
    if n_seg > 1:
        lines.append("segments: %d device segments split at readbacks (phase 10: stage boundaries)" % n_seg)
    lines.append(render_trust(graph))
    lines.append("=" * 100)
    return "\n".join(lines)


def render_summary(report: Report, link_bw_gbs: Optional[float] = None) -> str:
    graph = report.graph
    n_coll = len({v.node.id for v in report.views})
    flagged = {n for f in report.findings for n in f.nodes if _is_collective(graph, n)}
    per_rule: Dict[str, int] = {}
    for f in report.findings:
        per_rule[f.rule] = per_rule.get(f.rule, 0) + 1

    per_forward = sum(f.bytes_per_forward for f in report.findings if f.scope == "forward")
    per_gen = sum(f.bytes_per_forward for f in report.findings if f.scope == "generation")
    lines = [
        "collectives analyzed: %d distinct nodes (%d node/group pairs)" % (n_coll, len(report.views)),
        "flagged:              %d   necessary: %d" % (len(flagged), n_coll - len(flagged)),
        "findings by rule:     %s"
        % (", ".join("%s=%d" % (r, per_rule[r]) for r in RULE_ORDER if r in per_rule) or "none"),
        "recoverable traffic:  %s per forward pass, aggregated over participants" % human_bytes(per_forward),
    ]
    if report.withheld:
        lines.append(
            "withheld:             %d finding(s) need op coverage first: %s"
            % (len(report.withheld), ", ".join(report.missing_ops[:4]))
        )
    if per_gen:
        lines.append(
            "                      + %s once per generation (step-invariant collectives)" % human_bytes(per_gen)
        )
    if link_bw_gbs:
        worst = max((f.bytes_per_device for f in report.findings), default=0)
        lines.append(
            "                      largest single finding: ~%s/device at %.1f GB/s"
            % (_us(worst, link_bw_gbs), link_bw_gbs)
        )
    if report.diagnostics:
        codes: Dict[str, int] = {}
        for d in report.diagnostics:
            codes[d.code] = codes.get(d.code, 0) + 1
        lines.append("diagnostics:          %s" % ", ".join("%s=%d" % kv for kv in sorted(codes.items())))
    return "\n".join(lines)


def _is_collective(graph: Graph, node_id: str) -> bool:
    try:
        return lookup(graph.node(node_id).op).is_collective
    except KeyError:
        return False


def _render_source(f: Finding, indent: str = "    ") -> List[str]:
    """The model call site, with the library frames it went through underneath."""
    chain = f.source_chain
    if not chain:
        return []
    lines = ["%ssource: %s" % (indent, chain[0])]
    for frame in chain[1:]:
        lines.append("%s        via %s" % (indent, frame))
    return lines


def render_finding(
    index: int, f: Finding, graph: Graph, link_bw_gbs: Optional[float] = None, proof: bool = True
) -> str:
    lines = []
    lines.append("-" * 100)
    lines.append("#%d  [%s/%s]  %s" % (index, f.severity.upper(), f.confidence, f.rule))
    lines.append("    %s" % f.title)
    lines += _render_source(f)
    lines.append("    nodes:  %s" % ", ".join(f.nodes))
    for r in f.reason:
        lines.append("    why:    %s" % r)
    cost = "    cost:   %s per call x %d calls = %s of link traffic per %s (all participants)" % (
        human_bytes(f.bytes_per_call),
        f.calls,
        human_bytes(f.bytes_per_forward),
        f.scope,
    )
    if f.steps > 1 and f.scope == "forward":
        cost += "; x%d steps = %s per generation" % (f.steps, human_bytes(f.bytes_per_forward * f.steps))
    lines.append(cost)
    if link_bw_gbs:
        lines.append(
            "    est:    ~%s per %s per device (%s / device at %.1f GB/s; first-order, ignores overlap)"
            % (_us(f.bytes_per_device, link_bw_gbs), f.scope, human_bytes(f.bytes_per_device), link_bw_gbs)
        )
    lines.append("    fix:    %s" % f.suggestion)
    if proof:
        lines.append("    proof:")
        for k in (
            "tensor",
            "shape",
            "dtype",
            "mesh_axis_name",
            "participants",
            "layout_before",
            "layout_after",
            "value_id",
            "equivalent_symbol",
            "equivalent_producer",
            "shares_with",
            "devices_needing_remote_data",
            "devices_whose_data_is_read",
            "wasted_fraction",
            "invalidation_check",
            "semantics_complete",
            "conclusion",
        ):
            if k in f.proof:
                lines.append("      %-28s %s" % (k + ":", f.proof[k]))
        for k in ("available_before", "materialised_after", "needed_downstream", "unneeded", "overlap"):
            if k not in f.proof:
                continue
            lines.append("      %s:" % k)
            for dev, region in sorted(f.proof[k].items(), key=lambda kv: int(kv[0])):
                lines.append("        device %-3s %s" % (dev, region))
    return "\n".join(lines)


def rollup_findings(findings: List[Finding]) -> List[Tuple[Finding, int, int]]:
    """Collapse a finding repeated across layers into one entry.

    A 50-layer stack reports the same redundancy 50 times -- same rule, same call site, same
    per-call cost, different node ids. Listing them separately buries every *other* finding
    below a wall of duplicates (at production depth: 321 findings, 211 of them one rule), and
    a top-N cut then shows one repeat 8 times instead of 8 distinct problems.

    Groups on what makes two findings the same problem: rule, source chain, per-call bytes and
    verdict. Returns (representative, occurrences, total bytes per forward), ranked by total.
    """
    groups: Dict[Tuple, List[Finding]] = {}
    order: List[Tuple] = []
    for f in findings:
        key = (f.rule, tuple(f.source_chain), f.bytes_per_call, f.severity, f.confidence, f.scope)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f)
    rolled = [(groups[k][0], len(groups[k]), sum(x.bytes_per_forward for x in groups[k])) for k in order]
    rolled.sort(key=lambda t: -t[2])
    return rolled


def render_findings(
    report: Report,
    top: int = 10,
    link_bw_gbs: Optional[float] = None,
    proof: bool = True,
    rollup: bool = True,
) -> str:
    if not report.findings:
        return "no redundancy findings: every collective is needed by a downstream consumer."
    if not rollup:
        out = ["ranked findings (top %d of %d)" % (min(top, len(report.findings)), len(report.findings))]
        for i, f in enumerate(report.findings[:top], start=1):
            out.append(render_finding(i, f, report.graph, link_bw_gbs, proof))
        out.append("-" * 100)
        return "\n".join(out)

    rolled = rollup_findings(report.findings)
    header = "ranked findings (top %d of %d distinct; %d total across layers)" % (
        min(top, len(rolled)),
        len(rolled),
        len(report.findings),
    )
    out = [header]
    for i, (f, n, total) in enumerate(rolled[:top], start=1):
        body = render_finding(i, f, report.graph, link_bw_gbs, proof)
        if n > 1:
            body = body.replace(
                "#%d  [%s/%s]  %s" % (i, f.severity.upper(), f.confidence, f.rule),
                "#%d  [%s/%s]  %s   x%d occurrences (same call site, one per layer)"
                % (i, f.severity.upper(), f.confidence, f.rule, n),
            )
            body += "\n    rolled:  %s x %d occurrences = %s per forward across the stack" % (
                human_bytes(f.bytes_per_forward),
                n,
                human_bytes(total),
            )
        out.append(body)
    out.append("-" * 100)
    return "\n".join(out)


def render_states(report: Report, node_filter: Optional[str] = None) -> str:
    """Per-device before/after table at every collective (Phase 2 deliverable)."""
    graph = report.graph
    out: List[str] = []
    seen = set()
    for v in report.views:
        if v.node.id in seen:
            continue
        if node_filter and node_filter not in v.node.id and node_filter not in (v.node.label or ""):
            continue
        seen.add(v.node.id)
        node = v.node
        xs, ys = v.in_sym, v.out_sym
        out.append("=" * 100)
        head = "%s  %s  dim=%s  %s=%s" % (
            node.display,
            node.op,
            node.attrs.get("dim", "-"),
            graph.mesh.axis_names[node.mesh_axis],
            list(v.group),
        )
        if node.fused_in:
            head += "  (fused in %s)" % node.fused_in
        out.append(head)
        out.append(
            "  tensor %s %s %s   calls/forward=%d%s"
            % (xs.id, list(xs.shape), xs.dtype, node.calls, "  src=" + node.loc if node.loc else "")
        )
        vm = graph.mesh_of(node)  # the collective's own mesh (blocker 22)
        out.append("  layout %s  ->  %s" % (v.in_state.dist.describe(vm), v.out_state.dist.describe(vm)))
        out.append("  %-6s %-34s %-34s %s" % ("device", "available before", "materialised after", "needed downstream"))
        for d in v.group:
            out.append(
                "  %-6d %-34s %-34s %s"
                % (
                    d,
                    v.local[d].describe(xs.shape),
                    v.out_state.regions[d].describe(ys.shape),
                    v.needed[d].describe(ys.shape),
                )
            )
        verdict = "NECESSARY"
        for f in report.findings:
            if node.id in f.nodes:
                verdict = "%s (%s, %s)" % (f.rule.upper(), f.severity, f.confidence)
                break
        out.append("  verdict: %s   nominal traffic: %s/call" % (verdict, human_bytes(v.moved_bytes())))
    return "\n".join(out) if out else "no collectives in this graph."


def render_hints(report: Report, top: int = 5) -> str:
    """Opportunities that are not redundancy claims (kept out of the ranking)."""
    if not report.hints:
        return ""
    out = ["hints (%d) -- opportunities, not redundancy; no bytes are provably wasted" % len(report.hints)]
    for i, f in enumerate(report.hints[:top], start=1):
        out.append("  h%d [%s/%s] %s" % (i, f.severity, f.confidence, f.title))
        out += _render_source(f, indent="      ")
        out.append("      %s" % f.suggestion)
        out.append("      applies %d times per forward" % f.calls)
    if len(report.hints) > top:
        out.append("  ... %d more" % (len(report.hints) - top))
    return "\n".join(out)


def render_withheld(report: Report, top: int = 10) -> str:
    """What the analysis refused to claim, and what would unlock it.

    A finding here is not a weaker finding: the shim invented the output metadata
    of some op along its proof, so the claim is unsupported rather than uncertain.
    Reporting the *registration* instead of the guess is the point.
    """
    if not report.withheld:
        return ""
    out = [
        "withheld (%d) -- findings blocked on op coverage, not reported and not downgraded" % len(report.withheld),
        "  register these ops to unlock them: %s" % ", ".join(report.missing_ops),
    ]
    for i, w in enumerate(report.withheld[:top], start=1):
        f = w.finding
        out.append("  w%d [%s] %s" % (i, f.rule, f.title))
        out += _render_source(f, indent="      ")
        out.append("      blocked by: %s" % ", ".join(w.ops))
    if len(report.withheld) > top:
        out.append("  ... %d more" % (len(report.withheld) - top))
    return "\n".join(out)


def render_diagnostics(report: Report) -> str:
    if not report.diagnostics:
        return "no diagnostics: all ops had registered semantics and consistent layouts."
    out = ["diagnostics (semantics gaps and suspicious layouts -- read these before trusting findings)"]
    for d in report.diagnostics:
        node = report.graph.node(d.node) if any(n.id == d.node for n in report.graph.nodes) else None
        label = node.display if node else d.node
        loc = ("  [%s]" % node.loc) if node and node.loc else ""
        out.append("  %-22s %-30s %s%s" % (d.code, label, d.message, loc))
    return "\n".join(out)


def render_report(
    report: Report,
    top: int = 10,
    states: bool = False,
    link_bw_gbs: Optional[float] = None,
    proof: bool = True,
    rollup: bool = True,
) -> str:
    parts = [render_header(report.graph), render_summary(report, link_bw_gbs), ""]
    if states:
        parts += [render_states(report), ""]
    parts += [render_findings(report, top=top, link_bw_gbs=link_bw_gbs, proof=proof, rollup=rollup), ""]
    hints = render_hints(report)
    if hints:
        parts += [hints, ""]
    withheld = render_withheld(report)
    if withheld:
        parts += [withheld, ""]
    parts.append(render_diagnostics(report))
    return "\n".join(parts)

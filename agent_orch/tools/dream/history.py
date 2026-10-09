"""history.md: the index of every attempt that workers read before proposing anything."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from .campaign import Campaign
from .scoring import case_metrics
from .tree import load_round, read_node_file, recorded_rounds


def reflection_next(text: str | None) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("## what a child"):
            return next((l.strip("- ").strip() for l in lines[i + 1 :] if l.strip() and not l.startswith("#")), "")
    return ""


def collect(c: Campaign) -> tuple[dict | None, list[dict]]:
    base_p = c.ledger / "baseline.json"
    baseline = json.loads(base_p.read_text()) if base_p.exists() else None
    rounds = []
    for r in recorded_rounds(c):
        rr = load_round(c, r)
        step_of = {}
        for d in rr.decisions:
            if d.get("type") == "decision":
                for it in d.get("batch", []):
                    step_of[it["node"]] = d["step"]
        nodes = []
        for n in rr.nodes:
            score = json.loads(read_node_file(c, n.node_id, "eval/score.json") or "{}")
            nodes.append(
                {
                    "obs": n,
                    "cases": case_metrics(score),
                    "next": reflection_next(read_node_file(c, n.node_id, "reflection.md")),
                    "step": step_of.get(n.node_id, n.attempt),
                }
            )
        lost = [d["node"] for d in rr.decisions if d.get("type") == "lost"]
        audited = {d["node"] for d in rr.decisions if d.get("type") == "audit"}
        for x in nodes:
            x["flagged"] = x["obs"].node_id in audited
        rounds.append({"rr": rr, "nodes": nodes, "closed": rr.closed(), "lost": lost})
    return baseline, rounds


def _cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_md(c: Campaign, path: Path) -> None:
    baseline, rounds = collect(c)
    unit = c.cfg["eval"].get("unit", "")
    allnodes = [x for r in rounds for x in r["nodes"]]
    valid = [x for x in allnodes if x["obs"].ok]
    best = max(valid, key=lambda x: x["obs"].score) if valid else None
    bcases = (
        (baseline.get("cases") or {k: {"value": v["us_chip_mean"]} for k, v in baseline.get("shapes", {}).items()})
        if baseline
        else {}
    )
    case_ids = list(bcases)
    L = [f"# {c.name}: discovery history", ""]
    L.append(
        f"Generated {datetime.datetime.now().isoformat(timespec='minutes')} from `{c.ref_prefix}/n/*` and the ledger."
    )
    if c.cfg.get("description"):
        L.append(f"Goal: {c.cfg['description']} ({c.cfg['eval']['direction']} the measured value).")
    L.append("")
    if baseline:
        L.append(
            f"**Baseline** ({unit or 'value'}; noise ±{baseline['noise_pct']}%): "
            + ", ".join(f"{cid} {bcases[cid]['value']:.4g}" for cid in case_ids)
        )
    if best:
        b = best["obs"]
        L.append(f"**Best valid:** `{b.node_id}` score {b.score:.4f} ({_cell(b.mechanism)})")
    lost = sum(len(r["lost"]) for r in rounds)
    L.append(f"**Attempts:** {len(allnodes)} committed ({len(valid)} valid) over {len(rounds)} round(s); {lost} lost.")
    L.append("Score = geomean over cases of the improvement ratio vs baseline; 1.0 = baseline, higher is better.")
    L.append("")
    if valid:
        L += ["## Leaderboard (top 10 valid)", ""]
        L.append("| node | score | mechanism | " + " | ".join(f"{cid} {unit}" for cid in case_ids) + " |")
        L.append("|---|---|---|" + "---|" * len(case_ids))
        for x in sorted(valid, key=lambda x: -x["obs"].score)[:10]:
            vals = [f"{x['cases'].get(cid, {}).get('value', float('nan')):.4g}" for cid in case_ids]
            L.append(
                f"| `{x['obs'].node_id}` | {x['obs'].score:.4f} | {_cell(x['obs'].mechanism)} | "
                + " | ".join(vals)
                + " |"
            )
        L.append("")
    for r in rounds:
        rr, m = r["rr"], r["rr"].manifest
        L += [f"## Round r{rr.round:02d}", ""]
        if m:
            L += [
                f"Policy `{m.get('policy')}` (beta {m.get('beta')}), W={m.get('W')}, R={m.get('R')}, "
                f"root `{m.get('round_root')}`. Plan: {m.get('reason')}",
                "",
            ]
        by_branch: dict[int, list] = {}
        for x in r["nodes"]:
            by_branch.setdefault(x["obs"].branch, []).append(x)
        for b in sorted(by_branch):
            closed = r["closed"].get(b)
            L += [f"### Branch b{b:02d}" + (f" (closed: {closed})" if closed else ""), ""]
            L.append("| node | parent | mechanism | tags | score | Δ parent | fail_class | next |")
            L.append("|---|---|---|---|---|---|---|---|")
            for x in by_branch[b]:
                n = x["obs"]
                d = f"{n.delta_vs_parent:+.4f}" if n.delta_vs_parent is not None else "-"
                L.append(
                    f"| `{n.node_id}` | {n.parent} | {_cell(n.mechanism)}{' ⚑ isolation' if x['flagged'] else ''} | "
                    f"{', '.join(n.tags)} | "
                    f"{n.score:.4f} | {d} | {n.fail_class} | {_cell(x['next'])} |"
                )
            L.append("")
        if r["lost"]:
            L += [f"Lost (worker never committed): {', '.join(r['lost'])}", ""]
    P = c.attempts_rel("<id>")
    N = f"{c.ref_prefix}/n/<id>"
    L += [
        "## Reading a node in full",
        "",
        "```bash",
        f"git show {N}:{P}/proposal.md",
        f"git show {N}:{P}/reflection.md",
        f"git show {N}:{P}/eval/summary.md",
        f"git diff {N}~1 {N} -- . ':!agent_orch'",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")

"""The standardized campaign report: collect one JSON document from git + the ledger, render it into
agent_orch/report/template.html, write $DREAM_HOME/<c>/report/{data.json,index.html}.

The page is the same for every campaign; only the data changes. `dream report` copies it to the
caller's machine, and the dream skill publishes it as a live artifact.
"""

from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

import yaml

from .campaign import ORCH_DIR, Campaign, round_root_mode
from .gitops import lineage, read_jsonl
from .scoring import case_metrics
from .tree import load_round, read_node_file, recorded_rounds

TEMPLATE = ORCH_DIR / "report" / "template.html"


def _status(c: Campaign) -> dict:
    p = c.home / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"state": "not started", "message": ""}


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return text.strip().splitlines()[0] if text.strip() else ""


def collect(c: Campaign) -> dict:
    base_p = c.ledger / "baseline.json"
    baseline = json.loads(base_p.read_text()) if base_p.exists() else None
    rounds, allnodes = [], []
    for r in recorded_rounds(c):
        rr = load_round(c, r)
        rdir = c.ledger / "rounds" / f"r{r:02d}"
        step_of = {}
        for d in rr.decisions:
            if d.get("type") == "decision":
                for it in d.get("batch", []):
                    step_of[it["node"]] = d["step"]
        flags = {}
        for d in rr.decisions:
            if d.get("type") == "audit":
                flags.setdefault(d["node"], []).extend(d.get("flags", []))
        nodes = []
        for n in rr.nodes:
            x = {
                "flags": flags.get(n.node_id, []),
                "id": n.node_id,
                "b": n.branch,
                "a": n.attempt,
                "step": step_of.get(n.node_id, n.attempt),
                "valid": n.ok,
                "fail": n.fail_class,
                "score": n.score,
                "mech": n.mechanism,
                "parent": n.parent,
            }
            nodes.append(x)
            allnodes.append(x)
        m = rr.manifest
        rounds.append(
            {
                "round": r,
                "policy": m.get("policy"),
                "beta": m.get("beta"),
                "W": m.get("W"),
                "R": m.get("R"),
                "root": m.get("round_root", ""),
                "reason": m.get("reason", ""),
                "nodes": nodes,
                "closed": {str(k): v for k, v in rr.closed().items()},
                "lost": [d["node"] for d in rr.decisions if d.get("type") == "lost"],
                "steps": rr.steps_done(),
                "root_score": rr.root_score,
                "summary": (rdir / "summary.md").read_text() if (rdir / "summary.md").exists() else "",
                "finished": (rdir / "summary.md").exists(),
            }
        )

    best = None
    valid = [x for x in allnodes if x["valid"]]
    if valid:
        b = max(valid, key=lambda x: x["score"])
        score = json.loads(read_node_file(c, b["id"], "eval/score.json") or "{}")
        diffstat = c.git(
            "diff", "--stat", c.ref_root(), c.ref_node(b["id"]), "--", ".", f":!{c.campaign_rel}", check=False
        )
        best = {
            "node": b["id"],
            "score": b["score"],
            "mechanism": b["mech"],
            "cases": {
                cid: {k: v.get(k) for k in ("value", "baseline", "ratio")} for cid, v in case_metrics(score).items()
            },
            "lineage": [s.split("] ", 1)[-1] for s in lineage(c, b["id"])],
            "diffstat": diffstat.splitlines()[-1].strip() if diffstat else "",
            "ref": c.ref_node(b["id"]),
        }

    insights = {}
    for r in reversed(rounds):
        p = c.ledger / "rounds" / f"r{r['round']:02d}" / "insights.json"
        if p.exists():
            try:
                insights = json.loads(p.read_text())
                insights["round"] = r["round"]
            except json.JSONDecodeError:
                pass
            break

    policies = []
    pdir = c.ledger / "policies"
    active = (pdir / "ACTIVE").read_text().strip() if (pdir / "ACTIVE").exists() else None
    for v in (
        sorted((p for p in pdir.glob("v*") if p.is_dir() and "-" not in p.name), key=lambda p: int(p.name[1:]))
        if pdir.exists()
        else []
    ):
        notes = (v / "notes.md").read_text() if (v / "notes.md").exists() else ""
        rep = json.loads((v / "replay.json").read_text()) if (v / "replay.json").exists() else {}
        policies.append(
            {
                "version": v.name,
                "title": _first_heading(notes),
                "V": rep.get("objective_V"),
                "replay_rounds": rep.get("rounds"),
                "active": v.name == active,
                "ran": [r["round"] for r in rounds if r["policy"] == v.name],
            }
        )
    dreaming = []
    for r in rounds:
        p = c.ledger / "rounds" / f"r{r['round']:02d}" / "dreamed.json"
        if p.exists():
            d = json.loads(p.read_text())
            dreaming.append(
                {
                    k: d.get(k)
                    for k in (
                        "after_round",
                        "current",
                        "candidate",
                        "current_V",
                        "candidate_V",
                        "winner",
                        "rejected",
                        "skipped",
                    )
                }
            )

    costs = read_jsonl(c.ledger / "costs.jsonl")
    by_kind: dict[str, float] = {}
    for x in costs:
        by_kind[x["kind"]] = round(by_kind.get(x["kind"], 0.0) + float(x.get("usd") or 0), 2)
    bud = json.loads((c.ledger / "budget.json").read_text()) if (c.ledger / "budget.json").exists() else {}
    attempts = sum(
        len(d.get("batch", []))
        for r in rounds
        for d in read_jsonl(c.ledger / "rounds" / f"r{r['round']:02d}" / "decisions.jsonl")
        if d.get("type") == "decision"
    )
    b = c.cfg["budget"]
    status = _status(c)
    spec = {k: c.cfg.get(k) for k in ("name", "description", "machine", "editable", "rules", "forbidden_patterns")}
    spec["eval"] = {k: c.cfg["eval"].get(k) for k in ("command", "direction", "unit", "gates", "timeout_s")}
    spec["search"], spec["budget"], spec["models"] = c.cfg["search"], c.cfg["budget"], c.cfg["models"]
    return {
        "campaign": c.name,
        "description": c.cfg.get("description", ""),
        "machine": c.cfg.get("machine"),
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "direction": c.cfg["eval"]["direction"],
        "unit": c.cfg["eval"].get("unit", ""),
        "status": status,
        "budget": {
            "attempts": attempts,
            "max_attempts": b["max_attempts"],
            "hours": round(float(bud.get("active_seconds", 0)) / 3600, 2),
            "max_hours": b["max_hours"],
            "usd": round(sum(by_kind.values()), 2),
            "max_usd": b["max_usd"],
        },
        "baseline": {
            "noise_pct": baseline["noise_pct"],
            "runs": baseline.get("runs"),
            "cases": {
                k: v.get("value", v.get("us_chip_mean"))
                for k, v in (baseline.get("cases") or baseline.get("shapes", {})).items()
            },
        }
        if baseline
        else None,
        "best": best,
        "rounds": rounds,
        "insights": insights,
        "policies": policies,
        "dreaming": dreaming,
        "costs": {"total": round(sum(by_kind.values()), 2), "by_kind": by_kind},
        "isolation": c.cfg.get("isolation"),
        "round_root": round_root_mode(c.cfg),
        "refs": {
            "prefix": c.ref_prefix,
            "best_branch": c.ref_best().removeprefix("refs/heads/"),
            "best_exists": c.ref_exists(c.ref_best()),
        },
        "spec_yaml": yaml.safe_dump(spec, sort_keys=False, width=110),
    }


def render(data: dict) -> str:
    blob = json.dumps(data).replace("</", "<\\/")
    title = html.escape(f"{data['campaign']} Dream-RSI")
    return TEMPLATE.read_text().replace("__TITLE__", title, 1).replace("/*__DREAM_DATA__*/null", blob)


def build(c: Campaign, out_dir: Path | None = None) -> Path:
    out = out_dir or c.home / "report"
    out.mkdir(parents=True, exist_ok=True)
    data = collect(c)
    (out / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "index.html").write_text(render(data))
    return out / "index.html"

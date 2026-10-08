"""Load recorded rounds (discovery trees) from git tags + the ledger, or from a JSON fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .campaign import Campaign, parse_node_id
from .policy_api import ROOT, NodeObs, RoundView


@dataclass
class RecordedRound:
    round: int
    branches: dict[int, list[NodeObs]]
    manifest: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)

    @property
    def nodes(self) -> list[NodeObs]:
        return [n for b in sorted(self.branches) for n in self.branches[b]]

    def closed(self) -> dict[int, str]:
        out = {}
        for d in self.decisions:
            if d.get("type") == "decision":
                out.update({int(k): v for k, v in d.get("closed", {}).items()})
        return out

    def steps_done(self) -> int:
        return sum(1 for d in self.decisions if d.get("type") == "decision" and d.get("batch"))

    def lost(self) -> int:
        return sum(1 for d in self.decisions if d.get("type") == "lost")


def _with_deltas(branches: dict[int, list[NodeObs]]) -> dict[int, list[NodeObs]]:
    for nodes in branches.values():
        prev_score = 1.0  # root
        prev_ok = True
        for n in nodes:
            n.delta_vs_parent = (n.score - prev_score) if (n.ok and prev_ok) else None
            prev_ok, prev_score = n.ok, n.score
    return branches


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def node_tags(c: Campaign, rnd: int | None = None) -> list[str]:
    pat = f"refs/tags/dream/{c.name}/n/" + (f"r{rnd:02d}-*" if rnd else "*")
    out = c.git("for-each-ref", "--format=%(refname:short)", pat)
    return sorted(line.rsplit("/", 1)[1] for line in out.splitlines() if line)


def read_node_file(c: Campaign, nid: str, rel: str) -> str | None:
    try:
        return c.git("show", f"{c.ref_node(nid)}:{c.attempts_rel(nid)}/{rel}")
    except RuntimeError:
        return None


def load_node(c: Campaign, nid: str, overrides: dict[str, dict]) -> NodeObs:
    _, b, a = parse_node_id(nid)
    meta = json.loads(read_node_file(c, nid, "node.json") or "{}")
    score = json.loads(read_node_file(c, nid, "eval/score.json") or "{}")
    valid = bool(score.get("valid", False))
    fail = score.get("fail_class", "infra" if not score else "ok")
    val = float(score.get("score", 0.0) or 0.0)
    if nid in overrides:
        o = overrides[nid]
        valid, fail, val = False, o.get("fail_class", "infra"), 0.0
    return NodeObs(
        node_id=nid,
        branch=b,
        attempt=a,
        parent=meta.get("parent", ROOT if a == 1 else f"{nid.rsplit('-', 1)[0]}-a{a - 1:02d}"),
        valid=valid,
        fail_class=fail,
        score=val if valid else 0.0,
        delta_vs_parent=None,
        tags=list(meta.get("tags", [])),
        mechanism=meta.get("mechanism", ""),
    )


def load_round(c: Campaign, rnd: int) -> RecordedRound:
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    decisions = _read_jsonl(rdir / "decisions.jsonl")
    overrides = {d["node"]: d for d in decisions if d.get("type") == "override"}
    manifest = json.loads((rdir / "manifest.json").read_text()) if (rdir / "manifest.json").exists() else {}
    branches: dict[int, list[NodeObs]] = {}
    for nid in node_tags(c, rnd):
        n = load_node(c, nid, overrides)
        branches.setdefault(n.branch, []).append(n)
    for b in branches:
        branches[b].sort(key=lambda n: n.attempt)
    return RecordedRound(rnd, _with_deltas(branches), manifest, decisions)


def recorded_rounds(c: Campaign) -> list[int]:
    rdir = c.ledger / "rounds"
    from_ledger = {int(p.name[1:]) for p in rdir.glob("r[0-9]*")} if rdir.exists() else set()
    from_tags = {parse_node_id(t)[0] for t in node_tags(c)}
    return sorted(from_ledger | from_tags)


def load_fixture(path: Path) -> tuple[list[RecordedRound], float]:
    """Fixture: {"noise_pct": 3, "rounds": [{"round": 1, "W": 4, "R": 4,
    "branches": {"1": [1.08, "build_error", ...], ...}}]} - a number is a valid score,
    a string is a fail_class."""
    data = json.loads(Path(path).read_text())
    rounds = []
    for r in data["rounds"]:
        branches = {}
        for b, cells in r["branches"].items():
            b = int(b)
            nodes = []
            for i, cell in enumerate(cells, start=1):
                ok = isinstance(cell, (int, float))
                nid = f"r{r['round']:02d}-b{b:02d}-a{i:02d}"
                nodes.append(
                    NodeObs(
                        node_id=nid,
                        branch=b,
                        attempt=i,
                        parent=ROOT if i == 1 else f"r{r['round']:02d}-b{b:02d}-a{i - 1:02d}",
                        valid=ok,
                        fail_class="ok" if ok else cell,
                        score=float(cell) if ok else 0.0,
                        delta_vs_parent=None,
                    )
                )
            branches[b] = nodes
        rounds.append(RecordedRound(r["round"], _with_deltas(branches), {"W": r.get("W"), "R": r.get("R")}))
    return rounds, float(data.get("noise_pct", 2.0))


def round_summary(rr: RecordedRound) -> dict:
    """Completed-round facts a policy's plan() may use."""
    ok = [n.score for n in rr.nodes if n.ok]
    return {
        "round": rr.round,
        "W": rr.manifest.get("W"),
        "R": rr.manifest.get("R"),
        "attempts": len(rr.nodes),
        "steps": rr.steps_done(),
        "best": max(ok) if ok else 1.0,
        "branches": {
            b: [{"attempt": n.attempt, "score": n.score, "fail_class": n.fail_class, "tags": n.tags} for n in nodes]
            for b, nodes in rr.branches.items()
        },
    }


def online_view(rr: RecordedRound, W: int, R: int, noise_pct: float) -> RoundView:
    return RoundView(
        round=rr.round,
        W=W,
        R=R,
        noise_pct=noise_pct,
        branches={b: list(v) for b, v in rr.branches.items()},
        closed=rr.closed(),
        steps_done=rr.steps_done(),
    )

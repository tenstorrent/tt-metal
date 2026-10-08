"""The interface between an exploration policy and the round it explores.

The same classes are used online (policy_step.py builds the view from git) and
in replay (replay.py builds it from a frozen recorded round), so a policy can't
tell the two apart. A policy file defines:

    class Policy:
        def __init__(self, config: dict): ...                  # config["beta"], config["defaults"]
        def plan(self, ctx: PlanContext) -> Plan: ...
        def select_batch(self, view: RoundView) -> Batch: ...   # empty items = stop
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = "root"


@dataclass
class NodeObs:
    """What a policy may see about one revealed attempt."""

    node_id: str
    branch: int
    attempt: int
    parent: str  # "root" or a node id
    valid: bool
    fail_class: (
        str  # ok, build_error, jit_compile_error, runtime_error, hang, accuracy_fail, forbidden_edit, infra, lost
    )
    score: float  # speedup vs campaign baseline; 0.0 when invalid
    delta_vs_parent: float | None  # score - parent's score, when both are valid (root counts as 1.0)
    tags: list[str] = field(default_factory=list)
    mechanism: str = ""

    @property
    def ok(self) -> bool:
        return self.valid and self.fail_class == "ok"


@dataclass
class RoundView:
    round: int
    W: int
    R: int
    noise_pct: float
    branches: dict[int, list[NodeObs]]  # branch -> attempts in order (revealed prefix only)
    closed: dict[int, str]  # branch -> reason
    steps_done: int

    # ---- derived helpers -------------------------------------------------
    @property
    def nodes(self) -> list[NodeObs]:
        return [n for b in sorted(self.branches) for n in self.branches[b]]

    @property
    def attempts(self) -> int:
        return sum(len(v) for v in self.branches.values())

    def head(self, branch: int) -> NodeObs:
        return self.branches[branch][-1]

    def anchor(self, branch: int) -> float | None:
        """Best valid score on a branch (its successful anchor)."""
        vals = [n.score for n in self.branches[branch] if n.ok]
        return max(vals) if vals else None

    def best(self) -> float:
        vals = [n.score for n in self.nodes if n.ok]
        return max(vals) if vals else 1.0

    def within_noise(self, delta: float | None, ref: float = 1.0) -> bool:
        return delta is not None and abs(delta) <= ref * self.noise_pct / 100.0

    def can_open_root(self) -> bool:
        return len(self.branches) < self.W

    def legal_heads(self) -> list[str]:
        return [
            self.head(b).node_id
            for b in sorted(self.branches)
            if b not in self.closed and len(self.branches[b]) < self.R
        ]

    def legal(self) -> list[str]:
        return ([ROOT] if self.can_open_root() else []) + self.legal_heads()


@dataclass
class PlanContext:
    round: int
    defaults: dict  # campaign policy_defaults
    history: list[dict]  # one summary per earlier round (see tree.round_summary)


@dataclass
class Plan:
    W: int
    R: int
    reason: str


@dataclass
class BatchItem:
    parent: str  # ROOT or a legal head node id
    role: str = "exploit"  # exploit | explore | recover
    why: str = ""


@dataclass
class Batch:
    items: list[BatchItem] = field(default_factory=list)
    closed: dict[int, str] = field(default_factory=dict)  # branches to close now, with reasons

    def to_json(self) -> dict:
        return {"items": [asdict(i) for i in self.items], "closed": {str(k): v for k, v in self.closed.items()}}


def validate(view: RoundView, batch: Batch) -> list[str]:
    """Return a list of problems; empty means the batch is legal."""
    errs = []
    if len(batch.items) > view.W:
        errs.append(f"batch has {len(batch.items)} items > W={view.W}")
    heads = set(view.legal_heads())
    closing = set(batch.closed)
    n_roots = 0
    seen = set()
    for it in batch.items:
        if it.parent == ROOT:
            n_roots += 1
            continue
        if it.parent in seen:
            errs.append(f"duplicate parent {it.parent}")
        seen.add(it.parent)
        if it.parent not in heads:
            errs.append(f"{it.parent} is not a legal head")
        elif any(view.head(b).node_id == it.parent for b in closing):
            errs.append(f"{it.parent} is on a branch closed in the same batch")
    if len(view.branches) + n_roots > view.W:
        errs.append(f"opening {n_roots} branches would exceed W={view.W}")
    for b in batch.closed:
        if b not in view.branches:
            errs.append(f"cannot close unknown branch {b}")
    return errs


def load_policy(path: str | Path, config: dict):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"dream_policy_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Policy(config)


def replay_value(best: float, attempts: int, steps: int, cost: float, bonus: float) -> float:
    return best - cost * attempts + bonus * attempts / max(1, steps)


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

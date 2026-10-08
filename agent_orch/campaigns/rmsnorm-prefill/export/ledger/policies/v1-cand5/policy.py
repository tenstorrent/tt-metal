"""v1-cand5: v1 + history-derived depth cap (data-driven alternative to cand2).

Prefix signals: as v1, plus earlier rounds' per-depth scores (PlanContext
history, stored at plan time).
Batch rule: as v1. New: a branch is closed once it has `cap` attempts, where
cap = (deepest attempt index that lifted an earlier round's running best by
more than noise_pct) + depth_slack(beta), at least 2. No history -> no cap.
Beta: gap = noise_pct * (2 + 8*beta); depth_slack = round(2*beta) - 1
(0 at beta 0.6, +1 at 0.8-1.0).
Plan: campaign defaults (the cap needs noise_pct, which only the view has).
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "depth_slack": round(2.0 * beta) - 1}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)
        self.history = []

    def plan(self, ctx: PlanContext) -> Plan:
        self.history = ctx.history
        return Plan(
            W=int(ctx.defaults.get("W", 4)),
            R=int(ctx.defaults.get("R", 4)),
            reason=(
                f"campaign defaults ({len(ctx.history)} earlier round(s)); per-branch depth cap = deepest attempt "
                "that lifted an earlier round's best by more than noise (applied once noise_pct is known)"
            ),
        )

    def _useful_depth(self, noise_pct: float):
        """Deepest attempt index at which some earlier round's running best (over all
        attempts up to that depth, across branches) rose by more than noise_pct."""
        tol = noise_pct / 100.0
        deepest = None
        for h in self.history:
            nodes = [n for ns in h["branches"].values() for n in ns]
            if not nodes:
                continue
            run, d_use = 1.0, 0
            for d in range(1, max(n["attempt"] for n in nodes) + 1):
                ok = [n["score"] for n in nodes if n["attempt"] == d and n["fail_class"] == "ok"]
                top = max(ok, default=0.0)
                if top > run * (1 + tol):
                    d_use = d
                run = max(run, top)
            deepest = d_use if deepest is None else max(deepest, d_use)
        return deepest

    def _depth_reason(self, view: RoundView, b: int):
        d = self._useful_depth(view.noise_pct)
        if d is None:
            return None
        cap = max(2, d + self.s["depth_slack"])
        if len(view.branches[b]) >= cap:
            return f"depth cap {cap}: deeper attempts never lifted an earlier round's best by more than noise"
        return None

    def _gap_reason(self, view: RoundView, b: int, best: float):
        anchor = view.anchor(b)
        if anchor is None or anchor >= best:
            return None
        gap = self.s["gap_noise"] * view.noise_pct / 100.0
        if (best - anchor) / best > gap:
            return f"anchor {anchor:.4f} trails leader {best:.4f} by more than {100 * gap:.1f}%"
        return None

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = self._gap_reason(view, b, best) or self._depth_reason(view, b)
                if why:
                    closed[b] = why
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        for b in sorted(view.branches):
            h = view.head(b)
            if b in closed or h.node_id not in view.legal_heads():
                continue
            role = "exploit" if h.ok else "recover"
            items.append(BatchItem(h.node_id, role, "refine branch within gap of leader"))
        return Batch(items=items[: view.W], closed=closed)

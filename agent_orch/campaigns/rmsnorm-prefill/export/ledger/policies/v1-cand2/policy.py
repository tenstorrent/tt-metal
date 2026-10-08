"""v1-cand2: v1 with a shallower round (R from beta).

Prefix signals: branch anchors and the round leader (as v1).
Batch rule: unchanged from v1 (open W branches, refine every open branch,
gap-close branches whose anchor trails the leader by more than gap(beta)).
Plan: R = clamp(2 + round(2*beta), 2, default R) -> 3 at beta 0.6. In the
recorded rounds the 4th attempts never lifted the round best by more than
noise_pct, and every round re-roots at the previous round's best, so depth
lost here continues in the next round from a fresher root.
Beta: gap = noise_pct * (2 + 8*beta); R as above (2 / 3 / 3 / 4 / 4 for
beta 0.2 / 0.4 / 0.6 / 0.8 / 1.0).
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "R": 2 + round(2.0 * beta)}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)
        self.history = []

    def plan(self, ctx: PlanContext) -> Plan:
        r_max = int(ctx.defaults.get("R", 4))
        R = max(2, min(r_max, self.s["R"]))
        if R < r_max:
            why = (
                f"R={R} (beta {self.beta}): recorded 4th attempts never lifted a round best by more than noise; "
                "the next round re-roots at this round's best, so depth continues there"
            )
        else:
            why = "campaign defaults"
        return Plan(W=int(ctx.defaults.get("W", 4)), R=R, reason=f"{why} ({len(ctx.history)} earlier round(s))")

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
                why = self._gap_reason(view, b, best)
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

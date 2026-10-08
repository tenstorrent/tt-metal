"""v1-cand4: v1 + final-attempt gate (adaptive alternative to cand2's R=3).

Prefix signals: branch anchors, the round leader, the branch's last delta.
Batch rule: as v1. New: a branch about to take its final (R-th) attempt is
closed unless it holds the round lead and its last attempt lifted its anchor
by more than momentum(beta) * noise_pct.
Beta: gap = noise_pct * (2 + 8*beta); momentum = 1 + 2*(0.6 - beta) clamped
to >= 0 (1.0 at beta 0.6; lower bar at higher beta).
Plan: campaign defaults.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "momentum_noise": max(0.0, 1.0 + 2.0 * (0.6 - beta))}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)
        self.history = []

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(
            W=int(ctx.defaults.get("W", 4)),
            R=int(ctx.defaults.get("R", 4)),
            reason=f"campaign defaults ({len(ctx.history)} earlier round(s); no wider/deeper plan in support)",
        )

    def _gap_reason(self, view: RoundView, b: int, best: float):
        anchor = view.anchor(b)
        if anchor is None or anchor >= best:
            return None
        gap = self.s["gap_noise"] * view.noise_pct / 100.0
        if (best - anchor) / best > gap:
            return f"anchor {anchor:.4f} trails leader {best:.4f} by more than {100 * gap:.1f}%"
        return None

    def _last_attempt_reason(self, view: RoundView, b: int, best: float):
        """Before a branch's final (R-th) attempt: spend it only on the round leader,
        and only if its last attempt lifted the branch anchor by more than
        momentum(beta) * noise_pct."""
        nodes = view.branches[b]
        if len(nodes) != view.R - 1:
            return None
        anchor = view.anchor(b)
        h = nodes[-1]
        prev = [n.score for n in nodes[:-1] if n.ok]
        tol = self.s["momentum_noise"] * view.noise_pct / 100.0
        lifted = h.ok and prev and h.score > max(prev) * (1 + tol)
        if anchor is not None and anchor >= best and lifted:
            return None
        return "final attempt reserved for a leader that is still clearly improving"

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = self._gap_reason(view, b, best) or self._last_attempt_reason(view, b, best)
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

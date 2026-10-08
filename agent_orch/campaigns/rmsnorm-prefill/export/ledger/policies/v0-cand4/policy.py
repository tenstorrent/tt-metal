"""v0-cand4: v0-cand1 + final-attempt focus.

Prefix signals: each branch's anchor (best valid score on it) and the round's
best valid score so far.
Batch rule: step 1 opens W branches from the root. Later steps refine every
open branch, except that a branch whose anchor trails the round leader by more
than gap(beta) (relative, in multiples of noise_pct) is closed. A dip on a
branch never closes it by itself: the anchor, not the head, is compared, so a
regression followed by a repair keeps its slot. A branch with no valid result
yet has no anchor and is never gap-closed (failures are normally repairable).
Final-attempt focus: a non-leading branch whose next attempt would be its last
(R-1 attempts so far) is closed unless its anchor is within final_gap of the
leader; the leader's branch always gets its last attempt.
Beta: gap = noise_pct * (2 + 8*beta); final_gap = noise_pct * 2*beta.
Plan: campaign defaults (W, R); one recorded round, nothing wider is in support.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "final_noise": 2.0 * beta}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(
            W=int(ctx.defaults.get("W", 4)),
            R=int(ctx.defaults.get("R", 4)),
            reason=f"campaign defaults ({len(ctx.history)} earlier round(s); no wider/deeper plan in support)",
        )

    def _close_reason(self, view: RoundView, b: int, best: float):
        anchor = view.anchor(b)
        if anchor is None or anchor >= best:
            return None
        gap = self.s["gap_noise"] * view.noise_pct / 100.0
        if (best - anchor) / best > gap:
            return f"anchor {anchor:.4f} trails leader {best:.4f} by more than {100 * gap:.1f}%"
        final_gap = self.s["final_noise"] * view.noise_pct / 100.0
        if len(view.branches[b]) == view.R - 1 and (best - anchor) / best > final_gap:
            return f"last attempt only within {100 * final_gap:.1f}% of leader {best:.4f} (anchor {anchor:.4f})"
        return None

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = self._close_reason(view, b, best)
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

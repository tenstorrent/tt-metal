"""v0-cand2: parallel refine + no-signal pruning (root-relative, not leader-relative).

Prefix signals: each branch's anchor (best valid score on it).
Batch rule: step 1 opens W branches from the root. Later steps refine every
open branch, except that a branch whose anchor has not beaten the round root
(1.0) by more than lift(beta) (in multiples of noise_pct) is closed: its idea
has shown no signal. Other branches never affect a branch's fate. A dip on a
branch never closes it by itself: the anchor, not the head, is compared, so a
regression followed by a repair keeps its slot. A branch with no valid result
yet has no anchor and is never gap-closed (failures are normally repairable).
Beta: lift = noise_pct * (2 - 2*beta); higher beta asks for less signal.
Plan: campaign defaults (W, R); one recorded round, nothing wider is in support.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"lift_noise": 2.0 - 2.0 * beta}


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
        if anchor is None:
            return None
        lift = self.s["lift_noise"] * view.noise_pct / 100.0
        if anchor <= 1.0 + lift:
            return f"anchor {anchor:.4f} not above root by more than {100 * lift:.1f}%"
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
            items.append(BatchItem(h.node_id, role, "refine branch with signal over root"))
        return Batch(items=items[: view.W], closed=closed)

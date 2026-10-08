"""v1-cand1: v1 + plateau closure.

Prefix signals: branch anchors, the round leader, and each branch's last k
attempts.
Batch rule: as v1 (open W branches, refine every open branch, gap-close
branches whose anchor trails the leader by more than gap(beta)). New: a branch
is also closed when its last k attempts were all valid, none was a clear
regression (a repair in progress is not a plateau), and together they lifted
the branch anchor by no more than noise_pct.
Beta: gap = noise_pct * (2 + 8*beta); k = 1 + round(2*beta) (2 at 0.6).
Plan: campaign defaults.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "plateau_k": 1 + round(2.0 * beta)}


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

    def _plateau_reason(self, view: RoundView, b: int):
        """The last k attempts are all valid, none is a clear regression (that is a
        repair in progress), and together they lifted the branch anchor by no more
        than noise_pct."""
        k = self.s["plateau_k"]
        nodes = view.branches[b]
        if len(nodes) < k + 1:
            return None
        window, before = nodes[-k:], nodes[:-k]
        if not all(n.ok for n in window):
            return None
        tol = view.noise_pct / 100.0
        prev = [n.score for n in before if n.ok]
        if not prev:
            return None
        a0 = max(prev)
        if any(n.score < a0 * (1 - tol) for n in window):
            return None
        a1 = max(a0, max(n.score for n in window))
        if a1 <= a0 * (1 + tol):
            return f"plateau: last {k} attempts moved the anchor {a0:.4f} -> {a1:.4f} (within {view.noise_pct:.1f}%)"
        return None

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = self._gap_reason(view, b, best) or self._plateau_reason(view, b)
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

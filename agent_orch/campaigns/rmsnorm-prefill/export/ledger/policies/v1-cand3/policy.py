"""v1-cand3: cand2 (R from beta) + cand1 (plateau closure).

Batch rule: v1's gap closure plus a plateau close: last k attempts all valid,
no clear regression among them, anchor lift within noise_pct.
Plan: R = clamp(2 + round(2*beta), 2, default R) (3 at beta 0.6).
Beta: gap = noise_pct * (2 + 8*beta); k = 1 + round(2*beta); R as above.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {
        "gap_noise": 2.0 + 8.0 * beta,
        "plateau_k": 1 + round(2.0 * beta),
        "R": 2 + round(2.0 * beta),
    }


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

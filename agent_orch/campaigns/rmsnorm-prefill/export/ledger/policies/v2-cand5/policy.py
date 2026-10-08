"""v2-cand5: v2 + step-1 duplicate closure (isolates cand2's second rule).

Prefix signals: branch anchors and the round leader (as v1).
Batch rule: unchanged from v1 (open W branches, refine every open branch,
gap-close branches whose anchor trails the leader by more than gap(beta)), plus:
right after step 1, a branch whose first attempt is within dup_band of the
leading first attempt and shares at least half of its tags is closed as a copy
of the leader's idea.
Plan: R = clamp(2 + round(2*beta), 2, default R) -> 3 at beta 0.6. In the
recorded rounds the 4th attempts never lifted the round best by more than
noise_pct, and every round re-roots at the previous round's best, so depth
lost here continues in the next round from a fresher root.
Beta: gap = noise_pct * (2 + 8*beta); R as above (2 / 3 / 3 / 4 / 4 for
beta 0.2 / 0.4 / 0.6 / 0.8 / 1.0); dup_band = noise_pct * max(0, 1.6 - beta).
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "R": 2 + round(2.0 * beta), "dup_noise": max(0.0, 1.6 - beta)}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

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

    def _dup_reasons(self, view: RoundView) -> dict:
        """Step-1 duplicates: workers that start together from the same root with
        the same history tend to land on the same idea. A first attempt that sits
        within dup_band of the leading first attempt and shares at least half of
        its tags (and at least 2) is treated as a copy of the leader's idea."""
        if any(len(ns) != 1 for ns in view.branches.values()) or len(view.branches) < 2:
            return {}
        firsts = {b: ns[0] for b, ns in view.branches.items() if ns[0].ok}
        if len(firsts) < 2:
            return {}
        lead = max(sorted(firsts), key=lambda b: firsts[b].score)
        ln = firsts[lead]
        band = self.s["dup_noise"] * view.noise_pct / 100.0
        out = {}
        for b, n in sorted(firsts.items()):
            if b == lead:
                continue
            shared = set(n.tags) & set(ln.tags)
            need = max(2, -(-min(len(n.tags), len(ln.tags)) // 2))
            if 1.0 - n.score / ln.score <= band and len(shared) >= need:
                out[b] = (
                    f"duplicate of b{lead:02d}'s start: within {100 * band:.1f}% of it, "
                    f"shares tags {sorted(shared)}"
                )
        return out

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {b: w for b, w in self._dup_reasons(view).items() if b not in view.closed}
        for b in sorted(view.branches):
            if b not in view.closed and b not in closed:
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

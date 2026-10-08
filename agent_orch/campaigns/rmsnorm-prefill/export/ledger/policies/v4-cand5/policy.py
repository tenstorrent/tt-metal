"""v4-cand5: v4-cand1 + close a branch whose head broke a campaign rule (forbidden_edit).

Forbidden heads (new, as v4-cand2): a forbidden_edit head is not a repairable
failure. Its idea is the forbidden change itself, and a child built on it
inherits the edit (r04: b01-a02 HiFi2 -> b01-a03 HiFi2, both overridden).

Prefix signals: branch anchors, the round leader (as v2) and each refinement's
lift over its branch's prior anchor, plus one number from the previous round's
summary: how much its deepest attempts lifted that round's best.
Batch rule: v4's (open W branches, refine every open branch, gap-close branches
whose anchor trails the leader by more than gap(beta), stall-close branches
whose last refinement landed within noise of their anchor, stop instead of a
lone single-item step).
Earned depth (new): in plan(), take the previous round's deepest recorded
attempt depth d and the lift of its best depth-d attempt over everything before
it (root and shallower attempts). If that lift is within noise_pct, this round
caps depth at max(2, min(R, d) - 1): a branch whose head reaches the cap is
closed and the next round re-roots at the round best. If the deepest attempts
did pay, the full plan depth stays. A round run at the cap whose last attempts
pay earns the depth back, so the cap is not permanent.
Plan: R = clamp(2 + round(2*beta), 2, default R) (as v2); the soft cap is
applied in select_batch, where noise_pct is known.
Beta: gap = noise_pct * (2 + 8*beta); R 2/3/3/4/4; stall_k = 1 below beta 0.8,
2 from 0.8; lone step and earned depth only below beta 0.8.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {
        "gap_noise": 2.0 + 8.0 * beta,
        "R": 2 + round(2.0 * beta),
        "stall_k": 1 + int(beta >= 0.8),
        "lone_ok": beta >= 0.8,
        "earned_depth": beta < 0.8,
    }


def _deepest_lift(summary: dict):
    """(deepest attempt depth, lift of its best valid attempt over everything shallower)."""
    nodes = [n for ns in summary.get("branches", {}).values() for n in ns]
    if not nodes:
        return None
    d = max(n["attempt"] for n in nodes)
    ok = [n for n in nodes if n["fail_class"] == "ok" and n["score"] > 0]
    prior = max([n["score"] for n in ok if n["attempt"] < d] + [summary.get("root_score", 1.0)])
    deep = [n["score"] for n in ok if n["attempt"] == d]
    return d, (max(deep) / prior - 1.0) if deep else -1.0


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)
        self.R = int(self.defaults.get("R", 4))
        self.last_deep = None

    def plan(self, ctx: PlanContext) -> Plan:
        r_max = int(ctx.defaults.get("R", 4))
        self.R = max(2, min(r_max, self.s["R"]))
        if self.R < r_max:
            why = (
                f"R={self.R} (beta {self.beta}): recorded 4th attempts never lifted a round best by more than noise; "
                "the next round re-roots at this round's best, so depth continues there"
            )
        else:
            why = "campaign defaults"
        self.last_deep = _deepest_lift(ctx.history[-1]) if ctx.history and self.s["earned_depth"] else None
        if self.last_deep:
            d, lift = self.last_deep
            why += (
                f"; depth cap {max(2, min(self.R, d) - 1)} if noise_pct >= {100 * lift:+.2f}% "
                f"(previous round's depth-{d} attempts vs its shallower best)"
            )
        return Plan(W=int(ctx.defaults.get("W", 4)), R=self.R, reason=f"{why} ({len(ctx.history)} earlier round(s))")

    def _depth_cap(self, view: RoundView):
        if not self.last_deep:
            return None
        d, lift = self.last_deep
        if lift > view.noise_pct / 100.0:
            return None
        return max(2, min(view.R, d) - 1)

    def _rule_reason(self, view: RoundView, b: int):
        h = view.head(b)
        if h.fail_class == "forbidden_edit":
            return f"head {h.node_id} broke a campaign rule (forbidden_edit); a child would inherit the edit"
        return None

    def _gap_reason(self, view: RoundView, b: int, best: float):
        anchor = view.anchor(b)
        if anchor is None or anchor >= best:
            return None
        gap = self.s["gap_noise"] * view.noise_pct / 100.0
        if (best - anchor) / best > gap:
            return f"anchor {anchor:.4f} trails leader {best:.4f} by more than {100 * gap:.1f}%"
        return None

    def _stall_reason(self, view: RoundView, b: int):
        nodes = view.branches[b]
        k = self.s["stall_k"]
        if len(nodes) < k + 1:
            return None
        band = view.noise_pct / 100.0
        lifts = []
        for i in range(len(nodes) - k, len(nodes)):
            prior = [n.score for n in nodes[:i] if n.ok]
            if not nodes[i].ok or not prior:
                return None
            lift = nodes[i].score / max(prior) - 1.0
            if abs(lift) > band:
                return None
            lifts.append(lift)
        shown = ", ".join(f"{100 * x:+.2f}%" for x in lifts)
        return f"stalled: last {k} refinement(s) within {view.noise_pct:.1f}% noise of the anchor ({shown})"

    def _cap_reason(self, view: RoundView, b: int):
        cap = self._depth_cap(view)
        if cap is not None and len(view.branches[b]) >= cap:
            d, lift = self.last_deep
            return f"depth cap {cap}: previous round's depth-{d} attempts lifted its best only {100 * lift:+.2f}%; re-root"
        return None

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = (
                    self._rule_reason(view, b)
                    or self._gap_reason(view, b, best)
                    or self._stall_reason(view, b)
                    or self._cap_reason(view, b)
                )
                if why:
                    closed[b] = why
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        for b in sorted(view.branches):
            h = view.head(b)
            if b in closed or h.node_id not in view.legal_heads():
                continue
            role = "exploit" if h.ok and (view.anchor(b) or 0) <= h.score else "recover"
            items.append(BatchItem(h.node_id, role, "refine branch within gap of leader"))
        items = items[: view.W]
        if len(items) == 1 and items[0].parent != ROOT and not self.s["lone_ok"]:
            b = next(b for b in view.branches if view.head(b).node_id == items[0].parent)
            closed[b] = f"lone branch (anchor {view.anchor(b) or 0.0:.4f}, leader {best:.4f}): stop and re-root"
            items = []
        return Batch(items=items, closed=closed)

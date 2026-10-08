"""v0-cand5: v0-cand1 + plateau and repeated-hard-failure closes.

Prefix signals: each branch's anchor (best valid score on it) and the round's
best valid score so far.
Batch rule: step 1 opens W branches from the root. Later steps refine every
open branch, except that a branch whose anchor trails the round leader by more
than gap(beta) (relative, in multiples of noise_pct) is closed. A dip on a
branch never closes it by itself: the anchor, not the head, is compared, so a
regression followed by a repair keeps its slot. A branch with no valid result
yet has no anchor and is never gap-closed (failures are normally repairable).
Also closes a branch after plateau_n consecutive valid results within noise of
their parent, or after two consecutive hard failures (hang, runtime_error,
accuracy_fail) of the same class. Build/compile errors never close a branch.
Beta: gap = noise_pct * (2 + 8*beta); plateau_n = 2 + round(beta).
Plan: campaign defaults (W, R); one recorded round, nothing wider is in support.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6
HARD = {"hang", "runtime_error", "accuracy_fail"}


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "plateau_n": 2 + round(beta)}


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
        nodes = view.branches[b]
        tail = nodes[-self.s["plateau_n"] :]
        if len(tail) == self.s["plateau_n"] and all(
            n.ok and view.within_noise(n.delta_vs_parent, n.score - n.delta_vs_parent) for n in tail
        ):
            return f"{len(tail)} valid results in a row within noise of their parent"
        last2 = nodes[-2:]
        if len(last2) == 2 and last2[0].fail_class == last2[1].fail_class and last2[0].fail_class in HARD:
            return f"two {last2[0].fail_class} in a row"
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

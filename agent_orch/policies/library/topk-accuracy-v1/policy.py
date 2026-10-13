"""v1 (= v1-cand1, from v0): v0 with one branch fewer at low/medium beta.

Prefix signals: none beyond legality (as v0).
Batch rule: step 1 opens W branches from the root; every later step refines
every open branch whose head has fewer than R attempts. Never closes a branch:
r01 showed three branches failing twice (accuracy_fail) and then reaching the
round best at attempts 3-4, so failures are not closing evidence here.
Plan (new): W = default W - 1 below beta 0.8, default W from 0.8. In r01 every
branch converged on the same mechanism by attempt 3 (three of them by porting
b03's work), so the 4th parallel branch duplicated work. W - 1 stays inside the
recorded support; R stays at the default (the round best came at depth 4).
Beta: only the width (narrow below 0.8).
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"width_cut": 1 if beta < 0.8 else 0}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

    def plan(self, ctx: PlanContext) -> Plan:
        w_def = int(ctx.defaults.get("W", 4))
        W = max(2, w_def - self.s["width_cut"])
        R = int(ctx.defaults.get("R", 4))
        if W < w_def:
            why = f"W={W} (beta {self.beta}): parallel branches converged on one mechanism by porting; default R={R}"
        else:
            why = "campaign defaults"
        return Plan(W=W, R=R, reason=f"{why} ({len(ctx.history)} earlier round(s))")

    def select_batch(self, view: RoundView) -> Batch:
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        items += [BatchItem(h, "exploit", "refine every open branch") for h in view.legal_heads()]
        return Batch(items=items[: view.W])

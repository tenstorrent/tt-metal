"""v0: parallel refine (the paper's fixed-exploration baseline).

Prefix signals: none beyond legality.
Batch rule: step 1 opens W branches from the root; every later step refines
every open branch whose head has fewer than R attempts. Never closes a branch.
Beta: unused (v0 has no thresholds).
Plan: campaign defaults (W, R).
Why: it records the most complete branch x attempt grid, which is the best
first world to dream in.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(
            W=int(ctx.defaults.get("W", 4)),
            R=int(ctx.defaults.get("R", 4)),
            reason="v0: fixed campaign defaults",
        )

    def select_batch(self, view: RoundView) -> Batch:
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        items += [BatchItem(h, "exploit", "refine every open branch") for h in view.legal_heads()]
        return Batch(items=items[: view.W])

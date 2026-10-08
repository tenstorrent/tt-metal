"""Example revision B: like A, but closes a branch on any failure."""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


class Policy:
    def __init__(self, config):
        self.beta = float(config.get("beta", DEFAULT_BETA))

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(int(ctx.defaults["W"]), int(ctx.defaults["R"]), "defaults")

    def select_batch(self, view: RoundView) -> Batch:
        closed = {}
        for b, nodes in view.branches.items():
            if b in view.closed:
                continue
            if not nodes[-1].ok:
                closed[b] = f"failed: {nodes[-1].fail_class}"
            elif len(nodes) >= 2 and all(n.ok and view.within_noise(n.delta_vs_parent) for n in nodes[-2:]):
                closed[b] = "two valid results within noise of parent"
        items = [BatchItem(ROOT, "explore", "open") for _ in range(view.W - len(view.branches))]
        for h in view.legal_heads():
            if not any(view.head(b).node_id == h for b in closed):
                items.append(BatchItem(h, "exploit", "refine"))
        return Batch(items=items[: view.W], closed=closed)

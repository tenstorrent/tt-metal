"""Example revision A: close after two consecutive valid results within noise of their
parent, or two consecutive hard failures of the same kind. A build error never closes."""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6
HARD = {"accuracy_fail", "hang", "runtime_error"}


class Policy:
    def __init__(self, config):
        self.beta = float(config.get("beta", DEFAULT_BETA))

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(int(ctx.defaults["W"]), int(ctx.defaults["R"]), "defaults")

    def _close_reason(self, view: RoundView, b: int):
        last2 = view.branches[b][-2:]
        if len(last2) < 2:
            return None
        if all(n.ok and view.within_noise(n.delta_vs_parent) for n in last2):
            return "two valid results within noise of parent"
        if last2[0].fail_class == last2[1].fail_class and last2[0].fail_class in HARD:
            return f"two {last2[0].fail_class} in a row"
        return None

    def select_batch(self, view: RoundView) -> Batch:
        closed = {}
        for b in view.branches:
            if b not in view.closed:
                why = self._close_reason(view, b)
                if why:
                    closed[b] = why
        items = [BatchItem(ROOT, "explore", "open") for _ in range(view.W - len(view.branches))]
        for h in view.legal_heads():
            if not any(view.head(b).node_id == h for b in closed):
                items.append(BatchItem(h, "exploit", "refine"))
        return Batch(items=items[: view.W], closed=closed)

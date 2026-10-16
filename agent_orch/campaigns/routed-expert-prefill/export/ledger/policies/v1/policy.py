"""v1: frontier prune + collapse close + recover role (promoted from v0-cand3 after r03).

Prefix signals: each branch's head and its best valid node before the head, the round's best
revealed score (root included), the running best by depth, noise_pct.
Batch rule: refine every open branch (v0), with these closes:
  - collapse (CLOSE_COLLAPSE): a valid head more than collapse_x_noise x noise below its
    branch's earlier best;
  - frontier (FRONTIER_PRUNE): a branch with at least min_attempts attempts whose valid head is
    more than band_x_noise x noise below the round's best revealed score. With FLAT_GATE this only
    applies after a step whose running best gained no more than flat_x_noise x noise. Failed heads
    are exempt (failures are repairable) unless FRONTIER_FAILED;
  - stall (CLOSE_STALL): the last two attempts are valid and within stall_x_noise x noise of
    their parent.
Recover role (RECOVER_ROLE): a refinement whose head failed or sits more than noise below its
revert target is tagged `recover`, naming the revert target and, when another branch leads by
more than noise, the node to port from.
Beta: one _schedule(beta); collapse 5 x (1 + beta) x noise, frontier band and stall band
(0.5 + beta) x noise (0.7% / 1.1% / 1.5% at beta 0.2 / 0.6 / 1.0). High beta keeps more open.
Plan: campaign defaults (W, R); every recorded round ran 4 x 4 and the next round re-roots on
the best node, so a wider or deeper grid would be out of support.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6

CLOSE_COLLAPSE = True
RECOVER_ROLE = True
FRONTIER_PRUNE = True
FLAT_GATE = False
FRONTIER_FAILED = False
CLOSE_STALL = False


def _schedule(beta: float) -> dict:
    """Every threshold, as a multiple of noise or a count; high beta = wider and more patient."""
    return {
        "collapse_x_noise": 5.0 * (1.0 + beta),  # 8% at beta 0.6 with 1% noise
        "band_x_noise": 0.5 + beta,  # frontier band, 1.1% at beta 0.6
        "min_attempts": 2,  # a branch is judged against the frontier from its second attempt
        "flat_x_noise": 1.0,
        "stall_x_noise": 0.5 + beta,
    }


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

    def plan(self, ctx: PlanContext) -> Plan:
        W, R = int(ctx.defaults.get("W", 4)), int(ctx.defaults.get("R", 4))
        if not ctx.history:
            return Plan(W, R, "campaign defaults: no earlier round to size the grid from")
        return Plan(W, R, "campaign defaults: every recorded round ran 4 x 4 and the next round re-roots on the best node")

    # ---- prefix signals --------------------------------------------------
    @staticmethod
    def _prior_anchor(view: RoundView, b: int):
        """Best valid node on branch b before its head, or None."""
        prior = [n for n in view.branches[b][:-1] if n.ok]
        return max(prior, key=lambda n: n.score) if prior else None

    @staticmethod
    def _best_node(view: RoundView):
        ok = [n for n in view.nodes if n.ok]
        return max(ok, key=lambda n: n.score) if ok else None

    @staticmethod
    def _best_upto(view: RoundView, d: int) -> float:
        return max([n.score for n in view.nodes if n.ok and n.attempt <= d] + [view.root_score])

    def _flat(self, view: RoundView) -> bool:
        """The latest step's running best gained no more than flat_x_noise x noise."""
        d = max(len(v) for v in view.branches.values())
        gain = self.s["flat_x_noise"] * view.noise_pct / 100.0
        return d >= 2 and self._best_upto(view, d) <= self._best_upto(view, d - 1) * (1.0 + gain)

    def _close_reason(self, view: RoundView, b: int, gate_open: bool):
        noise = view.noise_pct / 100.0
        nodes = view.branches[b]
        head = nodes[-1]
        prior = self._prior_anchor(view, b)
        if CLOSE_COLLAPSE and head.ok and prior is not None:
            drop = (prior.score - head.score) / prior.score
            if drop > self.s["collapse_x_noise"] * noise:
                return f"valid head fell {drop:.0%} below the branch's best ({prior.node_id})"
        if FRONTIER_PRUNE and gate_open and len(nodes) >= self.s["min_attempts"]:
            best = view.best()
            if not head.ok and FRONTIER_FAILED:
                return f"head failed ({head.fail_class}) while the round's best is {best:.4f}"
            if head.ok and head.score < best * (1.0 - self.s["band_x_noise"] * noise):
                return f"head {head.score:.4f} is more than {self.s['band_x_noise']:.1f}x noise below the round's best {best:.4f}"
        if CLOSE_STALL and len(nodes) >= 2:
            band = self.s["stall_x_noise"] * noise
            if all(
                n.ok and n.delta_vs_parent is not None and abs(n.delta_vs_parent) <= band * (n.score - n.delta_vs_parent)
                for n in nodes[-2:]
            ):
                return "two valid results in a row within noise of their parent"
        return None

    def _item(self, view: RoundView, b: int) -> BatchItem:
        head = view.head(b)
        if RECOVER_ROLE:
            noise = view.noise_pct / 100.0
            prior = self._prior_anchor(view, b)
            if prior is not None and prior.score > view.root_score:
                ref, where = prior.score, prior.node_id
            else:
                ref, where = view.root_score, "the round root"
            top = self._best_node(view)
            port = (
                f"; the round's best so far is {top.node_id} ({top.score:.4f}), port from it if the idea composes"
                if top is not None and top.score > ref * (1.0 + noise) and top.branch != b
                else ""
            )
            if not head.ok:
                return BatchItem(head.node_id, "recover", f"head failed ({head.fail_class}); repair it, or revert to {where} ({ref:.4f}){port}")
            if head.score < ref * (1.0 - noise):
                return BatchItem(head.node_id, "recover", f"head is below {where} ({ref:.4f}); revert to it before building{port}")
        return BatchItem(head.node_id, "exploit", "refine open branch")

    def select_batch(self, view: RoundView) -> Batch:
        gate_open = bool(view.branches) and (not FLAT_GATE or self._flat(view))
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed and len(view.branches[b]) < view.R:
                why = self._close_reason(view, b, gate_open)
                if why:
                    closed[b] = why
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        for b in sorted(view.branches):
            if b in view.closed or b in closed or len(view.branches[b]) >= view.R:
                continue
            items.append(self._item(view, b))
        return Batch(items=items[: view.W], closed=closed)

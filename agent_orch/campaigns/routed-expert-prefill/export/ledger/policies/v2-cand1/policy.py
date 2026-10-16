"""v2-cand1: v2 + late steps only continue repairs (from step 3 on, refine only branches still recovering)."""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6

CLOSE_COLLAPSE = True
RECOVER_ROLE = True
FRONTIER_PRUNE = True
FLAT_GATE = False
FRONTIER_FAILED = False
CLOSE_STALL = False
LEADER_GUARD = False
FLAT_STOP = False
SHALLOW_DEPTH = 1  # None = campaign default R; else rounds after the first run this many steps
REPAIR_STEP = False  # None = beta decides (see _schedule); False = never
LATE_REPAIR_ONLY = True  # from late_step on, refine only branches whose head or previous attempt failed or sat below the root
RESCUE_STEP = False  # rounds after the first get one more step only when step 1 found no gain outside noise
FIRST_PRUNE = False  # a valid, above-root head trailing the round's best is judged from its first attempt
HIGH_BETA_FULL_STEP = True  # v2: above beta 0.7 rounds after the first run one more full step
LATE_KEEP_IMPROVING = False  # late steps also keep a valid head that beat its parent by more than band_x_noise x noise


def _schedule(beta: float) -> dict:
    """Every threshold, as a multiple of noise or a count; high beta = wider and more patient."""
    return {
        "collapse_x_noise": 5.0 * (1.0 + beta),  # 8% at beta 0.6 with 1% noise
        "band_x_noise": 0.5 + beta,  # frontier band, 1.1% at beta 0.6
        "min_attempts": 2,  # a branch is judged against the frontier from its second attempt
        "flat_x_noise": 1.0,
        "stall_x_noise": 0.5 + beta,
        "late_step": 2 + (1 if beta > 0.7 else 0),  # steps after which valid, above-root lines stop refining
        # steps per round once there is history; above beta 0.7 one extra step (full if HIGH_BETA_FULL_STEP, else repairs only)
        "depth": None if SHALLOW_DEPTH is None else SHALLOW_DEPTH + (1 if HIGH_BETA_FULL_STEP and beta > 0.7 else 0),
        "repair_step": beta > 0.7 if REPAIR_STEP is None else REPAIR_STEP,
        # step 1 "found a gain" when its best beats the root by more than this; 1x noise at beta 0.6
        "rescue_x_noise": 0.4 + beta,
    }


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

    def _extra_step(self) -> bool:
        return bool(self.s["repair_step"] or RESCUE_STEP)

    def plan(self, ctx: PlanContext) -> Plan:
        W, R = int(ctx.defaults.get("W", 4)), int(ctx.defaults.get("R", 4))
        if not ctx.history:
            return Plan(W, R, "campaign defaults: no earlier round to size the grid from")
        depth = self.s["depth"]
        why = "in r02-r06 step 1 set the round's largest gain and steps 2-4 never paid for their attempts; the next round re-roots on the best node and runs W wide from it"
        if depth is not None and depth < R:
            if self._extra_step():
                extra = []
                if self.s["repair_step"]:
                    extra.append("repairs failed heads")
                if RESCUE_STEP:
                    extra.append("refines every head when step 1 found no gain outside noise")
                return Plan(W, depth + 1, f"{W} wide, {depth} step(s) deep plus one conditional step that {' and '.join(extra)}: {why}")
            return Plan(W, depth, f"{W} wide, {depth} step(s) deep: {why}")
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

    @staticmethod
    def _repairing(view: RoundView, b: int) -> bool:
        """The branch's head is a valid repair of a failed attempt."""
        ns = view.branches[b]
        return ns[-1].ok and len(ns) >= 2 and not ns[-2].ok

    @staticmethod
    def _recovering(view: RoundView, b: int) -> bool:
        """The head, or the attempt before it, failed or scored below the round root."""
        return any(not n.ok or n.score < view.root_score for n in view.branches[b][-2:])

    def _improving(self, view: RoundView, b: int) -> bool:
        """The head is valid and beat its parent by more than band_x_noise x noise."""
        h = view.head(b)
        if not h.ok or h.delta_vs_parent is None:
            return False
        parent = h.score - h.delta_vs_parent
        return parent > 0 and h.delta_vs_parent > self.s["band_x_noise"] * view.noise_pct / 100.0 * parent

    def _found_gain(self, view: RoundView) -> bool:
        """Some valid node beats the round root by more than rescue_x_noise x noise."""
        return view.best() > view.root_score * (1.0 + self.s["rescue_x_noise"] * view.noise_pct / 100.0)

    def _close_reason(self, view: RoundView, b: int, gate_open: bool):
        noise = view.noise_pct / 100.0
        nodes = view.branches[b]
        head = nodes[-1]
        prior = self._prior_anchor(view, b)
        if CLOSE_COLLAPSE and head.ok and prior is not None:
            drop = (prior.score - head.score) / prior.score
            if drop > self.s["collapse_x_noise"] * noise:
                return f"valid head fell {drop:.0%} below the branch's best ({prior.node_id})"
        judged = len(nodes) >= self.s["min_attempts"] or (FIRST_PRUNE and head.ok and head.score > view.root_score)
        if FRONTIER_PRUNE and gate_open and judged:
            best = view.best()
            if LEADER_GUARD and best > view.root_score and view.anchor(b) == best:
                return None  # the branch holds the round's best node; its next attempt reverts to it
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
        open_ = [b for b in sorted(view.branches) if b not in view.closed and b not in closed and len(view.branches[b]) < view.R]
        if LATE_REPAIR_ONLY and view.steps_done >= self.s["late_step"]:
            # late steps cost more than they add to the parallel bonus; spend them only on repairs in flight
            open_ = [b for b in open_ if self._recovering(view, b) or (LATE_KEEP_IMPROVING and self._improving(view, b))]
        depth = self.s["depth"]
        if self._extra_step() and depth is not None and view.R == depth + 1 and view.steps_done >= depth:
            # the extra step after a shallow round: every head if step 1 found no gain (rescue), else only failed heads
            if RESCUE_STEP and not self._found_gain(view):
                items = [self._item(view, b) for b in open_]
            elif self.s["repair_step"]:
                items = [self._item(view, b) for b in open_ if not view.head(b).ok]
            else:
                items = []
            return Batch(items=items[: view.W], closed=closed)
        if FLAT_STOP and view.steps_done >= 2 and self._flat(view) and not any(self._repairing(view, b) for b in open_):
            return Batch(items=[], closed=closed)  # stop: the last step set no best outside noise
        items = [BatchItem(ROOT, "explore", "open branch") for _ in range(view.W - len(view.branches))]
        items += [self._item(view, b) for b in open_]
        return Batch(items=items[: view.W], closed=closed)

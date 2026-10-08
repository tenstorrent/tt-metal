"""v3-cand3: v3-cand1 + extra depth for climbing branches.

Prefix signals: branch anchors, the round leader (as v2) and each refinement's
lift over its branch's prior anchor.
Batch rule: v2's (open W branches, refine every open branch, gap-close
branches whose anchor trails the leader by more than gap(beta)), plus: close a
branch once its last `stall_k` refinements were valid and each landed within
noise_pct of the branch's prior anchor (relative). A clear regression is not a
stall: it is a repair in progress and keeps its slot. First attempts are never
judged by this rule.
Lone step: if the batch would be a single refinement, stop instead (below
beta 0.8), whether or not that branch holds the round lead. A one-item step
pays the full attempt cost and cuts the mean batch width, so it costs about
as much as a full-width step while testing one idea. The next round re-roots
at the round best and refines it with full width.
Plan: R = default R (4). Up to the soft depth clamp(2 + round(2*beta), 2, R)
every branch may refine (as v2). Past it, a branch refines only while its
last attempt raised the branch anchor by more than climb*noise_pct (still
climbing); other branches close at the soft depth.
Beta: gap = noise_pct * (2 + 8*beta); R 2/3/3/4/4; stall_k = 1 below beta 0.8,
2 from 0.8.
"""

from dream.policy_api import ROOT, Batch, BatchItem, Plan, PlanContext, RoundView

DEFAULT_BETA = 0.6


def _schedule(beta: float) -> dict:
    return {"gap_noise": 2.0 + 8.0 * beta, "R": 2 + round(2.0 * beta), "stall_k": 1 + int(beta >= 0.8), "lone_ok": beta >= 0.8, "climb_noise": 2.0}


class Policy:
    def __init__(self, config: dict):
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.defaults = config.get("defaults", {})
        self.s = _schedule(self.beta)

    def plan(self, ctx: PlanContext) -> Plan:
        R = int(ctx.defaults.get("R", 4))
        why = (
            f"R={R} (campaign default); branches past depth {self._soft(R)} continue only while climbing "
            "(r04 summary: every branch was still improving when R=3 stopped it)"
        )
        return Plan(W=int(ctx.defaults.get("W", 4)), R=R, reason=f"{why} ({len(ctx.history)} earlier round(s))")

    def _soft(self, R: int) -> int:
        return max(2, min(R, self.s["R"]))

    def _depth_reason(self, view: RoundView, b: int):
        nodes = view.branches[b]
        soft = self._soft(view.R)
        if len(nodes) < soft:
            return None
        h = nodes[-1]
        prior = [n.score for n in nodes[:-1] if n.ok]
        climb = self.s["climb_noise"] * view.noise_pct / 100.0
        if h.ok and prior and h.score / max(prior) - 1.0 > climb:
            return None
        return f"soft depth {soft} reached and last attempt did not lift the anchor by more than {100 * climb:.1f}%"

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

    def select_batch(self, view: RoundView) -> Batch:
        best = view.best()
        closed = {}
        for b in sorted(view.branches):
            if b not in view.closed:
                why = self._gap_reason(view, b, best) or self._stall_reason(view, b) or self._depth_reason(view, b)
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

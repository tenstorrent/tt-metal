from langgraph.graph import StateGraph, END
from models.experimental.opt_transfer.schema import BringupState
from models.experimental.opt_transfer.handoff import dump_bundle
from models.experimental.opt_transfer.config import CONFIG


class RealImpl:
    """Production node implementations binding KB/trace/matcher/structural/codegen/verify.
    Device-touching; used by run.py and the e2e."""

    def __init__(self, model_key, device, matcher, kb):
        from models.experimental.opt_transfer.config import CONFIG

        self.cfg = CONFIG.models[model_key]
        self.device = device
        self.matcher = matcher
        self.kb = kb
        # Cross-node runtime objects (not serialisable; kept on self, not in state).
        self._ref = None
        self._graph = None
        self._proposals = []
        self._applied = []
        self._runners = []

    def trace(self, state):
        import importlib
        import torch
        from models.experimental.opt_transfer.trace import trace_module

        ref_mod = importlib.import_module(self.cfg["reference"]).SeamlessBlock(
            self.cfg["embed_dim"], self.cfg["num_heads"]
        )
        g = trace_module(ref_mod, (torch.randn(1, 8, self.cfg["embed_dim"]),))
        self._ref = ref_mod
        self._graph = g
        state["graph_summary"] = g.summary_json()
        return state

    def match(self, state):
        props = self.matcher.propose(state["graph_summary"], self.kb, diagnosis=state.get("diagnosis"))
        self._proposals = props
        state["proposals"] = [p.__dict__ for p in props]
        return state

    def gate(self, state):
        from models.experimental.opt_transfer.structural import validate

        kb_by_id = {e.id: e for e in self.kb}
        applied = []
        for p in self._proposals:
            ok, reason = validate(self._graph, p, kb_by_id[p.entry_id])
            if ok:
                applied.append(p)
        self._applied = applied
        state["applied"] = [p.entry_id for p in applied]
        return state

    def codegen(self, state):
        from models.experimental.opt_transfer.codegen import build_fused

        dims = {"H": self.cfg["num_heads"], "D": self.cfg["head_dim"], "embed": self.cfg["embed_dim"]}
        kb_by_id = {e.id: e for e in self.kb}
        ref = self._ref
        runners = []
        for p in self._applied:
            p = p.resolve(dims)
            weights = {
                n: {"weight": getattr(ref, n).weight.detach(), "bias": getattr(ref, n).bias.detach()}
                for n in p.matched_nodes
            }
            runners.append(build_fused(p, kb_by_id[p.entry_id], weights, self.device, dims))
        self._runners = runners
        return state

    def verify(self, state):
        # Per-block: compare fused QKV against the reference's separate-projection split.
        # Both sides must consume the SAME input the q/k/v projections see in the reference
        # forward, i.e. the post-attn_norm hidden state h — not raw x.
        import torch
        from models.experimental.opt_transfer.verify import pcc

        ref = self._ref
        embed = self.cfg["embed_dim"]
        ref.eval()
        with torch.no_grad():
            x = torch.randn(1, 64, embed)
            h = ref.attn_norm(x)
        worst = 1.0
        for run in self._runners:
            q, k, v = run(h)
            for name, got in zip(("q_proj", "k_proj", "v_proj"), (q, k, v)):
                with torch.no_grad():
                    gold = ref._split(getattr(ref, name)(h))
                worst = min(worst, pcc(gold, got))
        state["full_pcc"] = worst
        return state

    def repair(self, state):
        state["iteration"] = state.get("iteration", 0) + 1
        return state


def build_graph(impl, max_iterations: int = None):
    """impl provides node callables: trace, match, gate, codegen, verify, repair.
    Kept injectable so the graph is testable without device/API."""
    max_it = max_iterations or CONFIG.gates["max_iterations"]
    thr = CONFIG.gates["full_pcc"]

    def verify_node(state):
        return impl.verify(state)

    def perf_node(state):
        state["status"] = "pass"
        return state

    def handoff_node(state):
        dump_bundle(state, state.get("run_dir", CONFIG.run_dir))
        state["status"] = "handoff"
        return state

    def route(state) -> str:
        if state.get("full_pcc", 0.0) >= thr:
            return "perf"
        if state.get("iteration", 0) >= max_it:
            return "handoff"
        return "repair"

    wf = StateGraph(BringupState)
    wf.add_node("trace", impl.trace)
    wf.add_node("match", impl.match)
    wf.add_node("gate", impl.gate)
    wf.add_node("codegen", impl.codegen)
    wf.add_node("verify", verify_node)
    wf.add_node("repair", impl.repair)
    wf.add_node("perf", perf_node)
    wf.add_node("handoff", handoff_node)

    wf.set_entry_point("trace")
    wf.add_edge("trace", "match")
    wf.add_edge("match", "gate")
    wf.add_edge("gate", "codegen")
    wf.add_edge("codegen", "verify")
    wf.add_conditional_edges("verify", route, {"perf": "perf", "repair": "repair", "handoff": "handoff"})
    wf.add_edge("repair", "match")
    wf.add_edge("perf", END)
    wf.add_edge("handoff", END)
    return wf.compile()

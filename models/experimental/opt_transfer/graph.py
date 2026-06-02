from langgraph.graph import StateGraph, END
from models.experimental.opt_transfer.schema import BringupState
from models.experimental.opt_transfer.handoff import dump_bundle
from models.experimental.opt_transfer.config import CONFIG


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

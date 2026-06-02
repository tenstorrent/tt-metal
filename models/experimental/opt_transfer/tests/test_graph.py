from models.experimental.opt_transfer.graph import build_graph


class Fakes:
    def __init__(self, pcc_sequence):
        self.pcc_sequence = list(pcc_sequence)
        self.i = 0

    def trace(self, state):
        state["graph_summary"] = [{"name": "q_proj"}]
        return state

    def match(self, state):
        state["proposals"] = [{"entry_id": "qkv_merge"}]
        return state

    def gate(self, state):
        state["applied"] = ["qkv_merge"]
        return state

    def codegen(self, state):
        return state

    def verify(self, state):
        state["full_pcc"] = self.pcc_sequence[min(self.i, len(self.pcc_sequence) - 1)]
        self.i += 1
        return state

    def repair(self, state):
        state["iteration"] = state.get("iteration", 0) + 1
        return state


def test_graph_reaches_serve_on_pass():
    f = Fakes(pcc_sequence=[0.999])
    g = build_graph(f, max_iterations=3)
    out = g.invoke({"model": "seamless_m4t_v2", "iteration": 0})
    assert out["status"] == "pass"


def test_graph_hands_off_after_exhausting_iterations(tmp_path):
    f = Fakes(pcc_sequence=[0.80, 0.80, 0.80, 0.80])
    g = build_graph(f, max_iterations=3)
    out = g.invoke({"model": "seamless_m4t_v2", "iteration": 0, "run_dir": str(tmp_path)})
    assert out["status"] == "handoff"

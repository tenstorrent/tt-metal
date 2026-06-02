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

    def perf(self, state):
        state["perf"] = {"naive_ms": 100.0, "fused_ms": 60.0}
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


class GateFakes:
    def __init__(self, pcc, drift, gain=50.0):
        self.pcc, self.drift, self.gain = pcc, drift, gain

    def trace(self, s):
        s["graph_summary"] = [{"name": "q_proj"}]
        return s

    def match(self, s):
        s["proposals"] = [{"entry_id": "x"}]
        return s

    def gate(self, s):
        s["applied"] = ["x"]
        return s

    def codegen(self, s):
        return s

    def verify(self, s):
        s["full_pcc"] = self.pcc
        s["drift"] = {"first_divergence_step": self.drift, "horizon": 100}
        return s

    def perf(self, s):
        s["perf"] = {"naive_ms": 100.0, "fused_ms": 100.0 - self.gain}
        return s

    def repair(self, s):
        s["iteration"] = s.get("iteration", 0) + 1
        return s


def test_drift_failure_routes_to_repair_then_handoff(tmp_path):
    # pcc passes but divergence at step 20 of 100 (< 90% threshold) -> repair, exhaust -> handoff
    g = build_graph(GateFakes(pcc=0.999, drift=20), max_iterations=2)
    out = g.invoke({"model": "m", "iteration": 0, "run_dir": str(tmp_path)})
    assert out["status"] == "handoff"


def test_pass_records_perf_and_flags_weak_gain():
    g = build_graph(GateFakes(pcc=0.999, drift=100, gain=0.5))  # gain 0.5% < threshold
    out = g.invoke({"model": "m", "iteration": 0})
    assert out["status"] == "pass"
    assert out["perf"]["gain_pct"] < 2.0
    assert out.get("perf_warnings")  # flagged, not blocked


from models.experimental.opt_transfer.repair import build_diagnosis


class RepairFakes:
    def __init__(self):
        self.seen_diag = []
        self.calls = 0

    def trace(self, s):
        s["graph_summary"] = [{"name": "q"}]
        return s

    def match(self, s):
        self.seen_diag.append(s.get("diagnosis"))
        s["proposals"] = [{"entry_id": "x"}]
        return s

    def gate(self, s):
        s["applied"] = ["x"]
        return s

    def codegen(self, s):
        return s

    def verify(self, s):
        self.calls += 1
        s["full_pcc"] = 0.999 if self.calls >= 2 else 0.80  # fails once, then passes
        s["drift"] = {"first_divergence_step": 100, "horizon": 100}
        return s

    def perf(self, s):
        s["perf"] = {"naive_ms": 100.0, "fused_ms": 60.0}
        return s

    def repair(self, s):
        s["diagnosis"] = build_diagnosis(
            node="x", per_block_pcc=s["full_pcc"], tf_pcc=None, free_run_divergence_frac=None, config_tried={}
        ).__dict__
        s["iteration"] = s.get("iteration", 0) + 1
        return s


def test_repair_sets_diagnosis_and_match_consumes_it():
    f = RepairFakes()
    out = build_graph(f, max_iterations=3).invoke({"model": "m", "iteration": 0})
    assert out["status"] == "pass"
    assert f.seen_diag[0] is None  # first match: no diagnosis
    assert f.seen_diag[1] is not None  # second match: repair's diagnosis forwarded
    assert f.seen_diag[1]["node"] == "x" and f.seen_diag[1]["axis"] == "per_block_pcc"


from langgraph.checkpoint.memory import MemorySaver


def test_checkpointer_persists_state_under_thread():
    f = Fakes(pcc_sequence=[0.999])  # the H2 Fakes (success on first verify)
    g = build_graph(f, max_iterations=3, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    g.invoke({"model": "m", "iteration": 0}, cfg)
    snap = g.get_state(cfg)
    assert snap.values["status"] == "pass"  # state retrievable from the checkpoint

from agent import states
from agent.handlers import decide as decide_mod
from agent.handlers.remeasure import _comparable


def _prof(buckets):
    return {"buckets": buckets}


def test_comparable_ok():
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    it = _prof([{"id": "matmul", "count": 94, "device_ms": 6.0}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    assert _comparable(base, it) == (True, None)


def test_comparable_opcount_collapse():
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    it = _prof([{"id": "other", "count": 1, "device_ms": 0.1}, {"id": "datamove", "count": 9, "device_ms": 0.09}])
    ok, reason = _comparable(base, it)
    assert not ok and "structural_op_dropped" in reason and "matmul" in reason


def test_comparable_partial_drops_structural_not_to_zero():
    base = _prof(
        [{"id": "matmul", "count": 559, "device_ms": 30.0}, {"id": "datamove", "count": 1742, "device_ms": 40.0}]
    )
    it = _prof([{"id": "matmul", "count": 60, "device_ms": 3.0}, {"id": "datamove", "count": 200, "device_ms": 5.0}])
    ok, reason = _comparable(base, it)
    assert not ok and "structural_op_dropped" in reason and "matmul" in reason


def test_comparable_dominant_missing():
    base = _prof([{"id": "datamove", "count": 50, "device_ms": 6.7}, {"id": "matmul", "count": 50, "device_ms": 5.0}])
    it = _prof([{"id": "matmul", "count": 48, "device_ms": 5.0}])
    ok, reason = _comparable(base, it)
    assert not ok and "dominant_bucket_missing" in reason


def test_comparable_structural_bucket_vanished():
    base = _prof(
        [
            {"id": "datamove", "count": 1742, "device_ms": 40.0},
            {"id": "matmul", "count": 559, "device_ms": 30.0},
            {"id": "eltwise", "count": 1124, "device_ms": 20.0},
            {"id": "attention", "count": 1, "device_ms": 0.7},
        ]
    )
    it = _prof(
        [
            {"id": "datamove", "count": 1740, "device_ms": 39.0},
            {"id": "matmul", "count": 557, "device_ms": 29.0},
            {"id": "eltwise", "count": 1123, "device_ms": 20.0},
        ]
    )
    ok, reason = _comparable(base, it)
    assert not ok and "structural_op_dropped" in reason and "attention" in reason


def test_comparable_fusable_bucket_drop_not_rejected():
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 20, "device_ms": 2.0}])
    it = _prof([{"id": "matmul", "count": 96, "device_ms": 6.0}])
    assert _comparable(base, it) == (True, None)


def test_comparable_valid_layout_coherence_reduction_accepted():
    base = _prof(
        [
            {"id": "datamove", "count": 1742, "device_ms": 57.0},
            {"id": "matmul", "count": 559, "device_ms": 27.0},
            {"id": "eltwise", "count": 1124, "device_ms": 5.0},
            {"id": "attention", "count": 18, "device_ms": 1.0},
        ]
    )
    it = _prof(
        [
            {"id": "datamove", "count": 200, "device_ms": 8.0},
            {"id": "matmul", "count": 559, "device_ms": 27.0},
            {"id": "eltwise", "count": 1124, "device_ms": 5.0},
            {"id": "attention", "count": 18, "device_ms": 1.0},
        ]
    )
    assert _comparable(base, it) == (True, None)


def test_comparable_op_count_inflated_rejected():
    base = _prof([{"id": "matmul", "count": 100, "device_ms": 6.7}, {"id": "datamove", "count": 100, "device_ms": 5.0}])
    it = _prof([{"id": "matmul", "count": 100, "device_ms": 6.7}, {"id": "datamove", "count": 400, "device_ms": 20.0}])
    ok, reason = _comparable(base, it)
    assert not ok and "op_count_inflated" in reason


class _Ctx:
    def __init__(self, last_decision, direction="min"):
        self.state = {"last_decision": last_decision, "metric": {"direction": direction}}
        self.events = []

    def log_event(self, *a):
        self.events.append(a)


def test_decide_discards_untrusted_measurement():
    ctx = _Ctx({"before": 12.10, "after": 0.55, "measurement_ok": False, "measurement_reason": "op_count_mismatch: x"})
    nxt = decide_mod.decide(ctx)
    assert ctx.state["last_decision"]["result"] == "discard"
    assert "op_count_mismatch" in ctx.state["last_decision"]["reason"]
    assert nxt == states.REVERT


def test_decide_keeps_real_gain():
    ctx = _Ctx({"before": 12.10, "after": 11.50, "measurement_ok": True})
    nxt = decide_mod.decide(ctx)
    assert ctx.state["last_decision"]["result"] == "keep" and nxt == states.COMMIT


def test_decide_flags_suspicious_but_keeps():
    ctx = _Ctx({"before": 12.10, "after": 3.0, "measurement_ok": True})
    decide_mod.decide(ctx)
    d = ctx.state["last_decision"]
    assert d["result"] == "keep" and d.get("suspicious_gain") is not None

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
    assert not ok and "op_count_mismatch" in reason


def test_comparable_dominant_missing():
    base = _prof([{"id": "matmul", "count": 50, "device_ms": 6.7}, {"id": "eltwise", "count": 50, "device_ms": 0.1}])
    it = _prof([{"id": "eltwise", "count": 95, "device_ms": 0.1}])
    ok, reason = _comparable(base, it)
    assert not ok and "dominant_bucket_missing" in reason


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

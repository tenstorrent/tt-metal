from scripts.tt_hw_planner.commands.optimize import invalid_trace_flag_error


def _set(monkeypatch, v):
    if v is None:
        monkeypatch.delenv("TT_PERF_TRACE", raising=False)
    else:
        monkeypatch.setenv("TT_PERF_TRACE", v)


def test_unset_is_ok(monkeypatch):
    _set(monkeypatch, None)
    assert invalid_trace_flag_error() is None


def test_zero_is_ok(monkeypatch):
    _set(monkeypatch, "0")
    assert invalid_trace_flag_error() is None


def test_one_is_ok(monkeypatch):
    _set(monkeypatch, "1")
    assert invalid_trace_flag_error() is None


def test_two_errors_clearly(monkeypatch):
    _set(monkeypatch, "2")
    err = invalid_trace_flag_error()
    assert err is not None
    assert "TT_PERF_TRACE" in err
    assert "TT_PERF_NUM_CQ" in err


def test_garbage_errors(monkeypatch):
    _set(monkeypatch, "yes")
    assert invalid_trace_flag_error() is not None

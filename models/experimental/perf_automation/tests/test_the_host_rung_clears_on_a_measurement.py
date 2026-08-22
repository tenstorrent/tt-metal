"""One failed trace attempt must not seal the dispatch axis forever.

The host branch of _op_ladder_status cleared on PRESENCE:

    if not (kinds & {"structural", "trace", "trace-capture"}):
        return (False, "trace-capture", ...)
    return (True, "done", ...)

so a single row of any of those kinds closed the rung whatever it measured. Its comment said so
outright -- "a trace that doesn't help still counts as tried".

On gemma-3-12b-it that is precisely what happened. Run 20 recorded:

    op_signature=host_overhead   kind=trace   ms=400.44   beat_baseline=False

and dispatch was never offered again. Across 158 later attempts the axis has ONE entry, against
grid's 37 and shard's 26 -- while 20.92 ms of host_overhead sat in every profile and every top-gap op
kept reporting bound_by=dispatch. "Attempted" was being read as "resolved".

The same file already solves this correctly for the KV-cache lever. _decode_gate:

    kv_won = any(_ledger().is_win(a) for a in kv_clean)
    if kv_won or (len(kv_clean) + kv_wedged) >= max_kv:
        return None

-- clears on a MEASURED reduction, or after N real attempts so an infeasible lever cannot loop. Its
comment names this exact trap: it "clears ONLY when a KV-cache attempt actually reduced cost. A
generic 'structural'/trace attempt does NOT clear it."

This applies that rule to the host rung. Nothing here is model-specific: the branch keys on
bound_by == "host" / bucket == "host_fallback", which any profile with a material host_overhead
bucket reaches. It matters most on dispatch-dominated models, where the DRAM roofline says least.

PERF_MCP_MAX_HOST_ATTEMPTS (default 3) bounds it. Wedged attempts count toward the cap, matching how
_decode_gate treats a cache that crashes every time.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    monkeypatch.delenv("PERF_MCP_MAX_HOST_ATTEMPTS", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


HOST_OP = {
    "op_code": "host_overhead",
    "bucket": "host_fallback",
    "bound_by": "host",
    "gap_ms": 20.9195,
    "grid": "",
    "weight_dtype": "",
}


def _att(kind, beat=False, ms=400.44, **kw):
    r = {
        "op_signature": "host_overhead",
        "kernel_kind": kind,
        "measured_ms": ms,
        "beat_baseline": beat,
        "kernel_detected_in_source": True,
    }
    r.update(kw)
    return r


def _status(mcp, attempts):
    return mcp._op_ladder_status(HOST_OP, "host_overhead", attempts)


# ---------------------------------------------------------------- the reported case


def test_one_failed_trace_does_not_close_the_axis(mcp):
    """The gemma3 case verbatim: trace, 400.44 ms, beat_baseline=False."""
    done, rung, _ = _status(mcp, [_att("trace", beat=False)])
    assert done is False and rung == "trace-capture"


def test_an_untouched_axis_is_offered(mcp):
    done, rung, _ = _status(mcp, [])
    assert done is False and rung == "trace-capture"


# ---------------------------------------------------------------- it clears on a MEASURED win


def test_a_measured_win_clears_it_immediately(mcp):
    """A lever that actually reduced cost is resolved; there is nothing left to ask for."""
    done, _rung, reason = _status(mcp, [_att("trace", beat=True)])
    assert done is True and "WON" in reason


def test_a_win_clears_even_with_earlier_failures(mcp):
    done, _r, _x = _status(mcp, [_att("trace"), _att("structural"), _att("trace-capture", beat=True)])
    assert done is True


# ---------------------------------------------------------------- ...or after the cap


def test_the_cap_stops_it_looping(mcp):
    """An irreducible dispatch residual must retire, or the gate orders the same rewrite forever."""
    done, _r, reason = _status(mcp, [_att("trace"), _att("structural"), _att("trace-capture")])
    assert done is True and "cap" in reason


def test_two_failures_still_leave_one_attempt(mcp):
    done, rung, _ = _status(mcp, [_att("trace"), _att("structural")])
    assert done is False and rung == "trace-capture"


def test_the_cap_is_configurable(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_HOST_ATTEMPTS", "1")
    importlib.reload(mcp)
    done, _r, _x = _status(mcp, [_att("trace")])
    assert done is True, "a cap of 1 should retire after the first attempt"


def test_a_wedged_attempt_counts_toward_the_cap(mcp):
    """A transform that crashes every time is 'tried' -- mirrors _decode_gate's kv_wedged."""
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps([_att("trace", wedged=True), _att("structural", wedged=True)]))
    done, _r, _x = _status(mcp, [_att("trace")])
    assert done is True


# ---------------------------------------------------------------- unrelated attempts do not count


def test_a_matmul_attempt_does_not_clear_the_host_rung(mcp):
    other = _att("trace")
    other["op_signature"] = "MatmulDeviceOperation 32 x 3840 x 8192"
    done, rung, _ = _status(mcp, [other])
    assert done is False and rung == "trace-capture"


def test_a_knob_rung_does_not_clear_the_host_rung(mcp):
    """grid/dtype/shard act on device ops; none of them touches launch overhead."""
    done, rung, _ = _status(mcp, [_att("grid"), _att("dtype"), _att("shard"), _att("fidelity")])
    assert done is False and rung == "trace-capture"


# ---------------------------------------------------------------- the gate forwards it


def test_the_host_gate_offers_it_once_the_ladder_does(mcp):
    """_host_gate is a pass-through: it returns None only because the ladder said done."""
    prof = {
        "device_ms": 381.23,
        "buckets": [{"id": "host_overhead", "device_ms": 20.9195, "tags": {"source": "op_gap"}}],
    }
    blocking = mcp._host_gate(prof, [], [_att("trace", beat=False)])
    assert blocking and blocking.get("next_rung") == "trace-capture"
    assert blocking.get("bound_by") == "host"


def test_the_host_gate_still_yields_after_a_win(mcp):
    prof = {
        "device_ms": 381.23,
        "buckets": [{"id": "host_overhead", "device_ms": 20.9195, "tags": {"source": "op_gap"}}],
    }
    assert mcp._host_gate(prof, [], [_att("trace", beat=True)]) is None


def test_an_immaterial_host_bucket_is_still_ignored(mcp):
    """Unchanged behaviour: a bucket below the materiality threshold is not worth a round."""
    prof = {"device_ms": 381.23, "buckets": [{"id": "host_overhead", "device_ms": 0.01, "tags": {"source": "op_gap"}}]}
    assert mcp._host_gate(prof, [], []) is None

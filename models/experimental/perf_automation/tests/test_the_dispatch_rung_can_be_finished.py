"""A rung offered on a bucket cannot be retired by matching an op name it will never have.

The dispatch rung is offered on `host_overhead` -- idle time between launches. That is a bucket, not
an op: no class, no shape. Its lever is a transform of the GENERATION LOOP, so the agent edits the
loop and records the attempt against whatever it actually touched. _op_match then compares that op
class against "host_overhead", fails, and the try is not counted.

Measured on voxtral_mini_3b_2507 (2026-09-05): offered 54 times in one run, 0 recorded, cap 3 never
approached. The rung was re-issued for the entire run while prefill -- the only stage still short of
its band -- waited behind it.

The sibling counter already knew: _host_wedged read the full log unfiltered, because a wedge is a
property of the attempt and not of an op, while _host_tried filtered by op. Only the op filter was
wrong, so only the op filter changed -- the wedge counter still reads the full log, since the caller
filters `attempts` on kernel_detected_in_source and an attempt that wedged the device may never have
got one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HOST_OP = {"op_code": "host_overhead", "bucket": "host_fallback", "bound_by": "host"}


def _pm():
    spec = importlib.util.spec_from_file_location("pmcp_dispatch_rung", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _status(m, attempts):
    m._load_attempts = lambda: attempts
    return m._op_ladder_status(dict(_HOST_OP), "host_overhead", attempts)


def _trace(i, **kw):
    """What the agent really records: the rung's kind, against the op it edited."""
    return dict({"kernel_kind": "trace-capture", "op_signature": "MatmulDeviceOperation %d x 32 x 32" % i}, **kw)


def test_the_rung_retires_once_its_tries_are_spent():
    """The whole defect: three real tries, recorded the way the agent records them, must finish it."""
    m = _pm()
    assert _status(m, [_trace(i) for i in range(3)])[0] is True


def test_it_is_still_offered_before_the_cap():
    m = _pm()
    for n in (0, 1, 2):
        done, rung, _ = _status(m, [_trace(i) for i in range(n)])
        assert done is False and rung == "trace-capture", n


def test_a_try_counts_wherever_the_agent_recorded_it():
    """The lever edits the loop, so the op_signature is never the bucket the rung was offered on."""
    m = _pm()
    assert _status(m, [_trace(i) for i in range(3)])[0] is True, "still matching on an op name it cannot have"


def test_unrelated_structural_work_cannot_retire_the_dispatch_axis():
    """`structural` is a general rung. Counting it loose would let matmul work close this one."""
    m = _pm()
    other = [
        {"kernel_kind": "structural", "op_signature": "MatmulDeviceOperation %d x 8192 x 3072" % i} for i in range(5)
    ]
    assert _status(m, other)[0] is False


def test_a_wedged_lever_still_spends_the_allowance():
    """A transform that crashes the device every time must not be ordered forever either."""
    m = _pm()
    assert _status(m, [_trace(i, wedged=True) for i in range(3)])[0] is True


def test_the_try_counter_no_longer_filters_by_op():
    """The one line that was wrong. `matches` is the op-filtered list; trace kinds must not use it."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index('_host_kinds = {"structural", "trace", "trace-capture"}')
    code = "\n".join(ln for ln in src[i : i + 2600].splitlines() if not ln.strip().startswith("#"))
    tried = code[code.index("_host_tried = ") : code.index("_host_won")]
    assert "for a in attempts" in tried, "trace kinds are being filtered by op again"
    assert "for a in matches" in tried, "structural must still be scoped to this op"


def test_no_stage_or_model_name_is_typed_into_the_rung():
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index('_host_kinds = {"structural", "trace", "trace-capture"}')
    code = "\n".join(ln for ln in src[i : i + 2600].splitlines() if not ln.strip().startswith("#"))
    for typed in ("decode", "prefill", "encode", "voxtral"):
        assert typed not in code.lower(), typed

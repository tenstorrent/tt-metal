"""When the profile says the work is waiting, the run may not finish having only tuned ops.

`_wait_profile` now reports, per bucket, how much of the time an op HELD the device it spent
computing. On gemma-3-12b-it the answer is stark:

    matmul     fw 561.0  kernel 505.6  wait  55.3   busy
    eltwise    fw 112.1  kernel  21.6  wait  90.5   IDLE
    datamove   fw 137.6  kernel  21.1  wait 116.4   IDLE

The matmuls the ladder spent 130 attempts on are genuinely busy. 207 ms of idle sits in eltwise and
datamove, waiting on a producer, and no grid/dtype/shard/tt-lang/cpp edit to a waiting op recovers
one microsecond of it. That is why the end-to-end number did not move for seven hours.

Two things follow, and they are deliberately the smallest pair that changes the outcome:

  * The wait goes into the TARGET REASON. The agent reads one (op, rung) instruction per round; a
    catalogue entry it received on 3 of 51 recalls is not where decisions get made.
  * ONE model-scoped attempt must be on file before the run may stop. Not a rung: it never enters the
    op ladder, no op is blocked by it, the op x rung matrix is untouched, and it is satisfied by a
    single `model:*` row -- win, loss or "none: <evidence>".

It arms on measured evidence only. A model whose ops compute what they occupy never sees it, which
is why the trigger is the wait and not the roofline band -- 60-80%% of peak bandwidth is a dense-LLM
notion a compute-bound vision model would trip on for no reason. `kernel << FW` is true or false on
any architecture, with no knowledge of tokens, decode loops or transformers.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _buckets(idle=True):
    """The gemma3 shape: busy matmul, idle eltwise."""
    return [
        {
            "id": "matmul",
            "device_ms": 467.89,
            "wait": {"fw_ms": 561.0, "kernel_ms": 505.6, "wait_ms": 55.3, "idle": False},
        },
        {
            "id": "eltwise",
            "device_ms": 20.60,
            "wait": {"fw_ms": 112.1, "kernel_ms": 21.6, "wait_ms": 90.5, "idle": bool(idle)},
        },
    ]


def _attempt(sig, kind="grid"):
    return {"op_signature": sig, "kernel_kind": kind, "measured_ms": 400.0}


# ---------------------------------------------------------------- the gate arms on evidence


def test_a_waiting_profile_requires_a_model_attempt(mcp):
    assert mcp._model_lever_required(_buckets(idle=True), []) is True


def test_a_busy_profile_requires_nothing(mcp):
    """Every op computes what it occupies -> nothing upstream to fix, gate never arms."""
    assert mcp._model_lever_required(_buckets(idle=False), []) is False


def test_a_capture_with_no_raw_row_requires_nothing(mcp):
    """Absent evidence is not evidence. Older captures must not be blocked from finishing."""
    assert mcp._model_lever_required([{"id": "matmul", "device_ms": 400.0, "wait": None}], []) is False


def test_no_buckets_requires_nothing(mcp):
    assert mcp._model_lever_required([], []) is False


# ---------------------------------------------------------------- one attempt satisfies it


def test_a_recorded_model_attempt_satisfies_it(mcp):
    assert mcp._model_lever_required(_buckets(), [_attempt("model:prefetch", "prefetch")]) is False


def test_a_losing_model_attempt_still_satisfies_it(mcp):
    """Same contract as every rung: a measured dead end clears it. Otherwise the only way to finish
    is to win, which is not something a gate can require."""
    a = _attempt("model:prefetch", "prefetch")
    a["fullpipe_delta_ms"] = 2.4
    assert mcp._model_lever_required(_buckets(), [a]) is False


def test_a_different_model_lever_also_satisfies_it(mcp):
    """The evidence says 'work is waiting', not 'prefetch is the answer'. Host-loop is a legitimate
    response to the same signal and GUIDELINES 13 covers both."""
    assert mcp._model_lever_required(_buckets(), [_attempt("model:host-loop", "host")]) is False


def test_an_op_attempt_does_not_satisfy_it(mcp):
    """The whole point: tuning the waiting op is what does not work."""
    assert mcp._model_lever_required(_buckets(), [_attempt("BinaryNgDeviceOperation 128 x 15360", "shard")]) is True


def test_it_is_satisfied_once_and_stays_satisfied(mcp):
    """One attempt, not one per bucket -- a model-wide change is made once."""
    b = _buckets()
    b.append(
        {
            "id": "datamove",
            "device_ms": 19.5,
            "wait": {"fw_ms": 137.6, "kernel_ms": 21.1, "wait_ms": 116.4, "idle": True},
        }
    )
    assert mcp._model_lever_required(b, [_attempt("model:prefetch", "prefetch")]) is False


# ---------------------------------------------------------------- it never becomes a rung


def test_the_op_ladder_is_untouched(mcp):
    """A model lever must not appear as a rung on any op, or it enters the per-op checklist and
    blocks ops that have nothing to do with weight streaming."""
    done, rung, _r = mcp._op_ladder_status(
        {"op": "BinaryNgDeviceOperation", "grid": "full", "bound_by": "memory", "gap_ms": 8.0},
        "BinaryNgDeviceOperation",
        [_attempt("BinaryNgDeviceOperation", k) for k in ("grid", "fidelity", "dtype", "shard")],
    )
    assert "model" not in str(rung), rung

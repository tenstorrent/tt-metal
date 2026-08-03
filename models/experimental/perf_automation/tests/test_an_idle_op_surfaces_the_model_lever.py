"""An op that is waiting is not an op that is slow, and no rung on it can help.

The profiler charges an op the whole time it held the device. When that time is mostly spent
spinning on a producer's semaphore, the op tops the ranking while being the wrong thing to tune:

    GeGLU mul 128x15360 bf4_b:  40.557ms FW vs 6.151ms kernel   (34.4ms producer wait)

Six milliseconds of work in forty milliseconds of occupancy. Every lever on that op -- grid,
fidelity, dtype, shard, tt-lang, C++ -- makes the six smaller and cannot touch the thirty-four,
which belongs upstream. On gemma-3-12b-it that is the whole remaining gap: 130 attempts, one win,
and the end-to-end number never moved off 33.90 ms.

Two facts made this invisible to the loop rather than merely unaddressed:

  * The tool never computed the split. The report CSV carries one `Device Time` column; the FW/kernel
    pair lives in the RAW csv (`DEVICE FW DURATION [ns]`, `DEVICE KERNEL DURATION [ns]`), which
    _top_ops already has in hand per member and was discarding. The 34.4 ms figure above was measured
    by the agent by hand, in prose, into a stage label.
  * There is nowhere to record the fix. The loop hands out one (op, rung) pair and
    record_kernel_attempt takes an op_signature and a kernel_kind. A model-wide change -- weight
    prefetch, removing a host round-trip -- is neither, so doing it produces work that cannot be
    recorded, does not clear anything, and reads to the tool as a skipped rung.

So: surface the wait where the decision is made, and let ONE model-scoped attempt be recorded and
counted. Deliberately NOT a rung -- it never enters the per-op ladder, no op waits on it, and the
op x rung matrix is untouched. It activates only on measured evidence, so a model with no producer
wait never sees it.

The trigger is a measurement, not a model: `kernel << FW` is true or false on any architecture,
without knowing tokens, decode loops, or transformers. That is the whole reason it is gated on the
wait rather than on the roofline band -- 60-80%% of peak bandwidth is a dense-LLM notion that a
compute-bound vision model would trip on for no reason.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.agent import tracy_tool as tt  # noqa: E402


# ---------------------------------------------------------------- the split is captured


def _member(fw_ns, kernel_ns, device_us, op="BinaryNgDeviceOperation"):
    return {
        "report": {"OP Code": op, "Device Time": str(device_us), "Cores": "110", "Global Call Count": "1"},
        "raw": {"DEVICE FW DURATION [ns]": str(fw_ns), "DEVICE KERNEL DURATION [ns]": str(kernel_ns)},
    }


def test_fw_and_kernel_are_both_reported(tmp_path):
    """They are already attached per member as m['raw'] and were being dropped."""
    out = tt._wait_profile([_member(40_557_000, 6_151_000, 40557)])
    assert out is not None
    assert round(out["fw_ms"], 1) == 40.6 and round(out["kernel_ms"], 1) == 6.2


def test_the_wait_is_the_difference(tmp_path):
    out = tt._wait_profile([_member(40_557_000, 6_151_000, 40557)])
    assert round(out["wait_ms"], 1) == 34.4


def test_an_op_that_computes_what_it_occupies_shows_no_wait(tmp_path):
    out = tt._wait_profile([_member(6_200_000, 6_151_000, 6200)])
    assert out["wait_ms"] < 0.1 and out["idle"] is False


def test_the_geglu_case_is_flagged_idle(tmp_path):
    assert tt._wait_profile([_member(40_557_000, 6_151_000, 40557)])["idle"] is True


# ---------------------------------------------------------------- it needs real evidence


def test_a_small_absolute_wait_is_not_flagged(tmp_path):
    """A tiny op can be 90%% wait and still be worth nothing. Ratio alone would flag every cheap op
    in the profile and the note would stop being read."""
    out = tt._wait_profile([_member(300_000, 20_000, 300)])
    assert out["idle"] is False, out


def test_a_large_op_that_is_mostly_busy_is_not_flagged(tmp_path):
    """40 ms of occupancy against 35 ms of compute is a slow op, not an idle one."""
    out = tt._wait_profile([_member(40_000_000, 35_000_000, 40000)])
    assert out["idle"] is False, out


def test_missing_raw_columns_return_none(tmp_path):
    """Older captures and non-tracy paths have no raw row. Absent evidence must read as absent, not
    as zero wait -- the caller skips the note rather than asserting the op is busy."""
    assert tt._wait_profile([{"report": {"Device Time": "40557"}, "raw": {}}]) is None


def test_no_members_returns_none(tmp_path):
    assert tt._wait_profile([]) is None


def test_the_split_sums_across_members(tmp_path):
    """A bucket is many invocations; the wait is the bucket's, not one call's."""
    out = tt._wait_profile([_member(20_000_000, 3_000_000, 20000)] * 2)
    assert round(out["fw_ms"]) == 40 and round(out["kernel_ms"]) == 6


def test_a_garbage_value_does_not_raise(tmp_path):
    """Profiler CSVs carry blanks and '-' in the wild; this runs on every capture."""
    m = _member(40_557_000, 6_151_000, 40557)
    m["raw"]["DEVICE KERNEL DURATION [ns]"] = "-"
    assert tt._wait_profile([m]) is None

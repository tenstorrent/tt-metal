"""The ceiling divided by the census's byte TOTAL, which encodes the depth the census walked.

The perf test drives trace_replay, and trace_replay runs the weight census -- so the census executes
twice per cycle: once inside the full-pipeline gate, which measures every layer, and once inside the
TRACY profile, which is legitimately depth-capped because an uncapped capture overflows the marker
buffer. Whichever ran last wins.

On voxtral that is 7.043 GB against 1.718 GB -- a 4.1x swing in the ceiling's divisor decided purely
by ordering. Run 16 recorded one and run 17 the other, from identical code, and run 17's decode
printed a 4.31 ms floor against a 17.86 ms measurement.

simple_active_bytes twenty lines away already says why the total is the wrong quantity:
"Deliberately the RATIO and not the census's byte TOTAL. The total counts everything resident -- on
gemma-3, 15.49 GB of which ~6.85 GB is KV cache, which the ceiling must not divide by."

    the RATIO  1.3228 (2 layers) vs 1.3252 (62)   0.2% apart
    the TOTAL  1.718 GB vs 7.043 GB               4.1x apart
"""
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import perf_target as PT  # noqa: E402

_HW = {"dram_bw_gbps": 512.0, "peak_tflops_per_core": {"lofi": 5.4}, "cores": 130}
_BASE = {
    "weight_bytes": 9356474312,
    "dominant_dtype": "bfloat16",
    "source": "checkpoint bytes + HF config",
    "total_params": 3611483136,
    "device_census_complete": True,
}


def _ab(dwb, bpp):
    t = PT.compute_target({**_BASE, "device_weight_bytes": dwb, "bytes_per_param": bpp}, _HW)
    return getattr(t, "active_bytes", 0)


def test_the_depth_the_census_walked_no_longer_moves_the_ceiling():
    """THE WHOLE POINT. An average width does not care how many layers were built; a byte total is
    almost entirely a statement about how many were built."""
    two, full = _ab(1718081696, 1.3228), _ab(7043000000, 1.3252)
    assert abs(two - full) / full < 0.01, "the ceiling still swings with the census depth: %s vs %s" % (two, full)


def test_it_is_the_width_times_params_not_the_resident_total():
    ab = _ab(7043000000, 1.3252)
    assert abs(ab - 3611483136 * 1.3252) < 1e6, "not params x width"
    assert abs(ab - 7043000000) > 1e9, "still the resident total"


def test_the_source_says_which_quantity_it_used():
    """A reader auditing a ceiling has to be able to tell a width from a total."""
    t = PT.compute_target({**_BASE, "device_weight_bytes": 7043000000, "bytes_per_param": 1.3252}, _HW)
    src = getattr(t, "bytes_source", "")
    assert "B/param" in src, src
    assert "resident" not in src, "still reporting the resident total: %s" % src


def test_an_incomplete_census_is_still_refused():
    """Too few bytes reads as too HIGH a ceiling -- the direction that ends a run early believing it
    is at the wall. An incomplete census must not be used as a lower bound."""
    t = PT.compute_target(
        {**_BASE, "device_weight_bytes": 7043000000, "bytes_per_param": 1.3252, "device_census_complete": False}, _HW
    )
    ab = getattr(t, "active_bytes", 0)
    assert abs(ab - 3611483136 * 1.3252) > 1e6, "an incomplete census was used anyway"


def test_no_census_still_falls_back_to_the_declared_dtype():
    """Before any measurement there is still a declared width, and a ceiling must exist."""
    t = PT.compute_target({k: v for k, v in _BASE.items()}, _HW)
    assert getattr(t, "active_bytes", 0) > 0


def test_the_width_outranks_the_total_when_both_are_present():
    """Not a ban on the total -- where the census walks the whole model the two agree. The width
    simply wins, because it is the one that survives a capped build."""
    t = PT.compute_target({**_BASE, "device_weight_bytes": 7043000000, "bytes_per_param": 1.3252}, _HW)
    assert abs(getattr(t, "active_bytes", 0) - 3611483136 * 1.3252) < 1e6


def test_the_total_still_serves_facts_that_state_no_width():
    """gemma-3's recorded case: 11.9 GB resident, no bytes_per_param, and the ceiling must still be
    43.0 rather than falling back to a predicted width."""
    gem = {
        "total_params": 11180446320,
        "weight_bytes": 24374793024,
        "dominant_dtype": "bfloat16",
        "device_weight_bytes": int(11.9e9),
        "device_census_complete": True,
    }
    r = PT.compute_target(gem, {"dram_bw_gbps": 512.0}).theoretical_rate
    assert abs(r - 43.0) < 0.5, r


def test_the_two_roads_agree_on_a_full_depth_census():
    """Why the fallback is safe: on a census that walked everything, total and params x width are
    the same number. The fix changes voxtral and leaves gemma alone."""
    E, P = {"dram_bw_gbps": 512.0}, 11180446320
    by_total = PT.compute_target(
        {"total_params": P, "device_weight_bytes": int(11.9e9), "device_census_complete": True}, E
    ).theoretical_rate
    by_width = PT.compute_target(
        {"total_params": P, "bytes_per_param": 1.0625, "device_census_complete": True}, E
    ).theoretical_rate
    assert abs(by_total - by_width) / by_width < 0.02, (by_total, by_width)

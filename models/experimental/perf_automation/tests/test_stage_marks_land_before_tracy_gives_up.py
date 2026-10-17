"""Stage marks must be emitted while tracy is still instrumenting.

Tracy stops after 32K source locations -- "Instrumentation failure: Too many source locations" --
saves what it has, and records nothing further. A model forward exhausts that budget long before it
finishes, so marks appended after it were emitted (tracy's own logger printed all seven) into a
capture that had already closed. The profile carried a single `start` signpost, stage_buckets came
back empty, and every stack shared one math-fidelity peak with a note that blamed nothing in
particular. Reproduced on device: 1 signpost with the marks after the bulk loop, 7 with them before.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

from agent.probes import detect_capture_truncated, detect_marker_drop  # noqa: E402
from agent.stage_marks import _function_body_end, _mark_pass_site  # noqa: E402
from agent.tracy_tool import _capture_truncated_reason  # noqa: E402

_WITH_LOOP = """
def _eager_forward():
    pipe = build()
    batches = prepare(pipe)
    out = None
    for batch in batches:
        out = pipe.run_head(batch)
    return out
"""

_NO_LOOP = """
def _eager_forward():
    pipe = build()
    out = pipe.run_once()
    return out
"""


def _line_of(src, idx):
    return src.splitlines()[idx].strip()


def test_the_pass_goes_before_the_bulk_loop_not_after_it():
    idx, indent = _mark_pass_site(_WITH_LOOP, "_eager_forward")
    assert _line_of(_WITH_LOOP, idx).startswith("for "), _line_of(_WITH_LOOP, idx)
    assert indent == " " * 4, "the pass must be a sibling of the loop, not inside it"


def test_a_pass_with_no_loop_keeps_the_previous_site():
    """Additive: a body small enough not to exhaust tracy behaves exactly as before."""
    assert _mark_pass_site(_NO_LOOP, "_eager_forward") == _function_body_end(_NO_LOOP, "_eager_forward")


def test_a_loop_nested_in_a_block_keeps_its_own_indent():
    src = """
def _eager_forward():
    pipe = build()
    with ctx():
        for batch in batches:
            pipe.run_head(batch)
    return None
"""
    idx, indent = _mark_pass_site(src, "_eager_forward")
    assert _line_of(src, idx).startswith("for ")
    assert indent == " " * 8, "a loop inside a with-block sits deeper than the function body"


def test_an_unparseable_or_missing_body_is_survivable():
    assert _mark_pass_site("def broken(:", "broken") == (None, "")
    assert _mark_pass_site(_WITH_LOOP, "not_a_function") == (None, "")


def test_tracy_giving_up_is_not_the_device_dropping_markers():
    """They must not share a verdict: marker drops FAIL a profile outright, and a model whose
    forward always exhausts tracy's budget would then never profile again."""
    ceiling = "Instrumentation failure: Too many source locations. You cannot have more than 32K"
    drop = "Profiler DRAM buffers were full, markers were dropped!"
    assert detect_capture_truncated(ceiling) and detect_marker_drop(ceiling) is None
    assert detect_marker_drop(drop) and detect_capture_truncated(drop) is None


def test_a_clean_log_trips_neither():
    for benign in ("", "1 passed in 92.54s", "Saving trace... done!"):
        assert detect_capture_truncated(benign) is None and detect_marker_drop(benign) is None


def test_the_reason_is_read_from_the_runs_own_log(tmp_path):
    assert _capture_truncated_reason(tmp_path) is None, "no log means no claim"
    (tmp_path / "run0_tracy.log").write_text("Instrumentation failure: Too many source locations\n")
    assert _capture_truncated_reason(tmp_path) == "Instrumentation failure"
    assert _capture_truncated_reason(tmp_path / "nope") is None

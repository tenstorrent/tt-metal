"""The report must describe the code that EXISTS, not every experiment that was run.

Two sections went stale on gemma-3-12b-it's run, in the same way: they described a moment that has
passed and did not say so.

  CODE CHANGES. 3843 of the report's 4005 lines were diffs -- one per attempt, "win or fail". Most
  were reverted the moment they measured, so the report's bulk is source that is NOT in the tree, at
  full length, indistinguishable at a glance from what was kept. A reader scrolling it cannot tell
  which of two contradictory versions of get_mlp_ff2_prg_config is live. What belongs in a report is
  the diff of what SURVIVED; the rest is a lab notebook, and it can be summarised in a line.

The ROW stays for every attempt -- the matrix, the per-attempt table and this list must agree, and
omitting a tried-and-reverted lever entirely would read as "never tried" and invite the next run to
re-derive it. Only the diff BODY is dropped, and the line says so.

The op breakdown was the other candidate and is deliberately left alone: it is computed from a
profile file perf_mcp rewrites during the run, so labelling it "BASELINE" would be a provenance claim
the renderer cannot verify -- exactly what
test_headline_comparability.py::test_sections_say_which_profile_they_came_from removed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize import summary as sm  # noqa: E402

DIFF_A = "diff --git a/m.py b/m.py\n+++ kept change\n"
DIFF_B = "diff --git a/m.py b/m.py\n+++ reverted change\n"


def _attempt(op, kind, diff, ms=100.0, **kw):
    a = {"op_signature": op, "kernel_kind": kind, "diff": diff, "measured_ms": ms}
    a.update(kw)
    return a


def _render(attempts, tmp_path, **kw):
    klog = tmp_path / "kl.json"
    import json as _json

    klog.write_text(_json.dumps(attempts))
    kwargs = dict(model="m", task="main", finalized=True, metric="device_ms")
    kwargs.update(kw)
    return sm.render_summary(str(klog), **kwargs)


# ---------------------------------------------------------------- code changes


def test_a_reverted_diff_is_not_printed_in_full(tmp_path):
    """The attempt is still recorded in the per-attempt table; its dead source is not reproduced."""
    text = _render(
        [
            _attempt("Matmul A", "grid", DIFF_A, beat_baseline=True, committed=True),
            _attempt("Matmul B", "shard", DIFF_B, beat_baseline=False),
        ],
        tmp_path,
    )
    assert "kept change" in text
    assert "reverted change" not in text, "a reverted diff was reproduced in full"


def test_the_reverted_attempts_are_still_acknowledged(tmp_path):
    """Dropping the diff must not drop the FACT. Silently omitting a tried-and-reverted lever would
    read as 'never tried' and invite the next run to re-derive it."""
    text = _render(
        [
            _attempt("Matmul A", "grid", DIFF_A, beat_baseline=True, committed=True),
            _attempt("Matmul B", "shard", DIFF_B, beat_baseline=False),
        ],
        tmp_path,
    )
    assert "Matmul B" in text, "the reverted attempt vanished from the report entirely"


def test_a_report_with_no_surviving_diff_says_so(tmp_path):
    """Every attempt reverted is a real outcome and must not render as an empty section."""
    text = _render([_attempt("Matmul B", "shard", DIFF_B, beat_baseline=False)], tmp_path)
    assert "reverted change" not in text
    assert "Matmul B" in text


def test_the_omission_is_stated_not_silent(tmp_path):
    """A dropped diff must announce itself, or the section reads as if the attempt made no change."""
    text = _render([_attempt("Matmul B", "shard", DIFF_B, beat_baseline=False)], tmp_path)
    assert "reverted" in text.lower() and "omitted" in text.lower(), text[-400:]


# ---------------------------------------------------------------- op breakdown


# The op table is deliberately NOT labelled with a provenance: perf_mcp rewrites the profile file
# during a run, so naming it "BASELINE" would be a claim the renderer cannot verify. See
# test_headline_comparability.py::test_sections_say_which_profile_they_came_from.


def test_the_op_table_still_ranks_by_device_time(tmp_path):
    """Labelling it must not change what it does."""
    prof = {
        "device_ms": 100.0,
        "buckets": [
            {"id": "eltwise", "device_ms": 10.0, "count": 5, "pct": 10.0, "tags": {}},
            {"id": "matmul", "device_ms": 90.0, "count": 9, "pct": 90.0, "tags": {}},
        ],
    }
    text = _render([], tmp_path, baseline_profile=prof)
    if "Op breakdown" not in text:
        pytest.skip("no op breakdown in this render")
    body = text.split("Op breakdown", 1)[1]
    assert body.index("matmul") < body.index("eltwise"), "ranking order changed"

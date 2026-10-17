"""A stage's heading and its rate unit must answer the same question the same way.

Both ask whether one call retires exactly one item per user. They used to answer it with two
different tests -- the unit compared the stage's item count against the BATCH, the heading against
the literal 1. At batch 1 those agree, and until per-stage item counts existed every stage fell back
to a single item, so the disagreement could not appear. Once real counts arrived a batch-8 recurring
stage retiring 8 rows per call satisfied one and failed the other, and the report printed

    DECODE — per request
        59.1 tok/s/u   ...   56.3 tok/s/u

in adjacent lines. No number was wrong; the heading was.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_summary_for_test", PERF / "cc_optimize" / "summary.py")
_summary = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_summary)
_retires_one_per_user = _summary._retires_one_per_user


def _render(roofs, batch, unit="tok/s/u", step="token"):
    """The two comprehensions the report uses, side by side."""
    out = {}
    for st, rf in roofs.items():
        recurring = _retires_one_per_user(rf, batch)
        out[st] = (
            ("%s - per %s" % (st.upper(), step)) if recurring else ("%s - per request" % st.upper()),
            unit if recurring else "req/s",
        )
    return out


def test_the_case_that_contradicted_itself():
    """Batch 8, a recurring stage retiring 8 items: heading and unit must now agree."""
    rows = _render({"encode": {"tokens": 1500}, "prefill": {"tokens": 512}, "decode": {"tokens": 8}}, 8)
    assert rows["decode"] == ("DECODE - per token", "tok/s/u")
    assert rows["prefill"] == ("PREFILL - per request", "req/s")
    assert rows["encode"] == ("ENCODE - per request", "req/s")


def test_heading_and_unit_never_disagree_across_batches_and_counts():
    """The property, not one example: whichever way it answers, both must answer alike."""
    for batch in (1, 2, 8, 32):
        for tokens in (0, 1, 2, 8, 32, 512, 1500):
            heading, unit = _render({"s": {"tokens": tokens}}, batch)["s"]
            per_item = heading.endswith("per token")
            assert per_item == (unit == "tok/s/u"), (batch, tokens, heading, unit)


def test_batch_one_behaves_exactly_as_before():
    """The old `== 1` test and the new `== batch` test coincide at batch 1 -- no change there."""
    for tokens in (0, 1, 2, 8):
        assert _retires_one_per_user({"tokens": tokens}, 1) == (tokens == 1)


def test_the_word_follows_the_model_not_a_typed_name():
    """A run whose unit is steps must read 'per step', never a stage word this tool chose."""
    rows = _render({"denoise": {"tokens": 4}}, 4, unit="step/s", step="step")
    assert rows["denoise"] == ("DENOISE - per step", "step/s")


def test_missing_or_malformed_stage_data_is_not_recurring():
    for rf in (None, {}, {"tokens": None}, {"tokens": 0}):
        assert _retires_one_per_user(rf, 8) is False

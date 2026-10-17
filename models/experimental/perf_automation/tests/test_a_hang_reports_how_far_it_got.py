# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A wedged trace capture must say WHICH STAGE it died in.

`_extract_error` keeps whitelisted lines when any match and otherwise falls back to the tail of the
log. The per-stage markers were not whitelisted, so they survived only by accident -- via that
fallback -- and `_run_perf_node` appends its own "[perf_test_gen] WEDGE: ..." line before calling it.
That line anchors the whitelist, the fallback is never reached, and the stage evidence is dropped.
The tool's own error message deleted the tool's own diagnosis.

Observed cost: a Qwen-Image-Edit bring-up ran five rounds whose trace capture traced two stages fine
and froze in the third. Every round the agent was told only "trace did not engage", guessed, and
failed identically -- while the log it came from named the stage.
"""

from __future__ import annotations

from models.experimental.perf_automation.agent.perf_test_gen import _extract_error

# A capture that got through two stages and died in the third, as measure_adapter prints it.
_LOG = "\n".join(
    [
        "2026-01-01 00:00:00 | warning | Op | some deprecated arg warning",
        "TRACE_STAGE_BYTES[alpha]=15890066432 ops=68634",
        "TRACE_STAGE_MS[alpha]=68009.80 path=trace+1cq",
        "TRACE_STAGE_ITEMS[alpha]=10368",
        "TRACE_STAGE_BYTES[beta]=22881915648 ops=35692",
        "TRACE_STAGE_MS[beta]=85985.39 path=trace+1cq",
        "TRACE_STAGE_ITEMS[beta]=5568",
        "TRACE_STAGE_BYTES[gamma]=12858104960 ops=33515",
    ]
)
_WEDGE = "\n[perf_test_gen] WEDGE: tracy run made no forward progress for 300s; killed process group\n"


def _stages(text):
    return [ln for ln in _extract_error(text).splitlines() if "TRACE_STAGE" in ln]


def test_a_wedge_still_reports_the_stages_it_completed():
    """THE BUG: appending the WEDGE line used to drop every stage line (measured 3 -> 0)."""
    got = _extract_error(_LOG + _WEDGE)
    assert "TRACE_STAGE_MS[alpha]" in got
    assert "TRACE_STAGE_MS[beta]" in got
    assert "WEDGE" in got, "the wedge itself must still be reported"


def test_the_stage_it_died_in_is_identifiable():
    """A stage with BYTES but no MS is where it froze -- that is the whole diagnosis."""
    got = _extract_error(_LOG + _WEDGE)
    assert "TRACE_STAGE_BYTES[gamma]" in got
    assert "TRACE_STAGE_MS[gamma]" not in got


def test_the_wedge_line_no_longer_suppresses_the_evidence():
    """Same log, with and without the appended wedge, must report the same stages."""
    assert _stages(_LOG) == _stages(_LOG + _WEDGE)


def test_stage_names_are_never_enumerated_in_the_source():
    """Constraint: the marker PREFIX is matched; the names inside the brackets come from the model."""
    import inspect

    from models.experimental.perf_automation.agent import perf_test_gen as P

    src = inspect.getsource(P._extract_error)
    assert "_STAGE_MARKER" in src
    for name in ("vision_encode", "text_encode", "vae_encode", "denoise", "prefill", "decode"):
        assert name not in src, f"{name!r} would hardcode a stage name into the extractor"


def test_the_output_stays_bounded():
    """Keeping more lines must not let a 19 MB log through: the tail cap still applies."""
    flood = "\n".join("TRACE_STAGE_BYTES[s%d]=1 ops=1" % i for i in range(500))
    assert len(_extract_error(flood + _WEDGE).splitlines()) <= 25


def test_a_log_with_a_real_exception_is_unchanged():
    """Anchored errors behave exactly as before -- this only adds to the whitelist."""
    err = "E   RuntimeError: boom\nTraceback (most recent call last)\n"
    got = _extract_error(err)
    assert "RuntimeError: boom" in got


def test_a_log_with_no_anchor_still_falls_back_to_its_tail():
    got = _extract_error("just some output\nand a last line\n")
    assert "and a last line" in got

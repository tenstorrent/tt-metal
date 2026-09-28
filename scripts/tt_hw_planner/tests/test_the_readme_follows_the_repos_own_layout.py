# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The generated README uses this repo's layout, and carries no measurements.

The builder's brief for README.md was one line -- "what each Call does, how to run it, the PCC
numbers" -- which is not what the repo does. Measured over the 39 demos under models/demos: 12 have
no README, and of the 27 that do only 6 use `Platforms > Introduction > Prerequisites > How to Run >
Details`. That is nonetheless the ONLY heading sequence that recurs; the other 21 are each unique.
The convention existed and was simply not written anywhere the builder could read.

The brief also invited exactly the wrong content. `gemma4/README.md` carries a "Tokens/s | TTFT (ms)"
table, and on one bring-up a batch-32 result lived ONLY in a README while the gate had passed on 4
samples. Nothing verifies a README, so a number in one reads as certified when it is not -- the
tool-written RUN_REPORT.md is where measurements belong, and the README now points at it.
"""

from __future__ import annotations

import inspect

from scripts.tt_hw_planner.commands import emit_e2e as E

# The sequence the repo's own demos recur on. Asserted in order, so a reshuffle is caught.
_CONVENTION = ("## Platforms", "## Introduction", "## Prerequisites", "## How to Run", "## Details")


def test_the_repos_own_heading_sequence_is_given_in_order():
    block = E._README_LAYOUT_BLOCK
    positions = [block.find(h) for h in _CONVENTION]
    assert all(p >= 0 for p in positions), f"missing: {[h for h, p in zip(_CONVENTION, positions) if p < 0]}"
    assert positions == sorted(positions), "the headings must be given in the repo's order"


def test_inputs_is_nested_under_details():
    block = E._README_LAYOUT_BLOCK
    assert block.find("### Inputs") > block.find("## Details")


def test_the_trace_replay_result_is_required():
    """The one measurement that belongs: the tool prints it and the gate reads it, so it is not the
    agent's own number. Named by MARKER, so the stage names still come from the replay output."""
    flat = " ".join(E._README_LAYOUT_BLOCK.split())
    assert "### Trace replay" in E._README_LAYOUT_BLOCK
    for marker in ("TRACE_STAGE_MS", "TRACE_PER_TOKEN_MS", "TRACE_HEADLINE_UNIT", "TRACE_NOT_TRACE_CAPABLE"):
        assert marker in flat, f"{marker} must be named so the builder copies the real output"
    assert "one row per stage IT reports" in flat
    assert "never a stage list you type" in flat


def test_tracy_output_and_settings_are_forbidden():
    """Tracy is a debugging artefact and a set of harness knobs -- not demo documentation."""
    flat = " ".join(E._README_LAYOUT_BLOCK.split())
    assert "TRACY PROFILER OUTPUT OR SETTINGS" in flat
    assert "zone levels" in flat and "profiling env vars" in flat


def test_unverified_headline_numbers_are_still_forbidden():
    """Trace replay is allowed because the tool measures it; a hand-written league table is not."""
    flat = " ".join(E._README_LAYOUT_BLOCK.split())
    assert "HEADLINE THROUGHPUT TABLES" in flat
    assert "tokens/s" in flat and "TTFT" in flat
    assert "PCC VALUES" in flat and "never the value it produced" in flat


def test_it_ends_by_pointing_at_the_tool_written_report():
    block = E._README_LAYOUT_BLOCK
    assert "## Results" in block
    assert "[RUN_REPORT.md](RUN_REPORT.md)" in block
    assert block.find("## Results") > block.find("## Details"), "the pointer goes at the END"


def test_the_layout_line_no_longer_asks_for_the_pcc_numbers():
    """The old brief asked for exactly the thing now forbidden."""
    src = inspect.getsource(E._build_agent_prompt)
    assert "what each Call does, how to run it, the PCC numbers" not in src
    assert "THIS REPO'S layout" in src


def test_the_block_reaches_the_builder_prompt():
    assert "_README_LAYOUT_BLOCK" in inspect.getsource(E._build_agent_prompt)


def test_it_names_no_model_or_stage():
    lowered = E._README_LAYOUT_BLOCK.lower()
    for name in ("qwen", "gemma", "llama", "bert", "denoise", "prefill", "vae", "encoder"):
        assert name not in lowered, f"{name!r} would hardcode a model/stage name into the guidance"


def test_the_convention_is_really_the_repos(tmp_path):
    """Guard the premise: if the repo's demos stop using this sequence, this test should be revisited.

    Skipped when the demos are not present (the tool branch carries no model packages)."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "models" / "demos"
    readmes = sorted(root.glob("*/README.md")) if root.is_dir() else []
    if len(readmes) < 5:
        import pytest

        pytest.skip("models/demos READMEs not present in this checkout")
    full = 0
    for f in readmes:
        text = f.read_text(errors="ignore")
        if all(h in text for h in _CONVENTION):
            full += 1
    assert full >= 3, f"only {full} demo READMEs use the sequence this block teaches"

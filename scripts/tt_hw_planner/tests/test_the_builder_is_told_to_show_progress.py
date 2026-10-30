# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The builder is told that a silent test reads as a hang.

`probes._execute` kills a step only when its log has stopped growing -- the right rule, with one
consequence the builder was never told: device work enqueued without blocking gives the host nothing
to print, so a healthy run sits inside one call with a frozen log and is killed for looking dead.
That happened to a Qwen-Image-Edit gate replaying a captured trace, and the repair was a
per-iteration print in THAT MODEL's test -- so the lesson lived in one demo and was lost the moment
it was regenerated. These tests keep it in the tool, where every model gets it.
"""

from __future__ import annotations

import inspect

from models.experimental.perf_automation.agent import probes as PR
from scripts.tt_hw_planner.commands import emit_e2e as E


def test_the_builder_is_warned_that_silence_reads_as_a_hang():
    block = E._progress_prompt_block()
    flat = " ".join(block.split())
    assert "LOG HAS STOPPED GROWING" in flat
    assert "SILENT one is killed" in flat
    assert "print one line per iteration" in flat
    assert "SYNCHRONISE THE DEVICE before the print" in flat


def test_the_threshold_quoted_is_the_watchdogs_own():
    """A number retyped here would drift from the supervisor that enforces it."""
    stall = inspect.signature(PR._execute).parameters["stall_timeout_s"].default
    assert f"{int(stall)}s" in E._progress_prompt_block()


def test_the_guidance_survives_an_unreadable_watchdog(monkeypatch):
    """Discovery is best-effort: without the number the advice must still be given, not dropped."""
    monkeypatch.delattr(PR, "_execute")
    block = E._progress_prompt_block()
    assert "SILENT one is killed" in block and "{stall_s}" not in block


def test_it_names_no_model_or_stage():
    """Constraint: the builder reads its own iteration off the model; we never name it."""
    lowered = " ".join(E._PROGRESS_PROMPT_BLOCK.lower().split())  # the text wraps; match on words
    for name in ("denoise", "diffusion", "vae", "unet", "decode", "prefill", "qwen", "timestep"):
        assert name not in lowered, f"{name!r} would hardcode a stage name into the guidance"
    assert "do not assume what it is called" in lowered


def test_the_block_reaches_the_builder_prompt():
    src = inspect.getsource(E._build_agent_prompt)
    assert "_progress_prompt_block()" in src


def test_the_watchdog_docstring_matches_its_code():
    """The docstring said the watchdog waits for ~no CPU; progress_signature EXCLUDES CPU, and that
    wording is what made the mechanism get described wrongly."""
    doc = inspect.getdoc(PR._execute) or ""
    assert "burned ~no CPU" not in doc
    assert "CPU is deliberately NOT among them" in " ".join(doc.split())
    assert "EXCLUDES CPU" in (inspect.getdoc(PR.progress_signature) or "")

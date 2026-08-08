"""Tests for the SKIP-vs-graduated distinction (2026-06-03).

Background — seamless-m4t bring-up showed that:
  1. 22 of 23 NEW components had auto-PCC tests SKIPPED at the harness
     layer (couldn't synthesize inputs).
  2. The orchestrator silently dropped them from the candidate pool
     via `skipped_components_this_run`.
  3. OUTCOME banner declared "all graduated rc=0" because the queue
     was empty.

Two complementary fixes covered here:
  * `_run_auto_iterate_loop` now tracks harness-pattern SKIPs in a
    distinct `harness_skipped_this_run` set and persists it to
    `<demo_dir>/harness_skipped.json` for the OUTCOME banner.
  * `_final_outcome_banner` reads that file and surfaces a clear
    "HARNESS-SKIPPED" warning so a "SUCCESS rc=0" outcome with 22
    silently-dropped components is distinguishable from one with 0.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_outcome_banner_surfaces_harness_skipped(tmp_path, capsys):
    """End-to-end: when harness_skipped.json exists in the demo_dir,
    the OUTCOME banner must print a HARNESS-SKIPPED warning section."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    (tmp_path / "harness_skipped.json").write_text(
        json.dumps(
            {
                "harness_skipped_components": [
                    "seamless_m4_t_conformer_encoder_layer",
                    "seamless_m4_t_decoder_layer",
                ]
            }
        )
    )

    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out

    assert "HARNESS-SKIPPED" in out, "banner should flag harness-skipped components"
    assert "seamless_m4_t_conformer_encoder_layer" in out
    assert "seamless_m4_t_decoder_layer" in out
    assert "test-fixture fix" in out or "tests/pcc/" in out, "banner should point user to the actionable fix location"


def test_outcome_banner_no_harness_section_when_no_file(tmp_path, capsys):
    """If no harness_skipped.json exists, the banner shouldn't print the
    section at all — keeps healthy runs quiet."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "HARNESS-SKIPPED" not in out


def test_outcome_banner_handles_malformed_harness_file(tmp_path, capsys):
    """If the file exists but isn't valid JSON, the banner must NOT
    crash — degrade silently to no warning."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    (tmp_path / "harness_skipped.json").write_text("not valid json {")

    # Should not raise
    _final_outcome_banner(
        rc=0,
        model_id="test/model",
        path_label="test path",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "HARNESS-SKIPPED" not in out


def test_is_eligible_for_graduation_helper_canonical_check(tmp_path):
    """The canonical _is_eligible_for_graduation helper checks both:
    (1) stub exists on disk, (2) stub is not a torch wrapper."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _is_eligible_for_graduation

    # Case 1: stub doesn't exist
    assert _is_eligible_for_graduation(tmp_path / "nonexistent.py") is False

    # Case 2: torch wrapper stub — not eligible
    wrapper_stub = tmp_path / "wrapper.py"
    wrapper_stub.write_text(
        "import torch\n"
        "class Comp:\n"
        "    def __init__(self, device, torch_module):\n"
        "        self._torch_module = torch_module\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self._torch_module(*args, **kwargs)\n"
    )
    assert _is_eligible_for_graduation(wrapper_stub) is False

    # Case 3: real native stub — eligible
    native_stub = tmp_path / "native.py"
    native_stub.write_text(
        "import ttnn\n"
        "class Comp:\n"
        "    def __init__(self, device, torch_module):\n"
        "        self.device = device\n"
        "    def __call__(self, x):\n"
        "        return ttnn.linear(x, self.weight)\n"
    )
    assert _is_eligible_for_graduation(native_stub) is True

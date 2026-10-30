"""Unit tests for the PCC-trend block in the LLM iter prompt (Gap 5).

The brain already tracks ``pcc_history_per_component`` (list of mismatch
ratios, smaller is better) for convergence detection, but the value was
never surfaced to the LLM in the iter prompt. This left the LLM with
no way to tell whether prior patches moved PCC or stagnated — it kept
re-exploring similar approaches.

This module:
  * ``format_pcc_trend_block`` — pure formatter; renders the last N
    iters' PCC values with a trend tag (improving / stagnant /
    regressing).
  * Plumbed into ``build_per_target_blocks`` (parallel-extra path)
    and ``assemble_iter_prompt`` (primary path).
"""

from __future__ import annotations

import pytest


def test_empty_history_returns_empty_string():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    assert format_pcc_trend_block(target_component="attention", pcc_history=None) == ""
    assert format_pcc_trend_block(target_component="attention", pcc_history=[]) == ""


def test_single_iter_history_emits_first_attempt_tag():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    block = format_pcc_trend_block(target_component="mlp", pcc_history=[0.1])  # 1-0.1=0.9 PCC
    assert "mlp" in block
    assert "first attempt" in block
    assert "0.9000" in block


def test_improving_trend_tag():
    """mismatch ratios go down → PCC goes up → improving."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    # 1 - 0.3 = 0.7, 1 - 0.2 = 0.8, 1 - 0.1 = 0.9  (PCC improving by 0.2)
    block = format_pcc_trend_block(target_component="mlp", pcc_history=[0.3, 0.2, 0.1])
    assert "improving" in block.lower()
    assert "+0.200" in block or "+0.2" in block


def test_stagnant_trend_tag():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    # PCC barely moves (0.85 → 0.851 → 0.85)
    block = format_pcc_trend_block(target_component="mlp", pcc_history=[0.15, 0.149, 0.15])
    assert "STAGNANT" in block
    # Words may span line breaks — check the guidance is present in any wrapping
    flat = " ".join(block.split())  # collapse whitespace
    assert "different approach" in flat.lower()


def test_regressing_trend_tag():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    # PCC went DOWN (0.95 → 0.5)
    block = format_pcc_trend_block(target_component="mlp", pcc_history=[0.05, 0.30, 0.50])
    assert "REGRESSING" in block
    assert "revert" in block.lower()


def test_truncates_to_last_5_iters():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    block = format_pcc_trend_block(
        target_component="mlp",
        pcc_history=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    # First 4 should NOT be rendered (oldest), only last 5
    # 1-0.9=0.1 → would be rendered as 0.1000 if included
    assert "0.1000" not in block, "first iter (PCC=0.1) should be truncated — only last 5 shown"
    # Last 5 PCC values: 1-0.5=0.5, 1-0.4=0.6, 1-0.3=0.7, 1-0.2=0.8, 1-0.1=0.9
    assert "0.5000" in block
    assert "0.9000" in block


def test_block_includes_graduation_target_099():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    block = format_pcc_trend_block(target_component="mlp", pcc_history=[0.2, 0.15])
    assert "0.99" in block, "block must mention the 0.99 graduation target so LLM knows the goal"


def test_build_per_target_blocks_includes_pcc_trend(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import build_per_target_blocks

    blocks = build_per_target_blocks(
        demo_dir=tmp_path,
        target_component="attention",
        per_comp_failure={},
        last_failure_class_per_component={},
        attempts_per_component={},
        focused_stub_excerpts=[],
        pcc_history=[0.2, 0.15, 0.10],
    )
    assert "pcc_trend_block" in blocks
    assert "attention" in blocks["pcc_trend_block"]
    assert "improving" in blocks["pcc_trend_block"].lower()


def test_build_per_target_blocks_empty_when_no_history(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import build_per_target_blocks

    blocks = build_per_target_blocks(
        demo_dir=tmp_path,
        target_component="attention",
        per_comp_failure={},
        last_failure_class_per_component={},
        attempts_per_component={},
        focused_stub_excerpts=[],
    )
    assert "pcc_trend_block" in blocks
    assert blocks["pcc_trend_block"] == ""


def test_assemble_iter_prompt_includes_pcc_trend_block():
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import assemble_iter_prompt

    out = assemble_iter_prompt(
        hw_header="HW\n",
        task_block="TASK\n",
        systemic_block="",
        shape_probe_block="",
        agentic_block="",
        budget_clause="",
        failure_context="FAILURE\n",
        strategy_directive="DO X",
        escalated_scope_block="",
        native_directive="NATIVE",
        cross_component_block="",
        components_block="COMP",
        pcc_trend_block="\nPCC TREND: 0.85 → 0.86\n",
    )
    assert "PCC TREND: 0.85 → 0.86" in out


def test_assemble_iter_prompt_backward_compatible():
    """Existing callers that didn't pass pcc_trend_block still work."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import assemble_iter_prompt

    # Call without pcc_trend_block — should default to empty string
    out = assemble_iter_prompt(
        hw_header="HW\n",
        task_block="TASK\n",
        systemic_block="",
        shape_probe_block="",
        agentic_block="",
        budget_clause="",
        failure_context="FAILURE\n",
        strategy_directive="DO X",
        escalated_scope_block="",
        native_directive="NATIVE",
        cross_component_block="",
        components_block="COMP",
    )
    assert "TASK" in out
    assert "DO X" in out


def test_phi35_attention_scenario_renders_stagnant():
    """End-to-end sanity check using realistic numbers from the Phi-3.5
    attention failure: every iter had no PCC computed (forward never ran)
    so the mismatch ratio was effectively saturated at 1.0. The block
    should clearly flag this as not converging."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import format_pcc_trend_block

    # Simulate "PCC always at floor" pattern
    block = format_pcc_trend_block(
        target_component="attention",
        pcc_history=[0.99, 0.99, 0.99, 0.99],
    )
    assert "STAGNANT" in block
    assert "attention" in block

"""Pin: the emit-e2e builder contract instructs a model-grounded decode horizon.

Regression guard for the "magic N=40" defect — the builder must be told to
derive the autoregressive decode length from the model (stop token /
generation_config), apply the same rule to TT and the golden, and only fall
back to an LLM-chosen bound when no signal exists — without making trace
capture dynamic.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.emit_e2e import _TT_ONLY_CONTRACT  # noqa: E402


def test_contract_has_decode_horizon_section() -> None:
    assert "DECODE HORIZON" in _TT_ONLY_CONTRACT


def test_contract_forbids_magic_constant() -> None:
    c = _TT_ONLY_CONTRACT
    assert "magic" in c and "N=40" in c
    assert "NOT an arbitrary hardcoded constant" in c


def test_contract_prioritizes_stop_token_then_config_then_llm() -> None:
    c = _TT_ONLY_CONTRACT
    assert "STOP-TOKEN" in c
    assert "eos_token_id" in c
    assert "generation_config.max_new_tokens" in c
    assert "LLM FALLBACK" in c
    assert c.index("STOP-TOKEN") < c.index("LLM FALLBACK")


def test_contract_requires_safety_cap_and_same_rule_for_golden() -> None:
    c = _TT_ONLY_CONTRACT
    assert "safety cap" in c
    assert "SAME stop rule" in c and "HF golden" in c


def test_contract_keeps_trace_capture_fixed_capacity() -> None:
    c = _TT_ONLY_CONTRACT
    assert "FIXED max capacity" in c
    assert "PCC/correctness test only" in c

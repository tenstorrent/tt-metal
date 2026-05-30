"""Tests for the precision critic sub-agent.

The critic runs after a PCC-plateau failure: it takes the failing code
+ PCC value, runs a focused LLM call, and returns one structured
diagnosis. These tests pin parsing, caching, error handling, and the
prompt-injection format — without ever hitting a real LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.critic import (  # noqa: E402
    CriticDiagnosis,
    clear_cache,
    invoke_critic,
    load_diagnosis,
    persist_diagnosis,
)
from scripts.tt_hw_planner.agentic.critic import _parse_critic_response  # noqa: E402


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_picks_up_well_formed_fields() -> None:
    response = (
        "ROOT_CAUSE: ttnn.sum over the spatial reduction uses bf16 accumulation.\n"
        "SPECIFIC_CHANGE: Set fp32_dest_acc_en=True on the reduction kernel config.\n"
        "WHY: bf16 accumulation truncates each partial sum, costing ~0.01 PCC.\n"
        "ALTERNATIVE: Switch MathFidelity from LoFi to HiFi4 for the reduction.\n"
        "CONFIDENCE: high\n"
    )
    fields = _parse_critic_response(response)
    assert fields["ROOT_CAUSE"].startswith("ttnn.sum")
    assert "fp32_dest_acc_en" in fields["SPECIFIC_CHANGE"]
    assert "bf16 accumulation truncates" in fields["WHY"]
    assert "HiFi4" in fields["ALTERNATIVE"]
    assert fields["CONFIDENCE"] == "high"


def test_parse_handles_code_fence_wrapping() -> None:
    """LLM ignored 'no code blocks' instruction — strip the fence."""
    response = (
        "```\n"
        "ROOT_CAUSE: example cause\n"
        "SPECIFIC_CHANGE: example change\n"
        "WHY: because\n"
        "ALTERNATIVE: other\n"
        "CONFIDENCE: medium\n"
        "```"
    )
    fields = _parse_critic_response(response)
    assert fields.get("ROOT_CAUSE") == "example cause"
    assert fields.get("CONFIDENCE") == "medium"


def test_parse_handles_multiline_field_values() -> None:
    """A field value can spill onto the next line until the next KEY:"""
    response = (
        "ROOT_CAUSE: first line\n"
        "  continuing on second line\n"
        "SPECIFIC_CHANGE: do thing\n"
        "WHY: because\n"
        "ALTERNATIVE: -\n"
        "CONFIDENCE: low\n"
    )
    fields = _parse_critic_response(response)
    assert "first line" in fields["ROOT_CAUSE"]
    assert "continuing on second line" in fields["ROOT_CAUSE"]
    assert fields["SPECIFIC_CHANGE"] == "do thing"


def test_parse_tolerates_extra_commentary() -> None:
    """If the LLM prepends a line before the structured fields, parsing
    must still pick up the named fields."""
    response = (
        "Sure, here is the diagnosis:\n"
        "ROOT_CAUSE: a\n"
        "SPECIFIC_CHANGE: b\n"
        "WHY: c\n"
        "ALTERNATIVE: d\n"
        "CONFIDENCE: low\n"
    )
    fields = _parse_critic_response(response)
    assert fields["ROOT_CAUSE"] == "a"
    assert fields["SPECIFIC_CHANGE"] == "b"


# ---------------------------------------------------------------------------
# invoke_critic — happy path + caching
# ---------------------------------------------------------------------------


def _mock_llm_factory(response: str, calls: list):
    def _mock(*, prompt: str, agent_bin: str, model: str, timeout_s: int):
        calls.append(prompt)
        return response

    return _mock


def test_invoke_critic_parses_response() -> None:
    clear_cache()
    calls: list = []
    llm = _mock_llm_factory(
        (
            "ROOT_CAUSE: bf16 reduction\n"
            "SPECIFIC_CHANGE: set fp32_dest_acc_en=True\n"
            "WHY: increases precision\n"
            "ALTERNATIVE: use HiFi4\n"
            "CONFIDENCE: high\n"
        ),
        calls,
    )
    diag = invoke_critic(
        component="video_mask_down_sampler",
        code="def fwd(): pass",
        pcc=0.9877,
        _call_llm=llm,
    )
    assert diag.confidence == "high"
    assert diag.specific_change == "set fp32_dest_acc_en=True"
    assert diag.is_actionable
    assert len(calls) == 1


def test_invoke_critic_cache_hit_skips_llm() -> None:
    """Same component + same code + same pcc → no second LLM call."""
    clear_cache()
    calls: list = []
    llm = _mock_llm_factory(
        "ROOT_CAUSE: x\nSPECIFIC_CHANGE: y\nWHY: z\nALTERNATIVE: w\nCONFIDENCE: medium\n",
        calls,
    )
    invoke_critic(component="c", code="def fwd(): pass", pcc=0.95, _call_llm=llm)
    invoke_critic(component="c", code="def fwd(): pass", pcc=0.95, _call_llm=llm)
    assert len(calls) == 1, "second invocation must hit the cache, not the LLM"


def test_invoke_critic_cache_miss_when_code_changes() -> None:
    clear_cache()
    calls: list = []
    llm = _mock_llm_factory(
        "ROOT_CAUSE: x\nSPECIFIC_CHANGE: y\nWHY: z\nALTERNATIVE: w\nCONFIDENCE: medium\n",
        calls,
    )
    invoke_critic(component="c", code="def fwd_v1(): pass", pcc=0.95, _call_llm=llm)
    invoke_critic(component="c", code="def fwd_v2(): pass", pcc=0.95, _call_llm=llm)
    assert len(calls) == 2


def test_invoke_critic_llm_failure_returns_error_diagnosis() -> None:
    """If the LLM call raises, the critic must NOT crash the loop — it
    returns a non-actionable diagnosis with the error recorded."""
    clear_cache()

    def _explodes(**kwargs):
        raise RuntimeError("LLM unreachable")

    diag = invoke_critic(component="c", code="def fwd(): pass", pcc=0.5, _call_llm=_explodes)
    assert not diag.is_actionable
    assert "LLM unreachable" in diag.error


def test_invoke_critic_confidence_clamped_to_known_values() -> None:
    clear_cache()
    llm = _mock_llm_factory(
        "ROOT_CAUSE: a\nSPECIFIC_CHANGE: b\nWHY: c\nALTERNATIVE: d\nCONFIDENCE: VERY_HIGH\n",
        [],
    )
    diag = invoke_critic(component="c", code="x", pcc=0.9, _call_llm=llm)
    assert diag.confidence == "low"  # unknown values clamp to low


# ---------------------------------------------------------------------------
# Prompt-block rendering
# ---------------------------------------------------------------------------


def test_to_prompt_block_empty_when_not_actionable() -> None:
    diag = CriticDiagnosis(component="c", pcc=0.5)
    assert diag.to_prompt_block() == ""


def test_to_prompt_block_contains_all_fields() -> None:
    diag = CriticDiagnosis(
        component="c",
        pcc=0.9877,
        root_cause="r",
        specific_change="s",
        why="w",
        alternative="a",
        confidence="high",
    )
    block = diag.to_prompt_block()
    assert "PRECISION CRITIC DIAGNOSIS" in block
    assert "0.9877" in block
    assert "ROOT_CAUSE: r" in block
    assert "SPECIFIC_CHANGE: s" in block
    assert "WHY: w" in block
    assert "ALTERNATIVE: a" in block
    assert "CONFIDENCE: high" in block


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persist_then_load_round_trips(tmp_path: Path) -> None:
    diag = CriticDiagnosis(
        component="c1",
        pcc=0.98,
        root_cause="rc",
        specific_change="sc",
        why="w",
        alternative="alt",
        confidence="medium",
    )
    persist_diagnosis(diag, tmp_path)
    loaded = load_diagnosis("c1", tmp_path)
    assert loaded is not None
    assert loaded.specific_change == "sc"
    assert loaded.confidence == "medium"


def test_load_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert load_diagnosis("missing", tmp_path) is None


def test_load_tolerates_malformed_json(tmp_path: Path) -> None:
    out = tmp_path / "_critic"
    out.mkdir()
    (out / "c.json").write_text("not json")
    assert load_diagnosis("c", tmp_path) is None

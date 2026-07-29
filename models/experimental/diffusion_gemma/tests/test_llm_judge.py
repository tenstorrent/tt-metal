# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the LLM-judge gate tool. Nothing here talks to the network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "doc" / "decision_fidelity" / "gate" / "llm_judge.py"


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("dg_llm_judge", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = {"model": "claude-opus-5", "effort": "low", "votes": 1, "max_chars": 24000}
ITEM = {
    "question": "Which one?",
    "choices": ["alpha", "beta", "gamma particle", "delta"],
    "gold_letter": "C",
    "text": "Reasoning. The answer is (C).",
}
VERDICT = {
    "meaningful": True,
    "failure_mode": "none",
    "language": "en",
    "answered": True,
    "selected_letter": "C",
    "selected_answer_text": "gamma particle",
    "notes": "states an answer",
}


def _resp(payload=None, with_thinking=True):
    """A Messages response: thinking block first (thinking is on by default), then the verdict text."""
    blocks = [SimpleNamespace(type="thinking", thinking="")] if with_thinking else []
    blocks.append(SimpleNamespace(type="text", text=json.dumps(payload if payload is not None else VERDICT)))
    return SimpleNamespace(content=blocks, stop_reason="end_turn", stop_details=None)


def _client(response=None, capture=None):
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return response if response is not None else _resp()

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def _raising_client(exc):
    def create(**_kwargs):
        raise exc

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_truncate_keeps_the_tail_where_the_answer_lives(judge):
    got = judge.truncate("H" * 5000 + "ANSWER IS (B)", 1000)
    assert "ANSWER IS (B)" in got and "elided" in got and len(got) < 1200


def test_prompt_never_reveals_which_choice_is_correct(judge):
    prompt = judge.prompt_for(ITEM, 24000)
    assert "gold" not in prompt.lower()
    assert "(C) gamma particle" in prompt  # offered as a plain option, unmarked


def test_request_omits_temperature_and_sets_the_schema(judge):
    """Opus 5 returns a 400 for temperature/top_p/top_k, and the verdict must be schema-constrained."""
    sent = {}
    judge.judge_one(_client(capture=sent), ITEM, CFG)
    assert not {"temperature", "top_p", "top_k"} & sent.keys()
    assert sent["output_config"] == {"effort": "low", "format": judge.SCHEMA}
    assert sent["fallbacks"] == "default" and sent["model"] == "claude-opus-5"


def test_verdict_is_read_past_the_thinking_block(judge):
    """With thinking on, the verdict is not content[0]."""
    assert judge.judge_one(_client(), ITEM, CFG)["selected_letter"] == "C"


def test_a_refusal_is_not_treated_as_a_verdict(judge):
    """A refusal is a successful HTTP 200 -- indexing content blindly is what breaks."""
    refused = SimpleNamespace(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))
    with pytest.raises(RuntimeError, match="declined"):
        judge.judge_one(_client(refused), ITEM, CFG)


def test_a_bogus_letter_is_dropped(judge):
    got = judge.judge_one(_client(_resp({**VERDICT, "selected_letter": "Z"})), ITEM, CFG)
    assert got["selected_letter"] is None


def test_empty_text_is_settled_locally_with_no_call(judge):
    client = _raising_client(AssertionError("should not call the API for empty text"))
    got = judge.judge_item(client, {"text": "  \n "}, CFG)
    assert got["failure_mode"] == "empty" and not got["meaningful"]


def test_a_failing_item_does_not_lose_the_run(judge):
    got = judge.judge_item(_raising_client(RuntimeError("500 boom")), ITEM, CFG)
    assert "500 boom" in got["error"]


def test_votes_take_the_majority_and_flag_the_split(judge):
    dissent = {**VERDICT, "meaningful": False, "failure_mode": "repetition", "answered": False, "selected_letter": None}
    got = judge.majority([VERDICT, VERDICT, dissent])
    assert got["meaningful"] and got["selected_letter"] == "C" and got["failure_mode"] == "none"
    assert got["split"] is True and got["votes"] == 3


def test_judge_letter_falls_back_to_matching_the_answer_text(judge):
    """The text path is what makes grading shuffle-independent."""
    verdict = {"selected_letter": None, "selected_answer_text": "the gamma particle"}
    assert judge.judge_letter(verdict, ITEM) == "C"


def test_judge_letter_credits_nothing_when_the_response_never_answered(judge):
    """Deriving a letter from a non-answer is the regex extractor's stage-3 mistake."""
    verdict = {"answered": False, "selected_letter": None, "selected_answer_text": "gamma particle"}
    assert judge.judge_letter(verdict, ITEM) is None


def test_judge_letter_refuses_an_ambiguous_text_match(judge):
    item = {**ITEM, "choices": ["alpha", "alpha decay", "gamma", "delta"]}
    assert judge.judge_letter({"selected_letter": None, "selected_answer_text": "alpha"}, item) is None


def test_load_reads_one_row_per_question_not_per_filter(judge, tmp_path):
    path = tmp_path / "samples_gpqa.jsonl"
    doc = {"Record ID": "rec-9", "Question": "Q?", "choices": ["a", "b", "c", "d"], "answer": "(B)"}
    rows = [
        {"filter": "flexible-extract", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 1.0},
        {"filter": "strict-match", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 0.0},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _src, items = judge.load(path, "full")
    assert len(items) == 1 and items[0]["gold_letter"] == "B" and items[0]["regex_correct"] is True


def test_report_counts_what_the_regex_laundered(judge, capsys):
    items = [{**ITEM, "text": "(C) appears in prose", "regex_correct": True}, {**ITEM, "regex_correct": True}]
    verdicts = [{**VERDICT, "answered": False, "selected_letter": None}, VERDICT]
    judge.report("src", items, verdicts, CFG)
    out = capsys.readouterr().out
    assert "1 response(s) it scored correct that never answered" in out
    assert "correct: 1/2" in out
    assert "1 disagreement(s)" in out

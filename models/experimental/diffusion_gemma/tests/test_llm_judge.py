# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the LLM-judge gate tool, with the proxy stubbed out.

The judge's value is that it replaces regexes, so the tests that matter are the ones covering the
seams around the model call: the prompt keeps the answer-bearing TAIL, gold never reaches the judge,
a selected answer given as TEXT still resolves to a letter, empty responses cost nothing, and one bad
item does not lose the run. Nothing here talks to the network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "doc" / "decision_fidelity" / "gate" / "llm_judge.py"


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("dg_llm_judge", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(judge, **over):
    cfg = {"model": "azure/gpt-4o", "base": "https://x", "key": "k", "votes": 1, "max_chars": 24000, "timeout": 5.0}
    cfg.update(over)
    return cfg


def _item(text="Let me think. The answer is (C).", **over):
    item = {
        "id": "rec1",
        "question": "Which one?",
        "choices": ["alpha", "beta", "gamma particle", "delta"],
        "gold_letter": "C",
        "gold_text": "gamma particle",
        "text": text,
    }
    item.update(over)
    return item


def test_truncate_keeps_the_tail_where_the_answer_lives(judge):
    text = "H" * 5000 + "THE ANSWER IS (B)"
    got = judge.truncate(text, 1000)
    assert "THE ANSWER IS (B)" in got
    assert "elided" in got
    assert len(got) < 1200


def test_short_text_is_not_truncated(judge):
    assert judge.truncate("short", 1000) == "short"


def test_prompt_never_contains_gold(judge):
    """Gold in the prompt would let the judge launder a non-answer into the right letter."""
    prompt = judge.build_user_prompt(_item(text="I am not sure."), 24000)
    assert "gold" not in prompt.lower()
    # The correct-answer TEXT is also a choice, so its presence is expected; what must be absent is
    # any marking of WHICH choice is correct.
    assert "Correct Answer" not in prompt
    assert "(C) gamma particle" in prompt  # offered as a plain option, unmarked


def test_prompt_letters_follow_choice_order(judge):
    prompt = judge.build_user_prompt(_item(), 24000)
    for letter, choice in zip("ABCD", ["alpha", "beta", "gamma particle", "delta"]):
        assert f"({letter}) {choice}" in prompt


def test_no_choices_tells_the_judge_to_leave_the_letter_null(judge):
    prompt = judge.build_user_prompt({"text": "hello"}, 24000)
    assert "selected_letter" in prompt and "null" in prompt


@pytest.mark.parametrize(
    "content",
    [
        '{"meaningful": true, "failure_mode": "none", "answered": true, "selected_letter": "C"}',
        '```json\n{"meaningful": true, "failure_mode": "none", "answered": true, "selected_letter": "(C)"}\n```',
        'Sure:\n{"meaningful": true, "failure_mode": "none", "answered": true, "selected_letter": "c"}',
    ],
)
def test_parse_verdict_tolerates_fences_and_prose_and_normalizes_letters(judge, content):
    got = judge.parse_verdict(content)
    assert got["meaningful"] and got["answered"] and got["selected_letter"] == "C"


def test_parse_verdict_survives_a_missing_letter(judge):
    """The common case: a response that never selected anything. Must not raise, must not invent."""
    got = judge.parse_verdict('{"meaningful": true, "failure_mode": "none", "answered": false}')
    assert got["selected_letter"] is None and got["answered"] is False


@pytest.mark.parametrize("raw", ["null", '""', '"  "', '"Z"', '"none"'])
def test_parse_verdict_never_reports_a_bogus_letter(judge, raw):
    got = judge.parse_verdict('{"meaningful": true, "failure_mode": "none", "selected_letter": %s}' % raw)
    assert got["selected_letter"] is None


def test_parse_verdict_rejects_non_json(judge):
    with pytest.raises(ValueError):
        judge.parse_verdict("I think it is fine, honestly.")


def test_parse_verdict_maps_unknown_failure_mode_to_other(judge):
    got = judge.parse_verdict('{"meaningful": false, "failure_mode": "spaghetti"}')
    assert got["failure_mode"] == "other"


def test_derive_letter_prefers_the_explicit_letter(judge):
    got = judge.derive_letter({"selected_letter": "B", "selected_answer_text": "gamma particle"}, _item())
    assert got == "B"


def test_derive_letter_falls_back_to_matching_the_answer_text(judge):
    """The text path is what makes grading shuffle-independent."""
    got = judge.derive_letter({"selected_letter": None, "selected_answer_text": "the gamma particle"}, _item())
    assert got == "C"


def test_derive_letter_refuses_an_ambiguous_text_match(judge):
    item = _item(choices=["alpha", "alpha decay", "gamma", "delta"])
    assert judge.derive_letter({"selected_letter": None, "selected_answer_text": "alpha"}, item) is None


def test_empty_response_is_settled_locally_with_no_api_call(judge, monkeypatch):
    calls = []
    monkeypatch.setattr(judge, "judge_call", lambda *a, **k: calls.append(a) or {})
    got = judge.judge_item(_item(text="   \n "), _cfg(judge), judge.Cache(None))
    assert calls == []
    assert got["failure_mode"] == "empty" and not got["meaningful"]


def test_cache_avoids_a_second_call_and_persists(judge, tmp_path, monkeypatch):
    calls = []

    def fake(item, cfg, vote):
        calls.append(vote)
        return {**judge.parse_verdict('{"meaningful": true, "failure_mode": "none", "answered": true}'), "_usage": {}}

    monkeypatch.setattr(judge, "judge_call", fake)
    path = tmp_path / "cache.json"
    cache = judge.Cache(path)
    judge.judge_item(_item(), _cfg(judge), cache)
    judge.judge_item(_item(), _cfg(judge), cache)
    cache.flush()
    assert len(calls) == 1 and cache.hits == 1 and cache.writes == 1
    assert judge.Cache(path).data  # survived the round trip


def test_cache_key_changes_with_the_prompt_version(judge, monkeypatch):
    before = judge.Cache.key(_item(), _cfg(judge), 0)
    monkeypatch.setattr(judge, "PROMPT_VERSION", "v999")
    assert judge.Cache.key(_item(), _cfg(judge), 0) != before


def test_a_failing_item_does_not_lose_the_run(judge, monkeypatch):
    monkeypatch.setattr(judge, "judge_call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 500: boom")))
    got = judge.judge_item(_item(), _cfg(judge), judge.Cache(None))
    assert "HTTP 500" in got["error"]


def test_votes_take_the_majority_and_flag_the_split(judge, monkeypatch):
    seq = iter(
        [
            {"meaningful": True, "failure_mode": "none", "answered": True, "selected_letter": "C", "_usage": {}},
            {"meaningful": True, "failure_mode": "none", "answered": True, "selected_letter": "C", "_usage": {}},
            {
                "meaningful": False,
                "failure_mode": "repetition",
                "answered": False,
                "selected_letter": None,
                "_usage": {},
            },
        ]
    )
    monkeypatch.setattr(judge, "judge_call", lambda *a, **k: next(seq))
    got = judge.judge_item(_item(), _cfg(judge, votes=3), judge.Cache(None))
    assert got["meaningful"] and got["selected_letter"] == "C" and got["failure_mode"] == "none"
    assert got["_split"] is True and got["_votes"] == 3


def test_report_counts_the_laundered_regex_credits(judge):
    """The headline number: a response the regex scored correct but that never answered."""
    items = [
        _item(text="reasoning ... (C) appears in prose", regex_correct=True),
        _item(text="The answer is (C).", regex_correct=True),
    ]
    verdicts = [
        {"meaningful": True, "failure_mode": "none", "language": "en", "answered": False, "selected_letter": None},
        {"meaningful": True, "failure_mode": "none", "language": "en", "answered": True, "selected_letter": "C"},
    ]
    text, rate = judge.report("src", items, verdicts, _cfg(judge), judge.Cache(None))
    assert "regex credited a response that never answered:  1" in text
    assert "correct (judge-selected letter vs gold): 1/2" in text
    assert rate == 1.0


def test_report_excludes_errors_and_says_so(judge):
    items = [_item(), _item()]
    verdicts = [
        {"meaningful": True, "failure_mode": "none", "language": "en", "answered": True, "selected_letter": "C"},
        {"error": "HTTP 500: boom"},
    ]
    text, rate = judge.report("src", items, verdicts, _cfg(judge), judge.Cache(None))
    assert "ERRORED: 1" in text and "denominators are short" in text
    assert "meaningful:  1/1" in text and rate == 1.0


def test_load_samples_reads_one_row_per_question_not_per_filter(judge, tmp_path):
    path = tmp_path / "samples_gpqa.jsonl"
    doc = {
        "Record ID": "rec-9",
        "Question": "Q?",
        "choices": ["a", "b", "c", "d"],
        "answer": "(B)",
        "Correct Answer": "b",
    }
    rows = [
        {"filter": "flexible-extract", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 1.0},
        {"filter": "strict-match", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 0.0},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _src, items = judge.load_samples(path, "full")
    assert len(items) == 1
    assert items[0]["id"] == "rec-9" and items[0]["gold_letter"] == "B" and items[0]["regex_correct"] is True


def test_load_plain_handles_jsonl_and_whole_file(judge, tmp_path):
    jl = tmp_path / "r.jsonl"
    jl.write_text('{"text": "one"}\n{"response": "two"}\n{"resps": [["three"]]}\n')
    _src, items = judge.load_plain(jl)
    assert [i["text"] for i in items] == ["one", "two", "three"]

    txt = tmp_path / "r.txt"
    txt.write_text("just prose\nover two lines\n")
    _src, single = judge.load_plain(txt)
    assert len(single) == 1 and "two lines" in single[0]["text"]


def test_http_json_retries_5xx_then_succeeds(judge, monkeypatch):
    import urllib.error

    attempts = []

    class FakeResp:
        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        attempts.append(req.get_method())
        if len(attempts) < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return FakeResp()

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    assert judge.http_json("https://x/chat/completions", "k", {"a": 1})["ok"] is True
    assert len(attempts) == 3 and attempts[0] == "POST"


def test_http_json_does_not_retry_a_bad_request(judge, monkeypatch):
    import io
    import urllib.error

    attempts = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error":"unknown model"}'))

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="unknown model"):
        judge.http_json("https://x/chat/completions", "k", {"a": 1})
    assert len(attempts) == 1


def test_get_request_when_there_is_no_payload(judge, monkeypatch):
    seen = {}

    class FakeResp:
        def read(self):
            return b'{"data": [{"id": "azure/gpt-4o"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        return FakeResp()

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    assert judge.list_models("https://x", "secret") == 0
    assert seen["method"] == "GET" and seen["auth"] == "Bearer secret"


def test_resolve_key_prefers_explicit_then_file_then_env(judge, tmp_path, monkeypatch):
    for name in judge.KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    assert judge.resolve_key("  abc ", None) == "abc"
    kf = tmp_path / "key"
    kf.write_text("from-file\n")
    assert judge.resolve_key(None, kf) == "from-file"
    monkeypatch.setenv("LITELLM_API_KEY", "from-env")
    assert judge.resolve_key(None, None) == "from-env"


def test_resolve_key_exits_with_a_usable_message(judge, monkeypatch):
    for name in judge.KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exc:
        judge.resolve_key(None, None)
    assert "API_KEY" in str(exc.value)

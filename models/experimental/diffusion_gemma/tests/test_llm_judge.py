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
import threading
import time
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
    cfg = {
        "model": "azure/gpt-4o",
        "base": "https://x",
        "key": "k",
        "votes": 1,
        "max_chars": 24000,
        "timeout": 5.0,
        "concurrency": 4,
    }
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

    def fake(item, cfg, vote, **_kw):
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


# ------------------------------------------------------------------ concurrency


def test_judge_all_fans_out_across_votes_not_just_items(judge, monkeypatch):
    """The point of scheduling per CALL: 3 items x 3 votes must reach 9 in flight, which a per-item
    pool structurally cannot do -- it would cap at 3 and the barrier would break."""
    barrier = threading.Barrier(9, timeout=10)

    def fake(item, cfg, vote, **_kw):
        barrier.wait()  # BrokenBarrierError if fewer than 9 calls are ever concurrent
        return {"meaningful": True, "failure_mode": "none", "answered": True, "_usage": {}}

    monkeypatch.setattr(judge, "judge_call", fake)
    items = [_item(text=f"answer {i}") for i in range(3)]
    got = judge.judge_all(items, _cfg(judge, votes=3, concurrency=9), judge.Cache(None), None)
    assert len(got) == 3 and not any(v.get("error") for v in got)


def test_judge_all_keeps_verdicts_aligned_with_items_despite_completion_order(judge, monkeypatch):
    """Results arrive out of order under as_completed; index i must still be item i's verdict."""

    def fake(item, cfg, vote, **_kw):
        time.sleep(0.02 if "slow" in item["text"] else 0.0)  # invert completion order
        return {
            "meaningful": True,
            "failure_mode": "none",
            "answered": True,
            "selected_answer_text": item["text"],
            "_usage": {},
        }

    monkeypatch.setattr(judge, "judge_call", fake)
    items = [_item(text="slow one"), _item(text="fast one"), _item(text="  ")]
    got = judge.judge_all(items, _cfg(judge, concurrency=4), judge.Cache(None), None)
    assert got[0]["selected_answer_text"] == "slow one"
    assert got[1]["selected_answer_text"] == "fast one"
    assert got[2]["failure_mode"] == "empty"  # blank text never reached the pool


def test_judge_all_flushes_the_cache_before_it_finishes(judge, tmp_path, monkeypatch):
    """A killed high-concurrency run must keep the calls it already paid for."""
    monkeypatch.setattr(judge, "judge_call", lambda *a, **k: {"meaningful": True, "failure_mode": "none", "_usage": {}})
    path = tmp_path / "c.json"
    cache = judge.Cache(path)
    items = [_item(text=f"response {i}") for i in range(60)]
    judge.judge_all(items, _cfg(judge, concurrency=8), cache, None)
    # 60 calls with flush_every = max(25, 3) = 25, so at least two flushes happened mid-run.
    assert path.exists() and len(json.loads(path.read_text())) >= 50


def test_an_errored_vote_poisons_its_item_rather_than_shrinking_the_denominator(judge):
    votes = [
        {"meaningful": True, "failure_mode": "none", "_usage": {}},
        {"error": "HTTP 500: boom"},
        {"meaningful": True, "failure_mode": "none", "_usage": {}},
    ]
    assert judge.assemble(votes) == {"error": "HTTP 500: boom"}


def test_limiter_caps_calls_in_flight(judge):
    limiter = judge.Limiter(3)
    peak, live, lock = [0], [0], threading.Lock()

    def worker():
        with limiter:
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.01)
            with lock:
                live[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert peak[0] <= 3 and peak[0] >= 2  # gated, but genuinely concurrent


def test_limiter_halves_on_throttle_and_creeps_back(judge):
    limiter = judge.Limiter(16)
    limiter.throttled(None)
    assert limiter.limit == 8
    limiter.epoch_until = 0.0  # a genuinely new congestion epoch
    limiter.throttled(None)
    assert limiter.limit == 4 and limiter.low_water == 4 and limiter.throttles == 2
    for _ in range(judge.Limiter.PROBE_OK):
        limiter.ok()
    assert limiter.limit == 5  # additive increase, one slot per clean window
    assert "low water 4" in limiter.summary() and "2 throttle(s)" in limiter.summary()


def test_limiter_halves_once_per_burst_not_once_per_rejection(judge):
    """The overshoot that a stub proxy exposed: 26 rejections from one burst drove 32 down to 2,
    below the 6 the proxy would have granted."""
    limiter = judge.Limiter(32)
    for _ in range(26):  # one burst: all inside the same epoch
        limiter.throttled(1.0)
    assert limiter.limit == 16 and limiter.throttles == 26
    assert limiter.pause_until > 0  # the burst still extends the pause


def test_limiter_never_throttles_below_the_floor(judge):
    limiter = judge.Limiter(4)
    for _ in range(10):
        limiter.throttled(None)
        limiter.epoch_until = 0.0  # force each one to count as its own epoch
    assert limiter.limit == judge.Limiter.FLOOR


def test_limiter_never_probes_above_its_ceiling(judge):
    limiter = judge.Limiter(2)
    for _ in range(judge.Limiter.PROBE_OK * 5):
        limiter.ok()
    assert limiter.limit == 2


def test_limiter_pause_blocks_then_expires(judge):
    limiter = judge.Limiter(4)
    limiter.throttled(0.2)
    started = time.monotonic()
    with limiter:
        waited = time.monotonic() - started
    assert 0.15 <= waited < 3.0  # honoured the pause, then let the call through


@pytest.mark.parametrize("code,expect_throttle", [(429, True), (503, True), (500, True), (400, False)])
def test_http_json_only_throttles_on_pushback(judge, monkeypatch, code, expect_throttle):
    import io
    import urllib.error

    class Limited:
        def __init__(self):
            self.throttled_with = []
            self.oks = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def throttled(self, after):
            self.throttled_with.append(after)

        def ok(self):
            self.oks += 1

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "no", {"Retry-After": "3"}, io.BytesIO(b"{}"))

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    limiter = Limited()
    with pytest.raises(RuntimeError):
        judge.http_json("https://x/chat/completions", "k", {"a": 1}, retries=1, limiter=limiter)
    assert bool(limiter.throttled_with) is expect_throttle
    if expect_throttle:
        assert limiter.throttled_with[0] == 3.0  # Retry-After obeyed, not guessed over
    assert limiter.oks == 0


def test_http_json_releases_the_gate_while_backing_off(judge, monkeypatch):
    """Sleeping inside the gate would make a throttled run look busy and starve live threads."""
    import io
    import urllib.error

    inflight, peak = [0], [0]

    class Gate:
        def __enter__(self):
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
            return self

        def __exit__(self, *a):
            inflight[0] -= 1
            return False

        def throttled(self, _after):
            pass

        def ok(self):
            pass

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, io.BytesIO(b"{}"))

    slept = []
    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge.time, "sleep", lambda s: slept.append(inflight[0]))
    with pytest.raises(RuntimeError):
        judge.http_json("https://x/chat/completions", "k", {"a": 1}, retries=2, limiter=Gate())
    assert slept and all(held == 0 for held in slept), "backoff held an in-flight slot"


def test_retry_after_parsing(judge):
    import urllib.error

    def exc(value):
        return urllib.error.HTTPError("u", 429, "m", {"Retry-After": value} if value is not None else {}, None)

    assert judge.retry_after_seconds(exc("5")) == 5.0
    assert judge.retry_after_seconds(exc("9999")) == 120.0  # clamped; never idle for hours
    assert judge.retry_after_seconds(exc("Wed, 21 Oct 2026 07:28:00 GMT")) is None
    assert judge.retry_after_seconds(exc(None)) is None


def test_progress_goes_to_stderr_not_stdout(judge, capsys):
    progress = judge.make_progress(total_items=10, quiet=False)
    progress(5, 10)
    progress(10, 10)
    out, err = capsys.readouterr()
    assert out == "" and "10/10" in err


def test_progress_is_silent_when_quiet_or_single_item(judge):
    assert judge.make_progress(10, quiet=True) is None
    assert judge.make_progress(1, quiet=False) is None


def test_flush_is_atomic_and_leaves_no_tmp_file(judge, tmp_path):
    path = tmp_path / "c.json"
    cache = judge.Cache(path)
    cache.put("k", {"meaningful": True})
    cache.flush()
    assert json.loads(path.read_text()) == {"k": {"meaningful": True}}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_corrupt_cache_is_rebuilt_not_fatal(judge, tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"k": {"meaning')  # truncated by a kill mid-write
    assert judge.Cache(path).data == {}


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

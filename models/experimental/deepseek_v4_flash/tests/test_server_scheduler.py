# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for ``demo/server.py``'s multi-user decode scheduler.

No device and no weights: the model is stubbed by a fake that hands back deterministic
logits in dispatch order, which is enough to exercise the parts of the scheduler that
are easy to get wrong and impossible to see by reading the code -- that a step's output
is collected by the turn that dispatched it, that turns for different users interleave
instead of running one at a time, and that a client hanging up strands no step.

Run (ttnn venv, from the repo root)::

    pytest -q models/experimental/deepseek_v4_flash/tests/test_server_scheduler.py
"""

from __future__ import annotations

import threading
from collections import deque

import pytest
import torch

from models.experimental.deepseek_v4_flash.demo import server as S

VOCAB = 64
EOS = 2


class FakeModel:
    """Records dispatch order and hands back logits in that same FIFO order.

    Each step's logits argmax to ``(token_id + 1) % VOCAB``, so a session's output
    depends only on its own input: any crossed-up readback shows as a wrong token.
    """

    # Mirrors ``DeepSeekV4Model``: paged, with per-group (blocks used, pool size).
    paged = True
    BLOCK = 32
    POOL = 512

    def __init__(self):
        self.queue: deque = deque()
        self.active_sid = None
        self.dispatch_log: list[tuple] = []
        self.max_inflight = 0
        self.blocks: dict[int, int] = {}

    def session_usage(self):
        used = sum(self.blocks.values())
        return {"sliding": (used, self.POOL), "compress": (used // 4, self.POOL)}

    def activate_session(self, sid):
        self.active_sid = sid

    def decode_traced_async(self, token_id, pos):
        assert self.active_sid is not None, "dispatched with no active session"
        self.blocks[self.active_sid] = pos // self.BLOCK + 1  # grows as the session does
        self.queue.append((self.active_sid, token_id, pos))
        self.dispatch_log.append((self.active_sid, token_id, pos))
        self.max_inflight = max(self.max_inflight, len(self.queue))

    def read_decoded_output(self):
        sid, token_id, pos = self.queue.popleft()
        logits = torch.zeros(1, 1, VOCAB)
        logits[0, 0, (token_id + 1) % VOCAB] = 10.0
        return logits


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        # One id per character, offset into the middle of the vocab so prompts never
        # collide with EOS.
        return {"input_ids": [(ord(c) % 20) + 8 for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + (i % 26)) for i in ids)


class FakeUser:
    def __init__(self, index, engine):
        self.index = index
        self.engine = engine
        self.sid = index
        self.pos = 0
        self.messages: list[dict] = []
        self.pending_id = None
        self._next_render = 0
        self.thinking_mode = "chat"
        self.reasoning_effort = None

    def activate(self):
        self.engine.model.activate_session(self.sid)

    def reset(self):
        self.pos = 0
        self.pending_id = None
        self.messages = []
        self._next_render = 0


class FakeEngine:
    def __init__(self, num_users):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.eos_id = EOS
        self.max_seq = 4096
        self.max_new_tokens = 32
        self.traced = True
        self.rope = None
        self.lm_head = None
        self.paged = True
        self.users = [FakeUser(i, self) for i in range(num_users)]

    def tokens_left(self):
        used, total = self.model.session_usage()["sliding"]
        return (total - used) * FakeModel.BLOCK


def _run_turns(api, keys, max_tokens=12, content="hello there friend "):
    """Fire one concurrent request per entry in ``keys``; return their stats by thread."""
    results: dict[int, dict] = {}
    errors: list = []

    def client(i, key):
        body = {"messages": [{"role": "user", "content": content * 4}], "max_tokens": max_tokens, "user": key}
        try:
            results[i] = api.generate(key, body, lambda r, c: None)
        except Exception as e:  # noqa: BLE001
            errors.append((key, e))

    threads = [threading.Thread(target=client, args=(i, k)) for i, k in enumerate(keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"turns failed: {errors}"
    return results


@pytest.mark.parametrize("key", ["shared-user", S._DEFAULT_USER])
def test_same_user_key_runs_in_parallel(key: str) -> None:
    """Concurrent requests carrying one ``user`` value must not serialize.

    A regression guard: slots used to be keyed by the OpenAI ``user`` field, so a client
    that sends one identifier for all its traffic (or none, landing on the default) had
    every request queue behind the previous one on a single KV session.
    """
    engine = FakeEngine(4)
    api = S.GenerationServer(engine, "fake", prefill_chunk=4)
    api.start()
    results = _run_turns(api, [key] * 4)
    api.stop()

    assert len(results) == 4, "not every request produced a reply"
    stepped = sorted({sid for sid, _tok, _pos in engine.model.dispatch_log})
    assert stepped == [0, 1, 2, 3], f"4 requests for one user key used only sessions {stepped}"


def test_requests_queue_when_every_slot_is_busy() -> None:
    """More requests than slots is not an error: the extra ones wait for a slot."""
    engine = FakeEngine(2)
    api = S.GenerationServer(engine, "fake", prefill_chunk=4)
    api.start()
    results = _run_turns(api, ["a", "b", "c", "d"], max_tokens=6)
    api.stop()

    assert len(results) == 4, "a queued request was dropped"
    stepped = sorted({sid for sid, _tok, _pos in engine.model.dispatch_log})
    assert stepped == [0, 1], f"only 2 slots exist but sessions {stepped} were stepped"
    assert all(row["busy"] is False for row in api.pool.rows()), "a slot was left claimed"


def test_slot_choice_prefers_the_warm_cache() -> None:
    """A follow-up lands on the slot holding its history, not on an empty one.

    With free slots available, the prefix match has to win: picking an empty slot would
    work but silently re-prefill the whole conversation every turn.
    """
    engine = FakeEngine(3)
    api = S.GenerationServer(engine, "fake", prefill_chunk=8)
    api.start()
    first = api.generate("u", {"messages": [{"role": "user", "content": "first"}], "max_tokens": 5}, lambda r, c: None)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": "second"},
    ]
    used_before = [u.pos for u in engine.users]
    api.generate("u", {"messages": history, "max_tokens": 5}, lambda r, c: None)
    api.stop()

    grew = [i for i, before in enumerate(used_before) if engine.users[i].pos > before]
    assert grew == [0], f"the follow-up should have continued slot 0, but slots {grew} grew"
    assert all(u.pos == 0 for u in engine.users[1:]), "an empty slot was used instead of the warm one"


def test_unrelated_request_takes_a_free_slot() -> None:
    """A request that continues nothing goes to an empty slot rather than evicting one."""
    engine = FakeEngine(2)
    api = S.GenerationServer(engine, "fake", prefill_chunk=8)
    api.start()
    api.generate("a", {"messages": [{"role": "user", "content": "alpha"}], "max_tokens": 4}, lambda r, c: None)
    api.generate("b", {"messages": [{"role": "user", "content": "beta"}], "max_tokens": 4}, lambda r, c: None)
    api.stop()

    assert engine.users[0].pos > 0 and engine.users[1].pos > 0, "the second request reused slot 0"
    rows = {row["index"]: row["id"] for row in api.pool.rows()}
    assert rows == {0: "a", 1: "b"}, f"slots ended up owned by {rows}"


@pytest.mark.parametrize(
    "num_users, max_tokens, prefill_chunk",
    [(4, 12, 4), (8, 6, 1), (1, 8, 16)],
    ids=["4users_chunk4", "8users_chunk1", "single_user"],
)
def test_concurrent_turns_interleave(num_users: int, max_tokens: int, prefill_chunk: int) -> None:
    """Concurrent turns share the rounds, and no session ever sees another's step."""
    engine = FakeEngine(num_users)
    api = S.GenerationServer(engine, "fake", prefill_chunk=prefill_chunk)
    api.start()
    results: dict[str, dict] = {}
    errors: list = []

    def client(u):
        key = f"user{u}"
        body = {
            "messages": [{"role": "user", "content": f"hello number {u}"}],
            "max_tokens": max_tokens,
            "user": key,
        }
        try:
            results[key] = api.generate(key, body, lambda r, c: None)
        except Exception as e:  # noqa: BLE001
            errors.append((key, e))

    threads = [threading.Thread(target=client, args=(u,)) for u in range(num_users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    api.stop()

    assert not errors, f"turns failed: {errors}"
    assert len(results) == num_users, f"expected {num_users} replies, got {sorted(results)}"
    for key, stats in sorted(results.items()):
        assert stats["completion_tokens"] > 0, f"{key} generated nothing"
        assert stats["completion_tokens"] <= max_tokens, f"{key} ran past max_tokens"

    log = engine.model.dispatch_log
    # Pipelining: a step is dispatched before the previous one's output is read.
    assert engine.model.max_inflight > 1, "no pipelining: never more than one step in flight"

    # Every session's positions must be a gapless 0..n-1 run, so no step of one turn was
    # ever charged to another session's cache.
    per_sid: dict[int, list[int]] = {}
    for sid, _tok, pos in log:
        per_sid.setdefault(sid, []).append(pos)
    assert len(per_sid) == num_users, f"only {len(per_sid)} of {num_users} sessions were stepped"
    for sid, positions in per_sid.items():
        assert positions == list(range(len(positions))), f"session {sid} positions out of order: {positions[:20]}"

    if num_users > 1:
        # The turns must share the rounds rather than run one to completion at a time.
        switches = sum(1 for a, b in zip(log, log[1:]) if a[0] != b[0])
        assert switches > num_users, f"turns did not interleave (only {switches} session switches)"


def test_stats_snapshot_feeds_the_console() -> None:
    """``stats()`` carries every field ``demo/tui.py`` paints, mid-generation."""
    from models.experimental.deepseek_v4_flash.demo import tui

    engine = FakeEngine(2)
    api = S.GenerationServer(engine, "fake", prefill_chunk=2)
    api.start()
    seen: list[dict] = []

    def on_chunk(reasoning, content):
        seen.append(api.stats())  # sampled while a turn is actually running

    api.generate("alice", {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 16}, on_chunk)
    api.stop()

    assert seen, "no stats were sampled"
    mid = seen[-1]
    for key in (
        "model_id",
        "uptime",
        "slots",
        "max_seq",
        "users",
        "active",
        "rounds",
        "steps",
        "step_rate",
        "per_user_rate",
        "inflight",
        "pool",
        "tokens_left",
        "broken",
    ):
        assert key in mid, f"stats() is missing {key!r}, which the console reads"
    assert mid["pool"], "paged pool usage not reported"
    assert mid["active"], "a turn was running but none was reported active"
    turn = mid["active"][0]
    for key in ("user", "slot", "phase", "prompt_tokens", "prefilled", "generated", "max_tokens", "decode_rate"):
        assert key in turn, f"active turn row is missing {key!r}"
    assert mid["users"][0]["id"] == "alice" and mid["users"][0]["index"] == 0

    # The console must render that snapshot without a terminal attached, so a bad frame
    # shows up here rather than as a blank pane on the server.
    view = tui.ServerConsole(lambda: mid, debug=True)
    view.write(_FakeRecord("hello from the log", "DEBUG"))
    view._drain()
    frame = view.console.render_str("")  # forces console init
    del frame
    rendered = "\n".join(seg.text for seg in view.console.render(view._frame(), view.console.options))
    assert "fake" in rendered and "alice" in rendered, f"frame did not render the status: {rendered[:200]}"
    assert "hello from the log" in rendered, "the log pane dropped its line"


class _FakeRecord:
    """The bit of a loguru message the console's sink reads."""

    def __init__(self, message: str, level: str):
        import datetime

        self.record = {
            "time": datetime.datetime.now(),
            "level": type("L", (), {"name": level})(),
            "message": message,
        }


def test_client_hangup_strands_no_step() -> None:
    """A client that hangs up mid-reply still leaves a consistent session."""
    engine = FakeEngine(2)
    api = S.GenerationServer(engine, "fake", prefill_chunk=4)
    api.start()
    seen = {"n": 0}

    def on_chunk(reasoning, content):
        seen["n"] += 1
        if seen["n"] == 3:
            raise S._ClientGone()

    stats = api.generate("gone", {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 50}, on_chunk)
    api.stop()
    assert not engine.model.queue, f"{len(engine.model.queue)} steps left in flight after cancel"
    assert stats["completion_tokens"] > 0, "the partial reply was dropped"
    user = engine.users[0]
    assert user.messages[-1]["role"] == "assistant", "assistant turn not recorded"
    assert user.pos == len(engine.model.dispatch_log), "session position out of step with the cache"


def test_follow_up_turn_continues_the_session() -> None:
    """A follow-up request continues the session instead of re-prefilling it."""
    engine = FakeEngine(1)
    api = S.GenerationServer(engine, "fake", prefill_chunk=8)
    api.start()
    body1 = {"messages": [{"role": "user", "content": "first"}], "max_tokens": 5}
    first = api.generate("u", body1, lambda r, c: None)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": "second"},
    ]
    pos_before = engine.users[0].pos
    second = api.generate("u", {"messages": history, "max_tokens": 5}, lambda r, c: None)
    api.stop()
    # The follow-up extends the existing cache: it starts at the position the first turn
    # ended on, and feeds only what the new turn adds rather than the whole history.
    assert engine.users[0].pos > pos_before, "second turn did not extend the cache"
    assert second["prompt_tokens"] < pos_before, "second turn re-prefilled the whole history"


def test_request_logging_shows_the_sender_options_and_messages() -> None:
    body = {
        "model": "deepseek",
        "stream": True,
        "temperature": 0.7,
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
    }
    text = S._describe_request("chatcmpl-abc", "10.0.0.4:5122", "alice", body)

    assert "chatcmpl-abc" in text and "10.0.0.4:5122" in text and "user='alice'" in text
    assert "stream=True" in text and "temperature=0.7" in text
    assert "2 messages, 10 chars" in text
    assert "[system] be brief" in text and "[user] hi" in text
    # Absent options stay out of the line rather than showing as None.
    assert "top_p" not in text and "thinking" not in text


def test_request_logging_shows_the_message_whole_by_default() -> None:
    """The log is what you debug a prompt with, so it is not previewed by default."""
    body = {"messages": [{"role": "user", "content": "x" * 5000}]}
    assert "x" * 5000 in S._describe_request("chatcmpl-abc", "10.0.0.4:5122", "alice", body)


def test_a_logged_message_stays_one_record() -> None:
    """Newlines are collapsed so a message cannot masquerade as several log lines."""
    body = {"messages": [{"role": "user", "content": "first\nsecond\n\nthird"}]}
    text = S._describe_request("chatcmpl-abc", "10.0.0.4:5122", "alice", body)

    assert "[user] first second third" in text
    assert len(text.splitlines()) == 3, "header, count, and one line for the message"


def test_content_parts_are_accepted_like_a_plain_string() -> None:
    """OpenAI's array-of-parts ``content`` carries the same prompt as the string form."""
    parts = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert S._normalized_messages([{"role": "user", "content": parts}]) == [{"role": "user", "content": "hello world"}]


def test_normalising_strips_the_extras_a_client_echoes_back() -> None:
    """An echoed assistant reply still matches the stored one, so the turn continues.

    Clients replay the whole history including fields the server never stored; comparing
    the raw dicts would miss the prefix and re-prefill the conversation from scratch.
    """
    stored = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    echoed = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "hey", "reasoning_content": None, "tool_calls": None},
        {"role": "user", "content": "again"},
    ]
    assert S._normalized_messages(echoed)[: len(stored)] == stored


def test_non_text_content_is_refused_rather_than_dropped(expect_error) -> None:
    """Silently discarding an image part would answer a prompt the user did not send."""
    image = [{"type": "image_url", "image_url": {"url": "http://example/x.png"}}]
    with expect_error(S.RequestError, "text only") as excinfo:
        S._normalized_messages([{"role": "user", "content": image}])
    assert excinfo.value.status == 400


@pytest.mark.parametrize(
    "messages, reason",
    [
        ([], "empty"),
        ("hello", "not a list"),
        ([{"content": "hi"}], "no role"),
        (["hi"], "not an object"),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}], "assistant last"),
        ([{"role": "user", "content": 7}], "content is a number"),
    ],
)
def test_malformed_messages_are_rejected(messages, reason: str, expect_error) -> None:
    with expect_error(S.RequestError, "") as excinfo:
        S._normalized_messages(messages)
    assert excinfo.value.status == 400, reason


@pytest.mark.parametrize(
    "body, greedy",
    [
        ({}, True),
        ({"temperature": 0}, True),
        ({"temperature": None, "top_p": None}, True),
        ({"temperature": 0.7}, False),
        ({"top_p": 0.9}, False),
        ({"temperature": 0, "top_p": 0.9}, False),
    ],
)
def test_sampler_is_greedy_unless_the_request_asks_otherwise(body: dict, greedy: bool) -> None:
    """A request that names neither temperature nor top_p decodes greedily.

    Greedy is both the documented default and the cheap one: the sampling path costs
    the scheduler thread real milliseconds per token, so defaulting into it silently
    taxes every request that never asked to sample.
    """
    sampler = S._make_sampler(body.get("temperature"), body.get("top_p"))
    assert (sampler is None) == greedy


def test_sampler_matches_the_exact_softmax() -> None:
    """Truncating to the top-k logits must not skew the distribution it samples from."""
    torch.manual_seed(0)
    logits = torch.full((1, 129280), -30.0)
    logits[0, 5], logits[0, 7], logits[0, 9] = 10.0, 9.0, 8.0
    sample = S._make_sampler(1.0, None)

    draws = [sample(logits) for _ in range(3000)]
    assert set(draws) <= {5, 7, 9}, "sampled a token the tail should never reach"
    expected = torch.softmax(torch.tensor([10.0, 9.0, 8.0]), dim=0)
    for token, want in zip((5, 7, 9), expected.tolist()):
        assert abs(draws.count(token) / len(draws) - want) < 0.03, f"token {token} is skewed"


def test_top_p_keeps_the_nucleus_and_drops_the_tail() -> None:
    """``top_p`` truncates to the smallest set of tokens covering the requested mass."""
    torch.manual_seed(0)
    logits = torch.full((1, 129280), -30.0)
    logits[0, 5], logits[0, 7], logits[0, 9] = 10.0, 9.0, 2.0
    sample = S._make_sampler(1.0, 0.9)

    draws = {sample(logits) for _ in range(2000)}
    assert draws == {5, 7}, f"nucleus should hold exactly the top two tokens, got {draws}"

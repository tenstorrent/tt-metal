# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``--decode-batch`` scheduling (:class:`~..demo.server._BatchScheduler`) without a device.

A fake model whose next token is a hash of its session's whole history stands in for ``DeepSeekV41Model``: its
one-row steps go through the real :class:`~..demo.server.V41ChatEngine` packet path and the scheduler's output reader,
its batch steps through ``decode_batch_block``. Turns join (prompt replayed through one-row steps, then decode) and
leave (EOS or ``max_tokens``) at different times, so the run mixes one-row decode of a lone reply, batch steps of
2-4 replies, and prompts replayed between batch steps. Every reply must be what that session decodes alone, and every
step must be fed at its session's next position (the fake asserts it)."""

from __future__ import annotations

import queue
import threading
from collections import deque
from types import SimpleNamespace

import torch

from models.experimental.deepseek_v41_flash.demo import server as srv
from models.experimental.deepseek_v4_flash.demo.chat_cli import UserSession
from models.experimental.deepseek_v4_flash.demo.server import _OutputReader

_VOCAB = 1000
_EOS = 1


def _next(history: list[int]) -> int:
    """The fake model's pick after ``history``: depends on every token and its position."""
    h = 0
    for token in history:
        h = (h * 1000003 + token + 7) % (2**31 - 1)
    return _EOS if h % 53 == 0 else 2 + h % (_VOCAB - 2)


def _alone(prompt: list[int], max_tokens: int) -> list[int]:
    history, generated = list(prompt), []
    token = _next(history)
    while len(generated) < max_tokens and token != _EOS:
        generated.append(token)
        history.append(token)
        token = _next(history)
    return generated


class _FakeModel:
    config = SimpleNamespace(vocab_size=_VOCAB)
    verify_rows = 4
    host_runtime_running = False

    def __init__(self):
        self.history: dict[int, list[int]] = {}
        self.resident = None
        self.posted: deque = deque()  # (sid, pos) of each posted one-row step, in posting order
        self.outputs: queue.Queue = queue.Queue()
        self.batch_rows: list[int] = []
        self.one_row = 0
        self.lock = threading.Lock()

    def open_session(self) -> int:
        sid = len(self.history)
        self.history[sid] = []
        return sid

    def activate_session(self, sid: int) -> None:
        self.resident = sid

    def reset_session(self, sid: int) -> None:
        self.history[sid] = []

    def ensure_session_capacity(self, pos: int, sid=None) -> None:
        pass

    def session_tokens_left(self) -> int:
        return 1 << 30

    def replay_traced_ahead(self, positions, token_ids=None) -> None:
        with self.lock:
            self.posted.extend((self.resident, int(p)) for p in positions)

    def write_step_packet(self, token: int, pos: int) -> None:
        with self.lock:
            sid, posted_at = self.posted.popleft()
            # A rerun (repeat_traced) feeds the session's newest position into a step posted later.
            assert posted_at >= pos, (sid, posted_at, pos)
            history = self.history[sid]
            assert len(history) >= pos, (sid, len(history), pos)
            del history[pos:]
            history.append(int(token))
            self.one_row += 1
            self.outputs.put(torch.tensor([[_next(history)]], dtype=torch.int32))

    def read_decoded_output(self) -> torch.Tensor:
        return self.outputs.get(timeout=10)

    def end_session_at(self, sid: int, pos: int) -> None:
        del self.history[sid][pos:]

    def decode_batch_block(self, token_ids, sids, positions) -> list[int]:
        assert 1 <= len(sids) <= self.verify_rows and len(set(sids)) == len(sids)
        with self.lock:
            assert not self.posted, f"a batch step with one-row steps still posted: {list(self.posted)}"
            picks = []
            for token, sid, pos in zip(token_ids, sids, positions):
                history = self.history[sid]
                assert len(history) == pos, (sid, len(history), pos)
                history.append(int(token))
                picks.append(_next(history))
            self.batch_rows.append(len(sids))
        return picks


class _Stream:
    content = reasoning = ""
    tool_calls: list = []

    def close(self, token_ids=None) -> None:
        pass

    def reasoning_tokens(self, token_ids) -> int:
        return 0


def _engine(model: _FakeModel, num_users: int):
    engine = srv.V41ChatEngine.__new__(srv.V41ChatEngine)
    engine.model = model
    engine.eos_id = _EOS
    engine.max_seq = 1 << 20
    engine.decode_batch = 4
    engine.dspark = False
    engine._replay_q = deque()
    engine._packets = queue.Queue()
    engine.packet_lock = threading.Lock()
    engine._packet_error = None
    engine._packet_writer = threading.Thread(target=engine._write_packets, daemon=True)
    engine._packet_writer.start()
    args = SimpleNamespace(think=False, reasoning_effort=None, system_prompt=None)
    engine.users = [UserSession(engine, i, args) for i in range(num_users)]
    return engine


def _scheduler(engine):
    pool = SimpleNamespace(max_running=4, reply_at={}, near_misses=set(), _cv=threading.Condition())
    server = SimpleNamespace(engine=engine, pool=pool, prefiller=None, prefix_cache=None)
    sched = srv._BatchScheduler(server, prefill_chunk=16, decode_ahead=8)
    sched._reader = _OutputReader(engine.model, engine.packet_lock, engine.write_prefed)
    return sched


def _turn(slot: int, prompt: list[int], max_tokens: int):
    turn = srv._Turn(f"u{slot}", slot, {"messages": [{"role": "user", "content": "x"}]}, None, max_tokens)
    turn.ids = list(prompt)
    turn.stream = _Stream()
    turn.t_admit = turn.t_submit
    return turn


def test_batch_scheduler_matches_alone() -> None:
    model = _FakeModel()
    engine = _engine(model, num_users=6)
    sched = _scheduler(engine)
    gen = torch.Generator().manual_seed(0)
    # (round it joins at, prompt, max_tokens): a lone reply first, then others join and leave while it runs.
    plan = [(0, 40, 60), (12, 25, 30), (14, 70, 80), (30, 5, 20), (31, 33, 45), (60, 50, 25), (61, 12, 40)]
    turns = []
    for _, n, max_tokens in plan:
        turns.append((list(torch.randint(2, _VOCAB, (n,), generator=gen).tolist()), max_tokens))
    joined: list = []
    slots = deque(range(len(engine.users)))
    held: dict = {}
    ended: set = set()
    rounds = 0
    try:
        while rounds < 5000:
            for i, (at, _, _) in enumerate(plan):
                if at == rounds and i not in held:
                    held[i] = None  # waits for a free slot
            for i in [i for i, t in held.items() if t is None]:
                if not slots:
                    break
                slot = slots.popleft()
                engine.users[slot].reset()  # a new conversation in a freed slot
                prompt, max_tokens = turns[i]
                turn = _turn(slot, prompt, max_tokens)
                held[i] = turn
                joined.append(i)
                sched._active.append(turn)
            if not sched._active and len(joined) == len(plan):
                break
            if sched._active:
                sched._round()
            for i, turn in held.items():
                if turn is not None and turn not in sched._active and i not in ended:
                    ended.add(i)
                    slots.append(turn.slot)
            rounds += 1
        assert sched._broken is None, sched._broken
        assert len(joined) == len(plan) and not sched._active, (joined, sched._active)
        for i, turn in held.items():
            prompt, max_tokens = turns[i]
            want = _alone(prompt, max_tokens)
            assert turn.generated == want, (i, len(turn.generated), len(want))
            assert not turn.events.empty()
        # Both kinds of step ran: one-row steps (prompts, the lone reply) and batch steps of several replies.
        assert model.one_row > 0
        assert model.batch_rows and max(model.batch_rows) >= 3, model.batch_rows
        assert sched.batch_steps == len(model.batch_rows)
        assert sched.batch_rows == sum(model.batch_rows)
    finally:
        sched._reader.close()
        engine._stop_packet_writer()


def test_batch_scheduler_cancel_mid_batch() -> None:
    """A reply stopped while it sits in the batch leaves between steps with what it had; the others go on."""
    model = _FakeModel()
    engine = _engine(model, num_users=4)
    sched = _scheduler(engine)
    gen = torch.Generator().manual_seed(2)  # every reply at least 5 tokens
    prompts = [list(torch.randint(2, _VOCAB, (n,), generator=gen).tolist()) for n in (20, 30, 9, 17)]
    turns = [_turn(slot, prompt, 200) for slot, prompt in enumerate(prompts)]
    wants = [_alone(prompt, 200) for prompt in prompts]
    stop = max(range(len(turns)), key=lambda i: len(wants[i]))  # stopped after 5 tokens, with more to come
    assert len(wants[stop]) > 10
    sched._active.extend(turns)
    try:
        rounds = 0
        while sched._active and rounds < 5000:
            sched._round()
            rounds += 1
            if len(turns[stop].generated) >= 5:
                turns[stop].cancelled.set()
        assert sched._broken is None, sched._broken
        assert not sched._active
        for i, (turn, want) in enumerate(zip(turns, wants)):
            if i == stop:
                assert 0 < len(turn.generated) < len(want) and turn.generated == want[: len(turn.generated)]
            else:
                assert turn.generated == want, (i, len(turn.generated), len(want))
        assert 4 in model.batch_rows, model.batch_rows
    finally:
        sched._reader.close()
        engine._stop_packet_writer()


def test_one_row_replies_interleave(monkeypatch) -> None:
    """Below the batch threshold two replies decode a row at a time, interleaved: the prefix snapshot each wants when
    its prompt is in (no prefiller) must not wait out the other's whole reply (its steps always posted ahead)."""
    monkeypatch.setattr(srv, "_BATCH_MIN_USERS", 3)
    model = _FakeModel()
    engine = _engine(model, num_users=4)
    sched = _scheduler(engine)
    gen = torch.Generator().manual_seed(2)
    prompts = [list(torch.randint(2, _VOCAB, (n,), generator=gen).tolist()) for n in (20, 9)]
    turns = [_turn(slot, prompt, 40) for slot, prompt in enumerate(prompts)]
    for turn in turns:
        turn.ignore_eos = True
    sched._active.extend(turns)
    try:
        rounds = 0
        while all(turn in sched._active for turn in turns) and rounds < 5000:
            sched._round()
            rounds += 1
        # When the first reply ends, the other is well into its own.
        assert min(len(turn.generated) for turn in turns) >= 20, [len(turn.generated) for turn in turns]
        while sched._active and rounds < 5000:
            sched._round()
            rounds += 1
        assert sched._broken is None and not model.batch_rows
        for turn, prompt in zip(turns, prompts):
            history, want = list(prompt), []
            for _ in range(40):
                want.append(_next(history))
                history.append(want[-1])
            assert turn.generated == want
    finally:
        sched._reader.close()
        engine._stop_packet_writer()


def test_batch_min_users_by_batch_size(monkeypatch):
    """B=4 takes batch steps from 3 seated replies, B=2 from both rows; DEEPSEEK_V41_BATCH_MIN_USERS sets either."""
    monkeypatch.setattr(srv, "_BATCH_MIN_USERS_ENV", None)
    monkeypatch.setattr(srv, "_BATCH_MIN_USERS", 3)
    assert srv._batch_min_users(4) == 3
    assert srv._batch_min_users(2) == 2
    monkeypatch.setattr(srv, "_BATCH_MIN_USERS_ENV", "1")
    assert srv._batch_min_users(4) == 1 and srv._batch_min_users(2) == 1

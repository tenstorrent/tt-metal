# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server on the full ttnn ``DeepSeekV4Model``.

Serves the same decode engine as ``demo/chat_cli.py`` (whose ``ChatEngine`` is
reused unmodified) over HTTP in the shape of the OpenAI API, so existing SDKs,
``curl`` and notebook clients can talk to the model without touching tt-metal
themselves:

* ``GET    /v1/models``           -- the served model
* ``POST   /v1/chat/completions`` -- chat completions (``stream=true`` for SSE)
* ``POST   /v1/completions``      -- legacy text completions
* ``GET    /v1/sessions``         -- active KV-cache sessions / usage (extension)
* ``DELETE /v1/sessions/<user>``  -- reset one session (extension)
* ``GET    /health``              -- liveness

**Decode over the device sockets.** The model is loaded once and kept resident;
every turn is a handful of traced decode steps. Each step pushes the token to the
model through the **host->device socket** (``DeepSeekV4Model.write_step_packet``
-- the only host->device traffic of a traced step) and receives the logits back
off the **device->host socket** (``DeepSeekV4Model.read_decoded_output``). There
is no prefill op: a prompt is replayed one decode step per token at ascending
absolute positions, so a follow-up turn only feeds the tokens it adds.

**Users & KV cache.** ``--num-users`` paged KV-cache sessions (8 by default) are claimed
at startup -- one captured trace, one block pool holding ``--total-context`` tokens
across everyone -- and then handed out a turn at a time, so ``--num-users`` is the number
of requests that can generate at once. Raising it costs little memory (a session's
sliding ring is bounded by ``sliding_window``, and the compressed blocks come from the
shared budget) but divides the device's token rate: the rounds interleave the users'
steps, they do not batch them, so more users means more total throughput and a slower
reply each. A slot holds a warm cache, so a request is given the slot whose stored
conversation its ``messages`` continue -- then only the new turn's tokens are fed --
falling back to an empty slot, or to the least recently used one, which is re-prefilled
from the request. The OpenAI ``user`` field only labels who last held a slot: clients
that send one identifier for all their traffic, or none, still run in parallel.

**Concurrency.** Up to ``--num-users`` turns generate at once. One scheduler thread
owns the device and walks the active turns in rounds, dispatching a step per turn and
only reading that step's logits back a round later, once every other turn's step has
been queued behind it -- the pipelined round-robin of
``tests/test_multi_user_paged_decode_demo.py``, with prefill and decode turns mixed
into the same rounds (a prompt is fed ``--prefill-chunk`` tokens per round). This
single thread is required, not incidental: the trace replays and the paged session
state have to be driven from one thread, and each step's output must be collected in
dispatch order. HTTP threads never touch the device -- they submit a turn and relay
its reply -- so total throughput scales with the users while each reply streams
independently. A request that arrives with every slot busy waits for one to free.

**Live console.** On a terminal the server runs a split view (``demo/tui.py``): a status
header with uptime, active turns, throughput and the KV block pool, a row per cache slot
showing who holds it and how fast it is decoding, and under it the log with the time in a
left column and the message on the right. ``d`` toggles the debug lines (admissions, slot
assignments, page allocations, prefill lengths, per-user tok/s), ``p`` pauses the scroll,
``c`` clears it and ``q`` quits. ``--debug`` starts with those lines on and ``--no-tui``
(or a redirected stdout) falls back to plain logging.

**OpenAI compatibility notes.**

* ``messages`` are OpenAI-shaped ``{role, content}`` dicts. If the incoming
  history is a strict extension of the session's stored conversation only the new
  user turn is fed; anything else (a first turn, a foreign client, an edited
  history) rewinds the session and re-prefills from the request, so a client that
  keeps its own full history always stays consistent with the server.
* ``max_tokens`` caps the reply (default: the engine-wide ``--max-new-tokens``);
  ``temperature`` / ``top_p`` sample from the per-step logits (default greedy
  argmax). ``thinking: true`` and ``reasoning_effort`` are DeepSeek-V4 extensions
  that switch that session to the thinking template. In thinking mode the reply
  carries the reasoning block in the ``reasoning_content`` field, exactly like the
  DeepSeek reasoner API, while ``content`` holds the answer. ``stop`` sequences
  are not supported.
* Errors use the OpenAI envelope ``{"error": {"message", "type", "code"}}``; for
  ``stream=true`` the error arrives as an SSE event before ``[DONE]``.

Run it (ttnn venv, from the repo root)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v4_flash/demo/server.py \\
        --num-users 8 --max-context 16384 --host 0.0.0.0 --port 8000 --debug

Then, from any other shell (see ``demo/client.py`` for the full client)::

    curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.demo import tui
from models.experimental.deepseek_v4_flash.demo.chat_cli import (
    ChatEngine,
    ContextFull,
    UserSession,
    open_mesh_device,
)
from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message
from models.experimental.deepseek_v4_flash.tt.common import _region
from models.experimental.deepseek_v4_flash.tt.paged_cache import PagedCacheFull

_VENDOR = "tenstorrent"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DEFAULT_USER = "default"


class RequestError(Exception):
    """A client-side error mapped onto an OpenAI error envelope."""

    def __init__(self, status: int, message: str, err_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type or ("invalid_request_error" if status < 500 else "server_error")


class _ClientGone(Exception):
    """The SSE client hung up mid-stream; stop generating and drop the reply."""


def _error_parts(exc: Exception) -> tuple[int, str, str]:
    """Map an engine exception onto an OpenAI ``(status, message, type)``."""
    if isinstance(exc, RequestError):
        return exc.status, exc.message, exc.err_type
    if isinstance(exc, ContextFull):
        return 413, str(exc), "context_length_exceeded"
    if isinstance(exc, PagedCacheFull):
        return 429, f"shared KV-cache pool is full: {exc}", "insufficient_quota"
    return 500, f"internal server error: {exc}", "server_error"


SAMPLER_TOP_K = 64


def _make_sampler(temperature, top_p):
    """A logits->token sampler honouring OpenAI's ``temperature``/``top_p``, or
    ``None`` for greedy argmax when the request asks for neither.

    Sampling narrows to the ``SAMPLER_TOP_K`` highest logits before normalising.
    A softmax over the full 129k vocabulary costs ~3ms per token on the host, and
    ~15ms once ``top_p`` sorts it, against a ~10ms decode step: the scheduler
    thread would spend more time sampling than the device spends decoding.
    """
    temperature = float(temperature) if temperature is not None else 0.0
    top_p = float(top_p) if top_p is not None else None
    if temperature <= 0 and top_p is None:
        return None

    def sample(logits: torch.Tensor) -> int:
        t = temperature if temperature > 0 else 1.0
        values, index = torch.topk(logits[0], min(SAMPLER_TOP_K, logits.shape[-1]))
        probs = torch.softmax(values / t, dim=-1)
        if top_p is not None and top_p < 1.0:
            cumulative = torch.cumsum(probs, dim=-1)
            probs = torch.where(cumulative - probs <= top_p, probs, torch.zeros_like(probs))
            probs = probs / probs.sum()
        return int(index[torch.multinomial(probs, 1)].item())

    return sample


class _Streamer:
    """Incrementally detokenize a reply, separating the ``<think>`` reasoning block
    from the answer and reporting only newly resolved text via
    ``on_chunk(reasoning, content)``.

    A token can carry half a UTF-8 sequence, so the ids are re-decoded cumulatively
    and a trailing U+FFFD is held back; trailing text that could still turn into a
    think tag is likewise withheld until it resolves, so deltas never carry half a
    tag. The reasoning block and the answer are emitted as tag-free strings, ready
    for the ``reasoning_content`` / ``content`` fields of the OpenAI response.
    """

    def __init__(self, tokenizer, on_chunk):
        self.tokenizer = tokenizer
        self.on_chunk = on_chunk
        self.text = ""  # raw decoded reply, tags included, no escapes
        self.reasoning = ""
        self.content = ""
        self._sent_r = 0
        self._sent_c = 0

    def push(self, token_ids) -> None:
        full = self.tokenizer.decode(token_ids, skip_special_tokens=False).rstrip("\ufffd")
        self.text = full
        held = self._held_back(full)
        reasoning, content = self._split(full[: len(full) - held])
        # A re-decode can revise the tail (split UTF-8); resync to the common prefix.
        if not reasoning.startswith(self.reasoning):
            self._sent_r = min(self._sent_r, len(os.path.commonprefix([reasoning, self.reasoning])))
        if not content.startswith(self.content):
            self._sent_c = min(self._sent_c, len(os.path.commonprefix([content, self.content])))
        if len(reasoning) > self._sent_r:
            self.on_chunk(reasoning[self._sent_r :], "")
            self._sent_r = len(reasoning)
        if len(content) > self._sent_c:
            self.on_chunk("", content[self._sent_c :])
            self._sent_c = len(content)
        self.reasoning, self.content = reasoning, content

    def close(self) -> None:
        """Flush the tail held back for a possible partial tag or character.

        Generation is over, so nothing more can complete the tail: it is emitted
        as-is and :attr:`reasoning` / :attr:`content` hold the complete reply."""
        reasoning, content = self._split(self.text)
        if len(reasoning) > self._sent_r:
            self.on_chunk(reasoning[self._sent_r :], "")
            self._sent_r = len(reasoning)
        if len(content) > self._sent_c:
            self.on_chunk("", content[self._sent_c :])
            self._sent_c = len(content)
        self.reasoning, self.content = reasoning, content

    def _split(self, text: str) -> tuple[str, str]:
        start = text.find(_THINK_OPEN)
        if start == -1:
            return "", text
        after = text[start + len(_THINK_OPEN) :]
        end = after.find(_THINK_CLOSE)
        if end == -1:
            return after, text[:start]
        return after[:end], text[:start] + after[end + len(_THINK_CLOSE) :]

    @staticmethod
    def _held_back(text: str) -> int:
        """Characters at the end that could still turn into a tag once more arrive."""
        for tag in (_THINK_CLOSE, _THINK_OPEN):
            for n in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:n]):
                    return n
        return 0


class _SlotPool:
    """The KV-cache slots, handed out per *request* rather than per ``user`` key.

    ``user`` in the OpenAI API is a stable end-user identifier: a client may send one
    value for all of its traffic, or none at all. Tying a slot to it would make every
    request from such a client queue behind the previous one, so a slot is instead
    claimed for the duration of a turn and released afterwards, and any request can use
    any slot that is not busy.

    Which free slot matters, because a slot holds a warm KV cache. :meth:`acquire`
    prefers, in order:

    1. a slot whose stored conversation is a prefix of this request's history -- the
       follow-up case, where the cache already holds everything but the new turn and
       only the new tokens are fed;
    2. a slot with no conversation at all, which costs nobody their cache;
    3. the least recently used slot, whose conversation is then dropped and the request
       re-prefilled from its own messages.

    When every slot is busy the caller waits: the turns in flight are bounded by
    ``--max-new-tokens``, so a slot always comes back.
    """

    def __init__(self, engine: ChatEngine):
        self.engine = engine
        self._cv = threading.Condition()
        self._busy = [False] * len(engine.users)
        self.owner: list[str | None] = [None] * len(engine.users)
        self._used_at = [0.0] * len(engine.users)
        self._closing = False

    def __len__(self) -> int:
        return len(self._busy)

    # -- claiming --------------------------------------------------------------- #
    def acquire(self, user_key: str, messages: list[dict]) -> int:
        """Claim a slot for one turn, waiting if they are all busy."""
        t0 = time.perf_counter()
        with self._cv:
            while True:
                if self._closing:
                    raise RequestError(503, "server is shutting down", "server_error")
                slot = self._pick(messages)
                if slot is not None:
                    break
                logger.debug(
                    f"user {user_key!r} waiting for a KV slot: all {len(self._busy)} busy "
                    f"({', '.join(str(o) for o in self.owner)})"
                )
                self._cv.wait(timeout=1.0)
            self._busy[slot] = True
            previous, self.owner[slot] = self.owner[slot], user_key
            self._used_at[slot] = time.time()
        waited = time.perf_counter() - t0
        logger.debug(
            f"user {user_key!r} -> slot {slot} (sid {self.engine.users[slot].sid}, "
            f"{self._reason(slot, previous, user_key, messages)}"
            f"{f', waited {waited:.2f}s' if waited > 0.01 else ''})"
        )
        return slot

    def reserve(self, slot: int) -> None:
        """Claim one specific slot, waiting for its turn to finish (used by a reset)."""
        with self._cv:
            while self._busy[slot] and not self._closing:
                self._cv.wait(timeout=1.0)
            if self._closing:
                raise RequestError(503, "server is shutting down", "server_error")
            self._busy[slot] = True

    def release(self, slot: int) -> None:
        with self._cv:
            self._busy[slot] = False
            self._used_at[slot] = time.time()
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()

    def _pick(self, messages: list[dict]) -> int | None:
        free = [i for i, busy in enumerate(self._busy) if not busy]
        if not free:
            return None
        # 1. The longest stored conversation this request continues.
        best, best_len = None, 0
        for i in free:
            stored = self.engine.users[i].messages
            if stored and len(messages) > len(stored) and messages[: len(stored)] == stored and len(stored) > best_len:
                best, best_len = i, len(stored)
        if best is not None:
            return best
        # 2. A slot holding nothing.
        for i in free:
            if not self.engine.users[i].messages:
                return i
        # 3. The one whose cache has been idle longest.
        return min(free, key=lambda i: self._used_at[i])

    def _reason(self, slot: int, previous: str | None, user_key: str, messages: list[dict]) -> str:
        user = self.engine.users[slot]
        stored = user.messages
        if stored and len(messages) > len(stored) and messages[: len(stored)] == stored:
            return f"continuing {len(stored)} messages at {user.pos} tokens"
        if not stored:
            return "empty slot"
        if previous not in (None, user_key):
            return f"reusing {previous!r}'s idle slot, dropping {user.pos} cached tokens"
        return f"re-prefill, dropping {user.pos} cached tokens"

    # -- reporting -------------------------------------------------------------- #
    def rows(self) -> list[dict]:
        """One row per slot, for ``/v1/sessions`` and the status pane."""
        with self._cv:
            busy, owner = list(self._busy), list(self.owner)
        rows = []
        for index, user in enumerate(self.engine.users):
            rows.append(
                {
                    "id": owner[index] or "",
                    "index": index,
                    "tokens": user.pos,
                    "max_context": self.engine.max_seq,
                    "messages": len(user.messages),
                    "thinking": user.thinking_mode == "thinking",
                    "busy": busy[index],
                }
            )
        return rows

    def slots_of(self, user_key: str) -> list[int]:
        with self._cv:
            return [i for i, owner in enumerate(self.owner) if owner == user_key]


class _Turn:
    """One in-flight chat turn: the request's decode state plus its output channel.

    Created by the HTTP thread, then owned by the :class:`_Scheduler` thread, which is
    the only one allowed to touch the model or the :class:`UserSession`. The two
    threads meet at :attr:`events` (the reply, delta by delta) and :attr:`cancelled`
    (the client hung up), both of which are thread-safe.

    A turn starts in the ``prefill`` phase, feeding its prompt tokens, and moves to
    ``decode`` once the last of them has been fed. :attr:`pending` records the steps
    this turn has in flight, in dispatch order, so the scheduler knows how many
    outputs to collect for it and whether each one is a prompt token's (discarded) or
    a generated token's.
    """

    PREFILL = "prefill"
    DECODE = "decode"

    def __init__(self, user_key: str, slot: int, body: dict, sampler, max_tokens: int):
        self.user_key = user_key
        self.slot = slot
        self.body = body
        self.sampler = sampler
        self.max_tokens = max_tokens
        self.text = str(body["messages"][-1]["content"])

        self.events: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()

        self.phase = self.PREFILL
        self.ids: list[int] = []  # the turn's prompt tokens
        self.fed = 0  # how many of them have been dispatched
        self.next_id: int | None = None  # produced but not yet fed back
        self.generated: list[int] = []
        self.pending: deque[bool] = deque()  # per in-flight step: is it the last prompt token?
        # Logits of steps run eagerly (``--no-trace``, which has no async dispatch and so
        # no pipelining); empty on the traced path, where they arrive over the D2H socket.
        self.eager: deque[torch.Tensor] = deque()
        self.stream: _Streamer | None = None
        self.hit_cap = False
        # Set when a dispatch fails. The turn stays in the round-robin until its steps
        # already in flight have been read back, because the outputs of every other turn
        # queued behind them can only be collected in order.
        self.error: Exception | None = None

        # Timings, for the status pane and the debug log. Prefill and decode are measured
        # separately because they cost very differently per token: a prompt's tokens
        # pipeline ``prefill_chunk`` at a time, a reply's are one per round.
        self.t_submit = time.perf_counter()
        self.t_admit = 0.0
        self.t_prefill_done = 0.0
        self.t_done = 0.0

    @property
    def prompt_left(self) -> int:
        return len(self.ids) - self.fed

    @property
    def prefill_seconds(self) -> float:
        end = self.t_prefill_done or time.perf_counter()
        return max(end - self.t_admit, 0.0) if self.t_admit else 0.0

    @property
    def decode_seconds(self) -> float:
        if not self.t_prefill_done:
            return 0.0
        return max((self.t_done or time.perf_counter()) - self.t_prefill_done, 0.0)

    @property
    def decode_rate(self) -> float:
        """Tokens per second this turn has generated, for this user alone."""
        seconds = self.decode_seconds
        return len(self.generated) / seconds if seconds > 0 else 0.0

    def status(self) -> dict:
        """A snapshot for the status pane (read from another thread, so keep it cheap)."""
        return {
            "user": self.user_key,
            "slot": self.slot,
            "phase": self.phase,
            "prompt_tokens": len(self.ids),
            "prefilled": self.fed,
            "generated": len(self.generated),
            "max_tokens": self.max_tokens,
            "prefill_seconds": self.prefill_seconds,
            "decode_rate": self.decode_rate,
            "cancelled": self.cancelled.is_set(),
        }


class _Scheduler:
    """The one thread that drives the device, generating for every active turn at once.

    Traced decode has two hard constraints: the trace replays and the paged session
    state must be driven from a single thread, and ``read_decoded_output`` returns the
    *oldest* in-flight step, so outputs have to be collected in dispatch order. Both
    are met by keeping all model work here and tracking dispatch order explicitly in
    :attr:`_inflight`.

    Each round walks the active turns in a stable order and, per turn, collects the
    outputs of the steps it dispatched last round and then dispatches its next ones.
    Because a turn's own output is only read at the start of its *next* round -- after
    every other turn's step has already been queued behind it -- the host never waits
    on one turn before feeding the next, and device work overlaps host work. This is
    the pipelining of ``tests/test_multi_user_paged_decode_demo.py``, with prefill and
    decode turns mixed into the same rounds.

    Prompt tokens are dispatched ``prefill_chunk`` at a time: their logits are thrown
    away (only the last prompt token's prediction is used), so they carry no
    step-to-step dependency and can all be in flight at once. A decode step, whose
    input is the previous step's sampled output, is necessarily one per round.
    """

    def __init__(self, server: "GenerationServer", prefill_chunk: int = 16):
        self.server = server
        self.engine = server.engine
        self.prefill_chunk = max(1, prefill_chunk)
        self._new: deque[_Turn] = deque()
        self._chores: deque[tuple] = deque()  # (callable, done event, result box)
        self._active: list[_Turn] = []
        self._inflight: deque[_Turn] = deque()  # dispatch order across all turns
        self._wake = threading.Condition()
        self._stop = False
        self._broken: Exception | None = None  # a failed readback; see _abort_all
        # Counters for the status pane and the periodic debug lines.
        self._pool_seen: dict = {}
        self.rounds = 0
        self.steps = 0
        self._window_t0 = time.perf_counter()
        self._window_steps = 0
        self.step_rate = 0.0  # steps/s across all users, over the last window
        self._t_started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="decode-scheduler", daemon=True)

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout=30)

    def submit(self, turn: _Turn) -> None:
        """Hand a turn to the scheduler; it is admitted at the next round boundary."""
        with self._wake:
            if self._stop:
                raise RequestError(503, "server is shutting down", "server_error")
            if self._broken is not None:
                raise RequestError(503, f"decode is wedged, restart the server: {self._broken}", "server_error")
            self._new.append(turn)
            self._wake.notify_all()

    def run_on_scheduler(self, fn):
        """Run ``fn`` on the scheduler thread, between rounds, and return its result.

        For work that touches the device or a session outside a turn -- a ``/v1/sessions``
        reset frees KV blocks and rewrites page tables -- which must not race with the
        rounds nor land between a step's dispatch and its readback."""
        done = threading.Event()
        box: list = []
        with self._wake:
            if self._stop:
                raise RequestError(503, "server is shutting down", "server_error")
            self._chores.append((fn, done, box))
            self._wake.notify_all()
        done.wait()
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]

    @property
    def active_turns(self) -> int:
        return len(self._active)

    def status(self) -> dict:
        """A snapshot of the scheduler for the status pane.

        Read from the TUI thread without a lock: every field is a plain int/float or a
        copy, and a status pane that is one round stale is harmless, whereas locking the
        scheduler to paint a screen would not be.
        """
        active = [turn.status() for turn in list(self._active)]
        return {
            "active": active,
            "rounds": self.rounds,
            "steps": self.steps,
            "step_rate": self.step_rate,
            "per_user_rate": self.step_rate / len(active) if active else 0.0,
            "inflight": len(self._inflight),
            "prefill_chunk": self.prefill_chunk,
            "pool": self.pool_usage(),
            "tokens_left": self.engine.tokens_left(),
            "broken": None if self._broken is None else str(self._broken),
        }

    def _loop(self) -> None:
        try:
            self._serve()
        except BaseException as e:  # noqa: BLE001 - nobody is left to report this to
            logger.exception(f"decode scheduler died: {e}")
            self._broken = e if isinstance(e, Exception) else RuntimeError(str(e))
            self._drain(RequestError(503, f"decode scheduler died: {e}", "server_error"))
            raise

    def _serve(self) -> None:
        while True:
            with self._wake:
                while not self._stop and not self._new and not self._chores and not self._active:
                    self._wake.wait()
                if self._stop:
                    self._drain(RequestError(503, "server is shutting down", "server_error"))
                    break
                if self._broken is not None:
                    # Nothing more can be generated (submit() rejects new turns too), but a
                    # turn that slipped in before the failure still needs an answer.
                    self._drain(self._broken)
                    continue
                admitting, self._new = list(self._new), deque()
                chores, self._chores = list(self._chores), deque()
            for fn, done, box in chores:
                try:
                    box.append(fn())
                except BaseException as e:  # noqa: BLE001 - handed back to the caller's thread
                    box.append(e)
                finally:
                    done.set()
            for turn in admitting:
                self._admit(turn)
            if self._active:
                self._round()

    # -- admission -------------------------------------------------------------- #
    def _admit(self, turn: _Turn) -> None:
        """Bring a submitted turn into the round-robin: sync its conversation against
        the request and render the tokens this turn adds.

        Runs here rather than on the HTTP thread because a rewind
        (``UserSession.reset``) frees KV blocks and rewrites page tables on the
        device."""
        user = self.engine.users[turn.slot]
        turn.t_admit = time.perf_counter()
        continuing = False
        logger.debug(
            f"admitting user {turn.user_key!r} (slot {turn.slot}, sid {user.sid}): "
            f"queued {turn.t_admit - turn.t_submit:.3f}s, at pos {user.pos}/{self.engine.max_seq}, "
            f"{len(user.messages)} stored messages, {len(self._active)} turns already active"
        )
        try:
            self.server._apply_options(user, turn.body)
            continuing = self.server._sync_messages(user, turn.body["messages"])
            if continuing:
                user.messages.append({"role": "user", "content": turn.text})
                turn.ids = self.server._render_ids(user, include_assistant=False)
            else:
                user.messages = user.messages + [{"role": "user", "content": turn.text}]
                user._next_render = 0
                turn.ids = self.server._render_ids(user, include_assistant=True)
            self._check_capacity(user, turn.ids)
        except Exception as e:  # noqa: BLE001 - reported to the client, the server stays up
            if continuing:
                user.messages.pop()  # nothing was fed: leave the conversation as it was
            logger.debug(f"user {turn.user_key!r} rejected at admission: {type(e).__name__}: {e}")
            turn.events.put(("error", e))
            return
        user._next_render = len(user.messages)
        turn.stream = _Streamer(self.engine.tokenizer, lambda r, c: turn.events.put(("delta", r, c)))
        self._active.append(turn)
        logger.debug(
            f"user {turn.user_key!r} admitted: prefill {len(turn.ids)} tokens "
            f"({'continuing' if continuing else 're-prefill from request'}), "
            f"cap {turn.max_tokens} new tokens, pool {self._pool_summary()}"
        )

    # -- KV pool reporting ------------------------------------------------------- #
    def pool_usage(self) -> dict:
        """Per-group ``(blocks used, blocks total)`` of the shared page pool, or ``{}``
        on the dense (``--no-trace``) path. Host-side bookkeeping, so it is cheap enough
        to poll for the status pane."""
        if not (self.engine.traced and getattr(self.engine.model, "paged", None)):
            return {}
        try:
            return dict(self.engine.model.session_usage())
        except Exception:  # noqa: BLE001 - status reporting must never break a turn
            return {}

    def _pool_summary(self) -> str:
        usage = self.pool_usage()
        if not usage:
            return "dense caches (no paging)"
        groups = ", ".join(f"{name} {used}/{total} blocks" for name, (used, total) in usage.items())
        return f"{groups}; ~{self.engine.tokens_left()} tokens free"

    def _log_pool_change(self) -> None:
        """Log the pool whenever a group's block count moves, i.e. when a session grew
        its KV pages (a step allocates as its sliding window or a compressor closes)."""
        usage = self.pool_usage()
        if usage != self._pool_seen:
            grew = [
                f"{name} {self._pool_seen.get(name, (0, 0))[0]}->{used}"
                for name, (used, _total) in usage.items()
                if self._pool_seen.get(name, (None, None))[0] != used
            ]
            if grew and self._pool_seen:
                logger.debug(f"KV pages allocated: {', '.join(grew)} ({self._pool_summary()})")
            self._pool_seen = usage

    def _check_capacity(self, user: UserSession, ids: list[int]) -> None:
        engine = self.engine
        if user.pos + len(ids) >= engine.max_seq:
            raise ContextFull(f"user {user.index} needs {user.pos + len(ids)} of {engine.max_seq} tokens")
        if len(ids) > engine.tokens_left():
            raise ContextFull(
                f"the shared cache pool has room for about {engine.tokens_left()} more tokens, "
                f"this turn needs {len(ids)}"
            )

    # -- one round -------------------------------------------------------------- #
    def _round(self) -> None:
        self.rounds += 1
        before = self.steps
        self._round_turns()
        self._account(self.steps - before)
        self._log_pool_change()

    def _account(self, steps: int) -> None:
        """Roll the step counters and, every few seconds, log the aggregate rate.

        ``step_rate`` is steps (tokens fed, prompt or generated) per second across all
        users; dividing by the turns that were active gives the per-user rate."""
        self._window_steps += steps
        elapsed = time.perf_counter() - self._window_t0
        if elapsed < 5.0:
            return
        self.step_rate = self._window_steps / elapsed if elapsed else 0.0
        if self._window_steps:
            active = self._active
            decoding = [t for t in active if t.phase == _Turn.DECODE]
            per_user = f"{self.step_rate / len(active):.2f}" if active else "-"
            logger.debug(
                f"round {self.rounds}: {self.step_rate:.2f} steps/s total, {per_user} steps/s/user "
                f"over {len(active)} active turns ({len(decoding)} decoding), "
                f"{self._window_steps} steps in {elapsed:.1f}s"
            )
        self._window_t0 = time.perf_counter()
        self._window_steps = 0

    def _round_turns(self) -> None:
        for turn in list(self._active):
            try:
                self._collect(turn)
            except Exception as e:  # noqa: BLE001 - see _abort_all: ordering is lost
                self._abort_all(e)
                return
            if turn.error is not None:
                self._fail(turn, turn.error)  # its in-flight steps have now been drained
                continue
            try:
                self._dispatch(turn)
            except Exception as e:  # noqa: BLE001 - one bad turn must not stop the others
                # Reported once the steps this turn already has in flight have been
                # collected, next round; dropping it now would strand their outputs.
                turn.error = e

    def _collect(self, turn: _Turn) -> None:
        """Read back every step this turn has in flight, in dispatch order.

        The turn's steps sit at the head of :attr:`_inflight` because the rounds walk
        the turns in a stable order and each turn's steps were dispatched
        contiguously, so its own are the oldest outstanding ones."""
        while turn.pending:
            assert self._inflight and self._inflight[0] is turn, "decode outputs read out of dispatch order"
            self._inflight.popleft()
            last_prompt_token = turn.pending.popleft()
            out = turn.eager.popleft() if turn.eager else self.engine.model.read_decoded_output().reshape(1, -1).float()
            if turn.phase == _Turn.DECODE or last_prompt_token:
                turn.next_id = turn.sampler(out) if turn.sampler is not None else int(out[0].argmax().item())
            if last_prompt_token:
                turn.phase = _Turn.DECODE
                turn.t_prefill_done = time.perf_counter()
                seconds = turn.prefill_seconds
                rate = f"{len(turn.ids) / seconds:.1f}" if seconds > 0 else "-"
                logger.debug(
                    f"user {turn.user_key!r}: prefill of {len(turn.ids)} tokens done in "
                    f"{seconds:.2f}s ({rate} tok/s), cache at {self.engine.users[turn.slot].pos} tokens"
                )

    def _dispatch(self, turn: _Turn) -> None:
        """Queue this turn's next steps, without waiting for their outputs."""
        user = self.engine.users[turn.slot]
        if turn.phase == _Turn.PREFILL:
            # A client that hangs up mid-prompt is not abandoned here: the prompt is fed
            # to the end so the KV cache matches the stored conversation, and the turn
            # then finishes below with an empty reply.
            n = min(self.prefill_chunk, turn.prompt_left)
            if turn.fed == 0:
                logger.debug(
                    f"user {turn.user_key!r}: prefilling {len(turn.ids)} tokens from pos {user.pos} "
                    f"in chunks of {self.prefill_chunk}"
                )
            user.activate()
            for _ in range(n):
                token_id = turn.ids[turn.fed]
                turn.fed += 1
                self._send(turn, user, token_id, last_prompt_token=turn.fed == len(turn.ids))
            return

        if turn.cancelled.is_set():  # client hung up: keep the partial reply
            self._finish(turn)
            return
        if turn.next_id == self.engine.eos_id:
            self._finish(turn)
            return
        if len(turn.generated) >= turn.max_tokens or user.pos >= self.engine.max_seq - 1:
            turn.hit_cap = True
            self._finish(turn)
            return
        turn.generated.append(turn.next_id)
        turn.stream.push(turn.generated)
        if len(turn.generated) % 32 == 0:
            logger.debug(
                f"user {turn.user_key!r}: {len(turn.generated)}/{turn.max_tokens} tokens "
                f"at {turn.decode_rate:.2f} tok/s, cache at {user.pos}/{self.engine.max_seq}"
            )
        user.activate()
        self._send(turn, user, turn.next_id, last_prompt_token=False)

    def _send(self, turn: _Turn, user: UserSession, token_id: int, last_prompt_token: bool) -> None:
        """Dispatch one traced step for the (already activated) session.

        ``activate_session`` is what makes the interleaving safe: it repoints the page
        tables and swaps in this session's compressor window state, ordered on the
        command queue behind the trace replays already queued for other sessions."""
        model = self.engine.model
        if self.engine.traced:
            model.decode_traced_async(int(token_id), int(user.pos))
        else:
            # No async dispatch on the eager path: the step runs to completion here and
            # its logits wait in the turn (``--no-trace`` is single-user anyway).
            hidden = model.decode(int(token_id), int(user.pos), self.engine.rope)
            with _region("LM_HEAD"):
                turn.eager.append(ttnn.to_torch(self.engine.lm_head(hidden)).reshape(1, -1).float())
        user.pos += 1
        turn.pending.append(last_prompt_token)
        self._inflight.append(turn)
        self.steps += 1

    # -- completion ------------------------------------------------------------- #
    def _finish(self, turn: _Turn) -> None:
        """Close out a finished turn: store the assistant message and report the stats.

        Only called with no steps of this turn in flight, so its session is quiescent
        and the next turn for that user can be admitted safely."""
        user = self.engine.users[turn.slot]
        turn.t_done = time.perf_counter()
        turn.stream.close()
        user.pending_id = turn.next_id  # produced but never fed; the next turn starts with it
        assistant: dict = {"role": "assistant", "content": turn.stream.content}
        if turn.stream.reasoning:
            assistant["reasoning_content"] = turn.stream.reasoning
        user.messages.append(assistant)
        user._next_render = len(user.messages)
        self._retire(turn)
        logger.info(
            f"user {turn.user_key!r} (slot {user.index}): prefill {len(turn.ids)} tokens in "
            f"{turn.prefill_seconds:.2f}s, decoded {len(turn.generated)} at {turn.decode_rate:.2f} tok/s, "
            f"context {user.pos}/{self.engine.max_seq}, finish={'length' if turn.hit_cap else 'stop'}"
            f"{', client gone' if turn.cancelled.is_set() else ''}, {len(self._active)} turns still active"
        )
        logger.debug(f"user {turn.user_key!r} finished: pool {self._pool_summary()}")
        turn.events.put(
            (
                "done",
                {
                    "prompt_tokens": len(turn.ids),
                    "completion_tokens": len(turn.generated),
                    "content": turn.stream.content,
                    "reasoning_content": turn.stream.reasoning,
                    "finish_reason": "length" if turn.hit_cap else "stop",
                },
            )
        )

    def _fail(self, turn: _Turn, exc: Exception) -> None:
        """Drop a turn whose step raised. Its session keeps whatever was written; the
        client is told, and the remaining turns keep generating."""
        logger.warning(f"turn for user {turn.user_key!r} failed: {exc}")
        self._retire(turn)
        turn.events.put(("error", exc))

    def _abort_all(self, exc: Exception) -> None:
        """Give up on every active turn after a failed readback.

        A read that fails leaves an unknown number of steps in flight, so which output
        belongs to which turn can no longer be established and no turn can be trusted to
        continue. Generation stops here rather than serving one user another user's
        logits; the process must be restarted (a wedged device would not recover anyway).
        """
        logger.error(f"decode output readback failed, abandoning {len(self._active)} turns: {exc}")
        self._broken = exc
        for turn in list(self._active):
            self._fail(turn, exc)
        self._inflight.clear()

    def _drain(self, exc: Exception) -> None:
        """Release everyone still waiting on a reply, on shutdown."""
        with self._wake:
            pending, self._new = list(self._new), deque()
        for turn in list(self._active) + pending:
            self._retire(turn)
            turn.events.put(("error", exc))

    def _retire(self, turn: _Turn) -> None:
        if turn in self._active:
            self._active.remove(turn)


class GenerationServer:
    """The resident model plus the user/session bookkeeping behind the HTTP layer.

    One :class:`ChatEngine` (weights, RoPE tables, decode traces) serves every request.
    The engine claimed one paged KV session per ``--num-users`` at startup, and
    :class:`_SlotPool` hands those out a turn at a time, preferring the slot that already
    holds the conversation the request continues.

    Turns run concurrently: an HTTP thread validates the request, claims a slot, hands a
    :class:`_Turn` to the :class:`_Scheduler` and then does nothing but relay the reply as
    it arrives, while the scheduler thread interleaves every active turn's decode steps on
    the device.
    """

    def __init__(
        self,
        engine: ChatEngine,
        model_id: str,
        max_body_bytes: int = 16 << 20,
        prefill_chunk: int = 16,
    ):
        self.engine = engine
        self.model_id = model_id
        self.max_body_bytes = max_body_bytes
        self.pool = _SlotPool(engine)
        self._created = int(time.time())
        self.scheduler = _Scheduler(self, prefill_chunk)

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.pool.close()  # release anyone waiting for a slot before the threads go
        self.scheduler.stop()

    # -- status ----------------------------------------------------------------- #
    def stats(self) -> dict:
        """Everything the status pane shows, in one snapshot (see :mod:`demo.tui`)."""
        stats = self.scheduler.status()
        stats.update(
            {
                "model_id": self.model_id,
                "uptime": time.time() - self._created,
                "slots": len(self.engine.users),
                "max_seq": self.engine.max_seq,
                "users": self.pool.rows(),
            }
        )
        return stats

    # -- users / sessions ------------------------------------------------------- #
    def reset_user(self, user_key: str) -> None:
        """Rewind the conversations ``user_key`` last held, and their KV pages.

        Reserves each slot first, so it waits for a turn in flight rather than pulling
        the pages out from under it."""
        slots = self.pool.slots_of(user_key)
        if not slots:
            logger.debug(f"reset for {user_key!r}: it holds no slot, nothing to do")
            return
        for slot in slots:
            self.pool.reserve(slot)
            try:
                before = self.engine.users[slot].pos
                self.scheduler.run_on_scheduler(self.engine.users[slot].reset)
                logger.debug(
                    f"user {user_key!r} (slot {slot}) reset: {before} tokens released, "
                    f"pool {self.scheduler._pool_summary()}"
                )
            finally:
                self.pool.release(slot)

    def session_rows(self) -> list[dict]:
        return self.pool.rows()

    # -- conversation sync ------------------------------------------------------ #
    def _sync_messages(self, user: UserSession, messages: list[dict]) -> bool:
        """Align the session's stored conversation with the request's history.

        Returns whether the request is a continuation. When the stored conversation
        is a strict prefix of the incoming messages the request continues it: only
        the new user turn will be fed -- the KV pages already hold the past.
        Anything else (a first turn, a foreign client, an edited history) rewinds
        the session so the next turn re-prefills from the request, keeping the
        session in step with whatever the client sends.
        """
        stored = user.messages
        if stored and len(messages) > len(stored) and messages[: len(stored)] == stored:
            user.messages.extend(messages[len(stored) : -1])
            return True
        user.reset()
        user.messages = list(messages[:-1])
        user._next_render = 0
        return False

    # -- generation ------------------------------------------------------------- #
    def generate(self, user_key: str, body: dict, on_chunk) -> dict:
        """Run one chat-completion turn for ``user_key``.

        ``on_chunk(reasoning, content)`` receives the reply's new text as it resolves
        (either argument can be empty); raising ``_ClientGone`` aborts the turn, keeping
        the reply generated so far.

        Blocks until the reply is complete, but holds nothing except this user's own
        lock: the decode steps run on the scheduler thread, interleaved with the other
        users' turns, and this thread only relays what they produce."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RequestError(400, "messages must be a non-empty array")
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "user" or not isinstance(last.get("content"), str):
            raise RequestError(400, "the final message must be a user message with string content")

        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.engine.max_new_tokens
        else:
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                raise RequestError(400, "max_tokens must be a positive integer")
            if max_tokens < 1:
                raise RequestError(400, "max_tokens must be a positive integer")

        sampler = _make_sampler(body.get("temperature"), body.get("top_p"))
        # A slot is claimed per turn, not per user, so two requests run concurrently even
        # when they carry the same ``user`` (or none). Held until the reply is complete:
        # the turn decodes into this slot's KV session.
        slot = self.pool.acquire(user_key, messages)
        try:
            turn = _Turn(user_key, slot, body, sampler, max_tokens)
            self.scheduler.submit(turn)
            return self._relay(turn, on_chunk)
        finally:
            self.pool.release(slot)

    @staticmethod
    def _relay(turn: _Turn, on_chunk) -> dict:
        """Forward a turn's events to ``on_chunk`` until it finishes.

        A client that hangs up (``_ClientGone`` out of ``on_chunk``) cancels the turn but
        does not abandon it: the loop keeps draining, silently, until the scheduler
        reports the turn done, so the session is left consistent and its steps in flight
        are still read off the socket in order."""
        while True:
            kind, *payload = turn.events.get()
            if kind == "delta":
                if turn.cancelled.is_set():
                    continue
                try:
                    on_chunk(*payload)
                except _ClientGone:
                    turn.cancelled.set()
            elif kind == "error":
                raise payload[0]
            else:  # done
                return payload[0]

    @staticmethod
    def _apply_options(user: UserSession, body: dict) -> None:
        if "thinking" in body:
            user.thinking_mode = "thinking" if body.get("thinking") else "chat"
        effort = body.get("reasoning_effort")
        if effort is not None:
            if effort not in ("high", "max"):
                raise RequestError(400, "reasoning_effort must be 'high' or 'max'")
            user.reasoning_effort = effort

    def _render_ids(self, user: UserSession, include_assistant: bool) -> list[int]:
        """Tokens for the not-yet-encoded part of ``user.messages``.

        With ``include_assistant`` (a fresh re-prefill) every message is rendered,
        past assistant turns included. Otherwise (a continuation) assistant turns
        are skipped -- the model's own tokens are in the cache already -- and the
        turn starts with the pending token that closed the previous one."""
        engine = user.engine
        first_turn = user._next_render == 0
        rendered = "".join(
            render_message(i, user.messages, user.thinking_mode, reasoning_effort=user.reasoning_effort)
            for i in range(user._next_render, len(user.messages))
            if include_assistant or user.messages[i]["role"] != "assistant"
        )
        ids = list(engine.tokenizer(rendered, add_special_tokens=first_turn)["input_ids"])
        if include_assistant:
            return ids
        prefix: list[int] = []
        if user.pending_id is not None:
            prefix.append(user.pending_id)
            if user.pending_id != engine.eos_id:
                prefix.append(engine.eos_id)  # close an assistant turn that hit the token cap
        return prefix + ids


class _Handler(BaseHTTPRequestHandler):
    """One HTTP connection, served on its own thread by :class:`_ModelHTTPServer`.

    The thread does no device work: it validates the request, hands a turn to the
    scheduler and relays the reply, so requests from different users generate
    concurrently (see :class:`_Scheduler`).
    """

    protocol_version = "HTTP/1.1"
    server_version = "DeepSeekV4Flash/1.0"

    @property
    def api(self) -> GenerationServer:
        return self.server.api  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        logger.info(f"{self.address_string()} {fmt % args}")

    # -- response plumbing ------------------------------------------------------ #
    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception) -> None:
        status, message, err_type = _error_parts(exc)
        logger.warning(f"{self.command} {self.path} -> {status} {err_type}: {message}")
        self._send_json({"error": {"message": message, "type": err_type, "code": status}}, status)

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        # The client reads the stream to EOF, so the connection must close rather
        # than be reused for another request.
        self.close_connection = True

    def _sse_event(self, obj: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- request parsing -------------------------------------------------------- #
    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        n = int(length) if length and length.isdigit() else 0
        if n > self.api.max_body_bytes:
            raise RequestError(413, "request body too large")
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RequestError(400, f"malformed JSON body: {e}") from e

    # -- completions ------------------------------------------------------------ #
    def _completion(self, body: dict, *, chat: bool) -> None:
        """Serve one ``/v1/chat/completions`` (``chat=True``) or ``/v1/completions``
        (legacy, ``chat=False``) request, streaming via SSE when ``stream`` is set."""
        if chat:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RequestError(400, "messages must be a non-empty array")
            last = messages[-1]
            if not isinstance(last, dict) or last.get("role") != "user" or not isinstance(last.get("content"), str):
                raise RequestError(400, "the final message must be a user message with string content")
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            object_kind, chunk_kind = "chat.completion", "chat.completion.chunk"
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RequestError(400, "prompt must be a non-empty string")
            body = {**body, "messages": [{"role": "user", "content": prompt}]}
            completion_id = "cmpl-" + uuid.uuid4().hex
            object_kind, chunk_kind = "text_completion", "text_completion"

        stream = bool(body.get("stream"))
        created = int(time.time())
        model = str(body.get("model") or self.api.model_id)
        user_key = str(body.get("user") or _DEFAULT_USER)

        def frame(delta: dict, finish) -> dict:
            return {
                "id": completion_id,
                "object": chunk_kind,
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        if stream:
            self._sse_start()
            first = [True]
            stopped = [False]

            def on_chunk(reasoning: str, content: str) -> None:
                if stopped[0] or (not reasoning and not content):
                    return
                if chat:
                    delta: dict = {}
                    if first[0]:
                        delta["role"] = "assistant"
                        first[0] = False
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if content:
                        delta["content"] = content
                else:
                    delta = {"text": reasoning + content}
                try:
                    self._sse_event(frame(delta, None))
                except (BrokenPipeError, ConnectionResetError):
                    stopped[0] = True
                    raise _ClientGone()

            try:
                stats = self.api.generate(user_key, body, on_chunk)
            except _ClientGone:
                return
            except Exception as e:
                status, message, err_type = _error_parts(e)
                logger.warning(f"{self.command} {self.path} -> {status} {err_type}: {message}")
                try:
                    self._sse_event({"error": {"message": message, "type": err_type, "code": status}})
                    self._sse_done()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            try:
                self._sse_event(frame({}, stats["finish_reason"]))
                self._sse_done()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            stats = self.api.generate(user_key, body, lambda r, c: None)
        except Exception as e:
            self._send_error_json(e)
            return
        if chat:
            message = {"role": "assistant", "content": stats["content"]}
            if stats["reasoning_content"]:
                message["reasoning_content"] = stats["reasoning_content"]
            choices = [{"index": 0, "message": message, "finish_reason": stats["finish_reason"]}]
        else:
            choices = [
                {"index": 0, "text": stats["content"], "logprobs": None, "finish_reason": stats["finish_reason"]}
            ]
        self._send_json(
            {
                "id": completion_id,
                "object": object_kind,
                "created": created,
                "model": model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                },
            }
        )

    # -- routing ---------------------------------------------------------------- #
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/health", "/healthz"):
                self._send_json(
                    {
                        "status": "ok",
                        "model": self.api.model_id,
                        "sessions": len(self.api.engine.users),
                        "active_turns": self.api.scheduler.active_turns,
                        "uptime": time.time() - self.api._created,
                    }
                )
            elif path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.api.model_id,
                                "object": "model",
                                "created": self.api._created,
                                "owned_by": _VENDOR,
                            }
                        ],
                    }
                )
            elif path == "/v1/sessions":
                self._send_json({"object": "list", "data": self.api.session_rows()})
            else:
                raise RequestError(404, f"unknown endpoint: GET {path}")
        except Exception as e:
            self._send_error_json(e)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        try:
            if not path.startswith("/v1/sessions/"):
                raise RequestError(404, f"unknown endpoint: DELETE {path}")
            user_key = unquote(path[len("/v1/sessions/") :])
            if not user_key:
                raise RequestError(400, "missing user in path")
            self.api.reset_user(user_key)
            self._send_json({"object": "session", "id": user_key, "deleted": True})
        except Exception as e:
            self._send_error_json(e)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/v1/chat/completions":
                self._completion(body, chat=True)
            elif path == "/v1/completions":
                self._completion(body, chat=False)
            else:
                raise RequestError(404, f"unknown endpoint: POST {path}")
        except Exception as e:
            self._send_error_json(e)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()


class _ModelHTTPServer(ThreadingHTTPServer):
    """A threaded HTTP server carrying the :class:`GenerationServer`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, api: GenerationServer):
        super().__init__(address, handler)
        self.api = api


def _add_model_args(p: argparse.ArgumentParser) -> None:
    """The model/engine flags, identical to ``chat_cli.parse_args``."""
    from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import _DEFAULT_MODEL_DIR

    p.add_argument("--model-dir", default=_DEFAULT_MODEL_DIR, help="HF snapshot (or hub cache) of the checkpoint")
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache"),
        help="converted ttnn weight-tile cache, reused across runs ('' disables it)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", "0")) or None,
        help="cap the decoder stack (the full 43 layers do not fit one Blackhole)",
    )
    p.add_argument(
        "--num-users",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_NUM_USERS", "8")),
        help="turns that can generate at once, each with its own KV session (fixed at "
        "startup: their cache blocks cannot be allocated once the traces exist). Note "
        "the rounds interleave rather than batch, so the device's token rate is shared: "
        "more users means more total throughput but a slower reply each",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_CONTEXT", "16384")),
        help="tokens (all turns) one user's caches are addressed for; rounded up. The "
        "model handles 131072, but that much per user costs page-table width and pool "
        "blocks for every session, so the default is sized for a busy server",
    )
    p.add_argument(
        "--total-context",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_TOTAL_CONTEXT", "0")) or None,
        help="total tokens the shared block pool holds across all users "
        "(default: --num-users x --max-context, i.e. every user can fill its context)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "2048")),
        help="cap on the tokens generated per reply",
    )
    p.add_argument(
        "--system-prompt",
        default=os.environ.get("DEEPSEEK_V4_SYSTEM_PROMPT", ""),
        help="system prompt prefixed to every user's conversation",
    )
    p.add_argument(
        "--system-prompt-file",
        help="read the system prompt from this file instead of --system-prompt",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help="thinking mode: the reply is preceded by a <think> reasoning block",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default=None,
        help="reasoning-effort hint, only meaningful with --think",
    )
    p.add_argument("--trace-region-size", type=int, default=int(os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE", "0")))
    p.add_argument("--no-trace", dest="traced", action="store_false", help="eager decode instead of traced decode")
    p.add_argument(
        "--no-prefetcher",
        dest="prefetcher",
        action="store_const",
        const=False,
        default=None,
        help="feed the attention projections with a DRAM->L1 copy per call instead of the "
        "DRISC tensor prefetcher (default: use it wherever the device supports it)",
    )
    p.add_argument("--quiet", action="store_true", help="only warnings and above from the model logs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_model_args(p)
    p.add_argument(
        "--host", default=os.environ.get("DEEPSEEK_V4_HOST", "0.0.0.0"), help="interface to bind (default: all)"
    )
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("DEEPSEEK_V4_PORT", "8000")), help="TCP port (default: 8000)"
    )
    p.add_argument(
        "--model-id",
        default=os.environ.get("DEEPSEEK_V4_MODEL_ID", "deepseek-v4-flash"),
        help="model id advertised by /v1/models and echoed in completions",
    )
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_PREFILL_CHUNK", "16")),
        help="prompt tokens a turn feeds per scheduling round (their logits are discarded, "
        "so they pipeline freely); higher prefills faster, lower keeps replies smoother "
        "for users already generating",
    )
    p.add_argument(
        "--max-body-bytes",
        type=int,
        default=16 << 20,
        help="largest accepted request body in bytes (default: 16 MiB)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="log every scheduler event: admissions, slot assignments, KV page allocations, "
        "prefill lengths and per-user decode rates (in the live console, the 'd' key toggles it)",
    )
    p.add_argument(
        "--no-tui",
        dest="tui",
        action="store_false",
        help="plain line-by-line logging instead of the live split console (implied when "
        "stdout is not a terminal, e.g. under nohup or in CI)",
    )
    args = p.parse_args(argv)
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).expanduser().read_text().strip()
    if args.reasoning_effort and not args.think:
        p.error("--reasoning-effort requires --think")
    if args.num_users < 1:
        p.error("--num-users must be at least 1")
    # Multiple users need the paged caches, which only the traced path has: the eager
    # path decodes against one dense cache per layer.
    if not args.traced and args.num_users > 1:
        p.error("--no-trace supports a single user; drop --num-users or --no-trace")
    if args.total_context is None:
        # Every user able to fill its own context, rather than all of them sharing one
        # context's worth of blocks (which made a busy server hand out 429s early).
        args.total_context = args.num_users * args.max_context
    elif args.total_context < args.max_context:
        p.error(
            f"--total-context {args.total_context} is below --max-context {args.max_context}: "
            "no user could ever fill its context"
        )
    if not 0 < args.port < 65536:
        p.error("--port must be a TCP port number")
    if args.prefill_chunk < 1:
        p.error("--prefill-chunk must be at least 1")
    if args.quiet and args.debug:
        p.error("--quiet and --debug ask for opposite things")
    return args


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The model emits plenty of non-ASCII (CJK, emoji, typographic punctuation), which
    # an ASCII/POSIX locale would turn into a UnicodeEncodeError mid-reply.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # loguru's default sink is DEBUG, which would make the scheduler's per-round lines the
    # normal output; they are opt-in via --debug (or the console's 'd' key) instead.
    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "DEBUG" if args.debug else "INFO")
    if not Path(args.model_dir).expanduser().exists():
        print(f"checkpoint not found: {args.model_dir}", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    logger.info(
        f"KV budget: {args.num_users} concurrent users x {args.max_context} tokens, "
        f"shared pool {args.total_context} tokens. The rounds interleave rather than batch, so "
        f"expect the device's token rate to be split across the users that are decoding."
    )
    with open_mesh_device(args.trace_region_size) as mesh_device:
        # The prefetcher session spans the trace capture, the warmup and every served
        # turn, so it is opened inside ``ChatEngine`` (once the model exists) against
        # this stack, and the senders are stopped before the mesh device is closed.
        with contextlib.ExitStack() as prefetcher:
            engine = ChatEngine(mesh_device, args, prefetcher)
            engine.warmup()
            api = GenerationServer(engine, args.model_id, args.max_body_bytes, args.prefill_chunk)
            # The scheduler is the only thread that touches the device from here on; the
            # traces it replays were captured by the warmup above, which a pipelined
            # caller cannot do mid-flight.
            api.start()
            httpd = _ModelHTTPServer((args.host, args.port), _Handler, api)
            # The live console owns the logger from here on, so it is opened after the
            # (long, chatty) model build and warmup have printed normally. ``shutdown``
            # has to come from another thread than ``serve_forever``, which the console's
            # key thread is.
            with tui.console(
                logger,
                api.stats,
                on_quit=lambda: threading.Thread(target=httpd.shutdown).start(),
                enabled=args.tui,
                debug=args.debug,
            ):
                logger.info(
                    f"serving {args.model_id!r} on http://{args.host}:{args.port} "
                    f"({len(engine.users)} concurrent sessions, context {engine.max_seq}/user, "
                    f"prefill chunk {args.prefill_chunk})"
                )
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    logger.info("interrupted, shutting down")
                finally:
                    httpd.server_close()
                    api.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

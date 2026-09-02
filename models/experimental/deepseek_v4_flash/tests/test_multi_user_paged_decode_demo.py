# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-user paged-KV decode demo for ``DeepSeekV4Model`` (batched traced path).

``DEEPSEEK_V4_NUM_USERS`` independent conversations share one model, one block pool
per layer and one set of captured decode traces. Each is a *session* on the model: its
KV lives in blocks addressed through row ``u`` of a per-layer ``page_table``, so the
users are kept apart by that table rather than by separate caches.

The users are served on two levels, which is the point of the demo:

  * **Within a step** -- ``DEEPSEEK_V4_DECODE_BATCH`` users are decoded *together*.
    The traces are captured at that batch, so one replay carries B tokens through the
    stack and returns B rows of logits. The batch slots index the shared pool
    independently, so B unrelated conversations ride a single trace.
  * **Across steps** -- the users are split into ``NUM_USERS / DECODE_BATCH`` batches
    that take turns, one batched step each per round. Switching batch repoints the
    page tables and swaps that batch's compressor window rows into the slots; the
    caches themselves never move, so no re-capture is needed. The default 8 users per
    step over 8 batches serves 64 conversations on one set of traces.

The users of a *batch* advance in lockstep on a shared position, which a trace requires
(it bakes in one compressor-pooling schedule and one SDPA mode -- see
``model._variant_key``). Prompts of different lengths are *not* padded to make that
true; instead every user feeds its own prompt token while it has one and its own
sampled token after that, all at the same absolute position. A user with a short prompt
simply starts generating earlier than one with a long prompt. Separate batches are
independent and need not agree on anything.

Every batch keeps a step in flight, with the logits read back on a thread of their own
(see ``_OutputReader``, which is what makes running them all at once safe). Reported
throughput: steps/s, and the B tokens per step that implies.

The assertion that matters is the interleaving one: neither the other users of a batch
nor the batch being swapped out and back in every round may change what user 0 says. So
one batch is re-run on its own, with user 0's prompt unchanged and *different* prompts
in the other slots, and user 0's tokens must come out identical -- if the sessions
shared a block, or the compressor window rows leaked across slots or failed to survive
a swap, the two runs diverge. (That the paged reads themselves match a dense cache is
covered at the op level by ``test_paged_kv_equivalence.py``, and that attention is
batch-invariant by ``test_attention_batching.py``.)

Run (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    DEEPSEEK_V4_NUM_USERS=64 DEEPSEEK_V4_DECODE_BATCH=8 \\
    DEEPSEEK_V4_MAX_NEW_TOKENS=64 \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_multi_user_paged_decode_demo.py
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
_TOPICS = ["movies", "tv shows", "books", "video games", "songs", "cartoons", "podcasts", "board games"]
_ALT_TOPICS = ["bicycles", "sandwiches", "planets", "card tricks", "mountains", "typefaces", "bridges", "teas"]
_PROMPT_TEMPLATE = (
    "Tell me the name of the top {count} {topic} of all time. Also list out the top {count} worst {topic} of "
    "all time. Give me details of why you choose those {topic}. Try to make your response as humours as possible."
)
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")
# Users decoded in one traced step, and users in total. The latter must be a whole
# number of the former: every step decodes a full batch (a trace runs all of its slots
# unconditionally), so a partly filled batch has nowhere safe to write its KV.
_DECODE_BATCH = int(os.environ.get("DEEPSEEK_V4_DECODE_BATCH", "8"))
_NUM_USERS = int(os.environ.get("DEEPSEEK_V4_NUM_USERS", "64"))
_MAX_NEW_TOKENS = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "64"))
# Steps allowed in flight at once across the batches; 0 means one per batch, which is as
# many as the round-robin can produce. A cap is only needed to *diagnose* the socket
# backpressure discussed in :class:`_OutputReader` -- with the reader thread draining the
# output socket, running every batch in flight is both safe and fastest.
_PIPELINE_DEPTH = int(os.environ.get("DEEPSEEK_V4_PIPELINE_DEPTH", "0"))
_PAGE_BLOCK_SIZE = 32
# Generated tokens compared between the two batches of the isolation re-run below.
_ISOLATION_STEPS = int(os.environ.get("DEEPSEEK_V4_ISOLATION_STEPS", "32"))
# The DRISC prefetcher reserves ~342 KB of L1 per receiver core for its global circular
# buffers, which a wide batch may want back: every user's row is padded to a whole tile,
# so the per-step tensors grow with the batch and can leave an op's own circular buffers
# nowhere to go. Unset lets the model decide by what the device supports.
_PREFETCHER = os.environ.get("DEEPSEEK_V4_PREFETCHER")
# What the isolation re-run puts in the slots *other* than user 0's. "alt" (the
# default) gives them different prompts, which is the property under test. "same"
# repeats user 0's own batch, a control that tells a genuine cross-user leak apart from
# a session-state bug when the assertion fires.
_ISOLATION_TOPICS = os.environ.get("DEEPSEEK_V4_ISOLATION_TOPICS", "alt")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _w(loader: DeepseekV4WeightLoader, name: str):
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _build_rope(config, max_seq: int) -> dict:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    dummy = torch.zeros(1, max_seq, 1, dtype=torch.float32)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(torch.float32)

    def half(layer_type: str, position_ids: torch.Tensor):
        cos, sin = rotary(dummy, position_ids=position_ids, layer_type=layer_type)
        return cos[0].contiguous(), sin[0].contiguous()

    positions = torch.arange(max_seq).unsqueeze(0)
    rope = {"main": half("main", positions), "compress": half("compress", positions), "win": {}}
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        win_pos = (torch.arange(max_seq // cr) * cr).unsqueeze(0)
        rope["win"][cr] = half("compress", win_pos)
    return rope


@dataclass
class UserSession:
    """One conversation: a model session riding slot :attr:`slot` of batch :attr:`group`."""

    user_id: int
    group: int
    slot: int
    sid: int
    prompt_ids: list[int]
    generated: list[int] = field(default_factory=list)
    next_token: int = 0
    done: bool = False

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    def feed(self, pos: int) -> int:
        """The token this user puts into the step at ``pos``: its next prompt token
        while the prompt lasts, else the token it sampled last.

        A finished user keeps feeding too. Its slot is decoded either way -- a trace
        runs every slot unconditionally -- so there is nothing to gain by feeding it
        something special, and its output is simply not recorded.
        """
        return self.prompt_ids[pos] if pos < self.prompt_len else self.next_token

    def takes_output_at(self, pos: int) -> bool:
        """Is the step at ``pos`` generating for this user? True from the step that
        consumes its last prompt token onwards -- that step's sample is its first
        generated token."""
        return pos >= self.prompt_len - 1


@dataclass
class Batch:
    """The users of one traced step, and the position they share.

    A batch is the unit the model seats: :meth:`DeepSeekV4Model.activate_sessions` puts
    these sessions in the slots, and one traced step advances all of them.
    """

    index: int
    sessions: list[UserSession]
    pos: int = 0
    # Position of the step this batch has in flight, whose logits are waiting on the
    # output socket to be picked up a round later. ``None`` when nothing is in flight.
    pending_pos: int | None = None

    @property
    def done(self) -> bool:
        return all(s.done for s in self.sessions)

    @property
    def sids(self) -> list[int]:
        return [s.sid for s in self.sessions]

    def can_dispatch(self, max_seq: int) -> bool:
        return not self.done and self.pos < max_seq

    def dispatch(self, model) -> None:
        """Seat this batch and enqueue its next step without waiting for the logits."""
        model.activate_sessions(self.sids)
        model.decode_traced_async([s.feed(self.pos) for s in self.sessions], self.pos)
        self.pending_pos = self.pos
        self.pos += 1

    def collect(self, sampled: list[int], eos_id: int, max_new: int) -> None:
        """Take each user's own row of the in-flight step's samples."""
        pos = self.pending_pos
        assert pos is not None, "collect() without a step in flight"
        self.pending_pos = None
        for session in self.sessions:
            if session.done or not session.takes_output_at(pos):
                continue
            token = int(sampled[session.slot])
            session.next_token = token
            session.generated.append(token)
            if token == eos_id:
                logger.info(f"user {session.user_id} hit EOS after {len(session.generated)} tokens")
                session.done = True
            elif len(session.generated) >= max_new:
                session.done = True


def _tokenize_prompt(tokenizer, text: str) -> list[int]:
    prompt = render_message(0, [{"role": "user", "content": text}], "chat")
    return list(tokenizer(prompt)["input_ids"])


def _prompts(topics: list[str], num_users: int) -> list[str]:
    """One prompt per user: the topics cycle, and the count grows each time round, so
    no two users are handed the same text (and their prompts differ in length, which is
    what puts users at different points of their prompt inside one batch)."""
    return [
        _PROMPT_TEMPLATE.format(topic=topics[u % len(topics)], count=10 + u // len(topics)) for u in range(num_users)
    ]


def _build(
    mesh_device, num_users: int, batch: int, prefetcher: contextlib.ExitStack, max_seq: int, config, loader
) -> DeepSeekV4Model:
    """Build the model and set it up to decode ``batch`` users per traced step.

    ``prefetcher`` is the caller's stack, which owns the DRISC prefetcher session so it
    spans every step of the run. No session is opened here -- :func:`_open_batch` does
    that, so the caller decides how many batches share the pool.
    """
    rope = _build_rope(config, max_seq)
    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    top_cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None

    model = DeepSeekV4Model(
        config,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
        use_prefetcher=None if _PREFETCHER is None else _PREFETCHER == "1",
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    # One session for the whole run, not one per step: starting the DRISC senders is not
    # free and each GCB's ring state carries across steps. A no-op when the model was
    # built without the prefetcher.
    prefetcher.enter_context(model.prefetcher_session())
    logger.info(f"tensor prefetcher: {'on' if model.use_prefetcher else 'off'}")

    # Every user needs a session of its own, plus one spare batch for the isolation
    # re-run at the end (which gets untouched blocks rather than recycling the first
    # batch's, so that what it compares is the effect of the *other users* and not of
    # what a freed block last held).
    sessions = num_users + batch
    model.prepare_static_decode(
        rope,
        max_seq,
        lm_head=lm_head,
        num_sessions=sessions,
        total_tokens=sessions * max_seq,
        block_size=_PAGE_BLOCK_SIZE,
        batch=batch,
    )
    return model


def _warmup(model, batch: int) -> None:
    """Capture the decode traces on throwaway sessions, before any session under test.

    Capturing runs each variant's programs for real -- that is how they are JITed --
    and those runs write KV wherever the seated session points. At position 0 the
    window-pooling variants write their (not yet meaningful) entry into the last row of
    the sliding ring, which subsequent steps *do* read, so the batch that happens to be
    seated during capture is not comparable with one that runs afterwards. Spending a
    throwaway set of sessions here keeps the measured batches clean and identical.
    """
    scratch = [model.open_session() for _ in range(batch)]
    model.activate_sessions(scratch)
    model.decode_traced([0] * batch, 0)
    for sid in scratch:
        model.close_session(sid)


def _open_batch(model, index: int, prompt_ids: list[list[int]], first_user: int) -> Batch:
    """Open one session per prompt and bind them to the slots of batch ``index``.

    The sessions are not seated here: with several batches taking turns, that happens
    per step in :func:`_generate`.
    """
    sessions = [
        UserSession(user_id=first_user + u, group=index, slot=u, sid=model.open_session(), prompt_ids=ids)
        for u, ids in enumerate(prompt_ids)
    ]
    return Batch(index=index, sessions=sessions)


class _OutputReader:
    """A thread that does nothing but drain the output socket, so the device never has
    to wait for the host to get round to reading.

    Worth the thread because of how small the socket's FIFO is: a single pinned host page
    (``model._OUT_FIFO_BYTES``), against a couple of megabytes of logits per step. So the
    sender kernel inside the last submesh's trace stalls almost immediately unless
    someone is reading, and steps queued behind it back up through the cross-submesh
    sockets until every submesh's command queue is full. If the thread that reads is also
    the thread that dispatches, that is a deadlock and not merely a stall: the host blocks
    queueing the next step's work while the device waits for the host to read. Reading
    from its own thread breaks the cycle -- ``read_tensor`` drops the GIL for the blocking
    part, so the dispatching thread runs on regardless of how long a read waits.

    Each dispatched step is announced with :meth:`expect` and its per-user sampled tokens
    come back from :meth:`take` in the same order (the socket is a FIFO, and one reader
    keeps it that way). The argmax runs here too, which keeps that host work off the
    dispatching thread as well.
    """

    def __init__(self, model, batch_size: int):
        self._model = model
        self._batch_size = batch_size
        self._tickets: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="d2h-reader", daemon=True)

    def _run(self) -> None:
        for batch in iter(self._tickets.get, None):
            try:
                logits = self._model.read_decoded_output()
                sampled = logits.reshape(self._batch_size, -1).float().argmax(dim=-1).tolist()
            except BaseException as exc:  # re-raised on the dispatching thread by take()
                self._results.put(exc)
                return
            self._results.put((batch, sampled))

    def expect(self, batch: Batch) -> None:
        """Announce a step, so the reader is already waiting on the socket for it."""
        self._tickets.put(batch)

    def take(self) -> tuple[Batch, list[int]]:
        """The oldest unread step's ``(batch, sampled token per slot)``."""
        item = self._results.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def __enter__(self) -> "_OutputReader":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._tickets.put(None)
        # Bounded: on the way out of a failed run the reader may be parked on a step that
        # will never arrive, and the daemon thread must not hold the test open for it.
        self._thread.join(timeout=10)


def _generate(
    model, batches: list[Batch], max_new: int, eos_id: int, max_seq: int, depth: int = _PIPELINE_DEPTH
) -> tuple[int, float]:
    """Round-robin the batches: one batched step each per round, until all users finish.

    A step decodes a whole batch at that batch's shared position, so a round advances
    every user of every batch by one token. Seating the next batch repoints the page
    tables and swaps its compressor window rows into the slots -- the KV itself stays
    where it is, which is what lets all the batches share one set of traces. The seating
    is safe to issue while earlier steps are still running: it is device work on the same
    command queue as the replays, so it lands between them in the order it was queued.

    Returns ``(steps, seconds)``; each step is one batch's worth of tokens.

    Every batch may have a step in flight at once, which is what the round-robin is for:
    steps of *one* batch cannot overlap (each user feeds back the token that step
    sampled), but different batches are independent, so the device is handed the next
    batch's step without waiting for the last one's logits. Those logits are read on
    :class:`_OutputReader`'s thread rather than here -- see there for why the reading
    cannot be left to this one. ``depth`` caps the steps in flight for diagnosing that;
    0 means one per batch, i.e. no cap.

    A batch is only made to wait for its *own* step. Results arrive in dispatch order, so
    waiting for one batch collects every earlier batch's on the way.
    """
    batch_size = len(batches[0].sessions)
    steps = 0
    rounds = 0
    elapsed = 0.0
    outstanding = 0
    # Split of the wall clock, to say which side the ceiling is on: ``dispatching`` is
    # this thread seating a batch and queueing its step, ``waiting`` is it blocked on a
    # step's logits with nothing else it may legally queue.
    dispatching = 0.0
    waiting = 0.0

    with _OutputReader(model, batch_size) as reader:

        def drain() -> None:
            nonlocal outstanding, waiting
            # Row ``u`` of the output is slot ``u``'s user, which is the whole point of
            # the batched step: one replay, B independent continuations.
            t0 = time.perf_counter()
            batch, sampled = reader.take()
            waiting += time.perf_counter() - t0
            batch.collect(sampled, eos_id, max_new)
            outstanding -= 1

        while outstanding or any(b.can_dispatch(max_seq) for b in batches):
            t0 = time.perf_counter()
            for batch in batches:
                while outstanding and (batch.pending_pos is not None or (depth and outstanding >= depth)):
                    drain()
                if batch.can_dispatch(max_seq):
                    t1 = time.perf_counter()
                    # Announced before dispatch so the reader is already parked on the
                    # socket when the step's first page lands.
                    reader.expect(batch)
                    batch.dispatch(model)
                    dispatching += time.perf_counter() - t1
                    outstanding += 1
                    steps += 1
            elapsed += time.perf_counter() - t0
            rounds += 1
            if rounds % 10 == 0 and elapsed > 0:
                live = [b for b in batches if b.can_dispatch(max_seq)]
                users = sum(len(b.sessions) for b in live)
                logger.info(
                    f"round {rounds:4d}: {steps / elapsed:.2f} steps/s, "
                    f"{steps * batch_size / elapsed:.2f} tok/s total over {len(live)} batches / {users} users "
                    f"({1e3 * dispatching / steps:.1f} ms/step dispatching, {1e3 * waiting / steps:.1f} waiting)"
                )
    return steps, elapsed


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_multi_user_paged_decode_demo(mesh_device, reset_seeds) -> None:
    """Decode ``DECODE_BATCH`` users per traced step, round-robin over enough batches
    to serve ``NUM_USERS`` conversations off one shared pool and one set of traces."""
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    assert _NUM_USERS % _DECODE_BATCH == 0, (
        f"NUM_USERS ({_NUM_USERS}) must be a whole number of batches of DECODE_BATCH "
        f"({_DECODE_BATCH}): a traced step decodes every slot it was captured with"
    )
    num_batches = _NUM_USERS // _DECODE_BATCH

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    prompts = _prompts(_TOPICS, _NUM_USERS)
    others = prompts if _ISOLATION_TOPICS == "same" else _prompts(_ALT_TOPICS, _DECODE_BATCH)
    alt_prompts = [prompts[0]] + others[1:_DECODE_BATCH]
    prompt_ids = [_tokenize_prompt(tokenizer, p) for p in prompts]
    alt_prompt_ids = [_tokenize_prompt(tokenizer, p) for p in alt_prompts]
    # One context length for every session the run opens, so the isolation batch needs
    # no re-capture: the traces are sized to ``max_seq``.
    longest = max(len(ids) for ids in prompt_ids + alt_prompt_ids)
    max_seq = round_context(longest + _MAX_NEW_TOKENS + 1, set(config.compress_rates.values()), _PAGE_BLOCK_SIZE)

    # Said before the slow parts rather than after: building the stack, allocating the
    # pool and capturing the traces take minutes on the full model and print little, so
    # without this the run looks stuck long before it has done anything wrong.
    longest_run = max(len(ids) for ids in prompt_ids) + _MAX_NEW_TOKENS
    in_flight = f"{_PIPELINE_DEPTH} steps in flight" if _PIPELINE_DEPTH else "a step per batch in flight"
    logger.info(
        f"batched paged decode: {_NUM_USERS} users as {num_batches} batches of {_DECODE_BATCH} "
        f"(one batch per step, round-robin, {in_flight}) "
        f"max_new_tokens={_MAX_NEW_TOKENS} block_size={_PAGE_BLOCK_SIZE} max_seq={max_seq}; "
        f"about {num_batches * longest_run} steps to run"
    )

    # The prefetcher session spans every step of the run, so it is opened inside
    # ``_build`` (once the model exists) against this stack.
    with contextlib.ExitStack() as prefetcher:
        t0 = time.perf_counter()
        model = _build(mesh_device, _NUM_USERS, _DECODE_BATCH, prefetcher, max_seq, config, loader)
        eos_id = config.eos_token_id
        logger.info(f"model built and pool allocated in {time.perf_counter() - t0:.1f}s; capturing traces")

        t0 = time.perf_counter()
        _warmup(model, _DECODE_BATCH)
        logger.info(f"traces captured in {time.perf_counter() - t0:.1f}s")
        batches = [
            _open_batch(model, i, prompt_ids[i * _DECODE_BATCH : (i + 1) * _DECODE_BATCH], i * _DECODE_BATCH)
            for i in range(num_batches)
        ]
        for batch in batches:
            lengths = ", ".join(str(s.prompt_len) for s in batch.sessions)
            logger.info(
                f"batch {batch.index}: users {batch.sessions[0].user_id}..{batch.sessions[-1].user_id} "
                f"({lengths} prompt tokens)"
            )
        logger.info(f"pool usage after seating: {model.session_usage()}")

        steps, elapsed = _generate(model, batches, _MAX_NEW_TOKENS, eos_id, max_seq)
        sessions = [s for b in batches for s in b.sessions]

        # --- isolation: neither the batch nor the swapping may change user 0 ---- #
        # User 0's prompt in slot 0, different prompts in the other slots, on its own
        # sessions and with no other batch to take turns with. If user 0 reproduces its
        # tokens, then its neighbours did not leak into it and its state survived being
        # swapped out and back in once per round above.
        reference = list(sessions[0].generated)
        alt_batch = _open_batch(model, num_batches, alt_prompt_ids, _NUM_USERS)
        _generate(model, [alt_batch], min(_ISOLATION_STEPS, _MAX_NEW_TOKENS), eos_id, max_seq)

    for session in batches[0].sessions:
        logger.info(f"USER {session.user_id} PROMPT    : {tokenizer.decode(session.prompt_ids)!r}")
        logger.info(
            f"USER {session.user_id} GENERATED : {tokenizer.decode(session.generated)!r} "
            f"({len(session.generated)} tokens)"
        )
    for batch in batches[1:]:
        counts = ", ".join(str(len(s.generated)) for s in batch.sessions)
        logger.info(f"batch {batch.index} generated token counts: {counts}")
    assert steps, "no steps were run"
    logger.info(
        f"batched decode throughput ({_DECODE_BATCH} users/step, {num_batches} batches, "
        f"{_NUM_USERS} users): {steps / elapsed:.2f} steps/s, "
        f"{steps * _DECODE_BATCH / elapsed:.2f} tok/s total, "
        f"{steps * _DECODE_BATCH / elapsed / _NUM_USERS:.2f} tok/s/user "
        f"({steps} steps in {elapsed:.2f}s)"
    )
    logger.info(f"pool usage after generation: {model.session_usage()}")

    compared = min(len(reference), len(alt_batch.sessions[0].generated))
    assert compared, "the isolation re-run generated nothing to compare"
    assert alt_batch.sessions[0].generated[:compared] == reference[:compared], (
        "user 0's tokens changed when it ran in a batch of its own with different "
        f"neighbours: {reference[:compared]} vs {alt_batch.sessions[0].generated[:compared]}"
    )
    logger.info(f"isolation: user 0 reproduced its first {compared} tokens in a different batch")
    # Whether two prompts produce different text is a property of the *model*, not of
    # the paging: a stack truncated by ``DEEPSEEK_V4_DECODE_LAYERS`` emits much the same
    # gibberish for any prompt. So it is logged, not asserted.
    if len(sessions) > 1 and all(s.generated == sessions[0].generated for s in sessions[1:]):
        logger.warning("all users produced identical tokens (expected on a heavily truncated stack)")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Interactive multi-user chat CLI on the full ttnn ``DeepSeekV4Model``.

Same decode engine as
``models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py``
(whose helpers this reuses), turned into a multi-turn REPL: the model, the RoPE
tables and the traced-decode buffers are built once, then every turn only the
*new* tokens are fed into the already-populated caches. There is no prefill op --
a turn's prompt tokens are replayed one decode step each at ascending absolute
positions -- so a follow-up question costs one step per new prompt token and
never recomputes the earlier turns.

Several **users** share that one model. Each is a numbered, independent
conversation (its own messages, position, system prompt and thinking mode) backed
by its own paged KV cache, and ``/user N`` switches between them mid-session:

    you[0]> what is the capital of France?
    bot> Paris.
    you[0]> /user 1
    [switched to user 1]
    you[1]> write me a haiku about cache coherence

The KV caches are *paged*: every layer reads its cache through a pool of
fixed-size blocks plus a per-user page table (``tt/paged_cache.py``), so all users
share one captured decode trace -- switching users rewrites a page table rather
than the cache -- and blocks are handed out on demand, letting the users share one
token budget (``--total-context``) instead of reserving a full context each. A
sliding-window layer needs only ``sliding_window`` tokens of blocks per user
however long the conversation runs.

Startup runs one throwaway decode step (and resets) so the kernel compile and the
trace capture -- minutes of it -- are done before the first prompt appears, which
also keeps the reported per-turn timings honest.

The chat template comes from ``encoding_dsv4.render_message``: ``--system-prompt``
(or ``--system-prompt-file``, or ``/system`` mid-session) puts a system message at
absolute position 0, and ``--think`` switches the template to thinking mode, where
the model opens its reply with a ``<think>`` reasoning block (streamed inline).

Commands at the prompt: ``/user``, ``/users``, ``/reset``, ``/system``, ``/think``,
``/context``, ``/help``, ``/exit``.

Run it (ttnn venv, from the repo root)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v4_flash/demo/chat_cli.py --num-users 4 \\
        --max-context 2048 --system-prompt "You are a terse assistant." --think
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.paged_cache import PagedCacheFull, round_context
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# The decode demo owns the reference RoPE-table construction and the lazy
# dequantizing weight thunk; reuse them so the CLI cannot drift from it.
from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import (
    _DEFAULT_MODEL_DIR,
    _WEIGHT_DTYPE,
    _build_rope,
    _w,
)

_PAGE_BLOCK_SIZE = 32


class ContextFull(RuntimeError):
    """A user ran past its own context, or the shared block pool ran dry."""


@contextlib.contextmanager
def open_mesh_device(trace_region_size: int | None):
    """Open the full system mesh the way the ``mesh_device`` pytest fixture does for
    the decode demo: 2D fabric (the submesh pipeline sockets need it) and two
    command queues."""
    from tests.scripts.common import get_updated_device_params

    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2}
    if trace_region_size:
        device_params["trace_region_size"] = trace_region_size
    params = get_updated_device_params(device_params)
    fabric_config = params.pop("fabric_config")
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    # The fixture defaults to the whole system flattened to a line, which is the
    # shape the model's submesh pipeline is built against.
    mesh_shape = ttnn.MeshShape(1, ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size())
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **params)
    logger.info(f"opened mesh device with {mesh_device.get_num_devices()} devices")
    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


class _PrefillProgress:
    """One-line "prefilling" status for the stretch before any reply token exists.

    A prompt costs one decode step per token, so a long turn (or the first turn of a
    long system prompt) sits silent for seconds. On a terminal the line is rewritten
    in place and erased once the reply starts; when stdout is redirected there is
    nothing to rewrite, so it prints a single note instead.
    """

    _INTERVAL = 0.1  # s between redraws; a step is ~10ms, so don't repaint per token

    def __init__(self, t0: float):
        self.t0 = t0
        self.tty = sys.stdout.isatty()
        self.last = 0.0
        self.drawn = False

    def __call__(self, done: int, total: int) -> None:
        now = time.perf_counter()
        if not self.tty:
            if not self.drawn:
                print(f"[prefilling {total} tokens...]", flush=True)
                self.drawn = True
            return
        if done < total and now - self.last < self._INTERVAL:
            return
        self.last = now
        elapsed = now - self.t0
        rate = f"{done / elapsed:.1f} tok/s" if elapsed else "-- tok/s"
        sys.stdout.write(f"\r\033[K[prefilling {done}/{total} tokens, {rate}]")
        sys.stdout.flush()
        self.drawn = True

    def done(self) -> None:
        """Erase the status line (the reply is printed where it was)."""
        if self.drawn and self.tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self.drawn = False


_ANSI = {"gray": "\033[90m", "blue": "\033[94m", "reset": "\033[0m"}
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ReplyStream:
    """Detokenize a reply incrementally and print it as it arrives.

    Two things make this more than a ``print``:

    * **Partial characters.** A token can carry half a UTF-8 sequence (or half an
      emoji's surrogate pair), so decoding the ids so far can end in U+FFFD that the
      *next* token completes. Writing that would leave a replacement character on
      screen forever, so a trailing one is held back until it resolves. If a re-decode
      ever revises text already written, the cursor rewinds to the common prefix and
      carries on from there rather than dropping the rest of the reply.
    * **Thinking blocks.** ``<think>`` / ``</think>`` are kept in the output rather
      than skipped: the tags are printed in blue, the reasoning between them in gray,
      and a newline follows the closing tag so the answer starts on its own line. A tag
      split across two tokens is held back until it is whole.
    """

    def __init__(self, tokenizer, thinking: bool):
        self.tokenizer = tokenizer
        self.color = sys.stdout.isatty()
        self.in_think = thinking
        self.text = ""  # stable decoded reply, tags included, no escapes
        self.cursor = 0  # how much of it has been written out
        self.painted: str | None = None  # colour the terminal is currently set to
        if thinking:
            # In thinking mode the template opens the block itself, so the model's
            # output holds only the closing tag; show the state it is already in.
            self._write(_THINK_OPEN, "blue")

    # -- output ---------------------------------------------------------------- #
    def _write(self, text: str, color: str | None = None) -> None:
        """Write ``text``, switching the terminal colour only when it changes -- the
        reply arrives a few characters at a time and wrapping every one of them in its
        own escape pair would bloat anything piped or copied out of the terminal."""
        if self.color and color != self.painted:
            sys.stdout.write(_ANSI["reset"] if color is None else _ANSI[color])
            self.painted = color
        sys.stdout.write(text)
        sys.stdout.flush()

    def _body(self, text: str) -> None:
        self._write(text, "gray" if self.in_think else None)

    # -- incremental decode ---------------------------------------------------- #
    def push(self, token_ids) -> None:
        """Take the reply's ids so far and write whatever new text they resolve to."""
        full = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        full = full.rstrip("\ufffd")  # an unfinished character: wait for the next token
        if not full.startswith(self.text):
            self.cursor = min(self.cursor, len(os.path.commonprefix([full, self.text])))
        self.text = full
        self._flush()

    def close(self) -> None:
        """Write the tail held back for a possible partial tag, drop any colour and end
        the line (a reply cut short by the token cap can end inside a think block)."""
        self._flush(final=True)
        self._write("\n", None)

    def _next_tag(self, start: int):
        """The first whole think tag at or after ``start``, as ``(index, tag)``."""
        hits = [(self.text.find(t, start), t) for t in (_THINK_OPEN, _THINK_CLOSE)]
        hits = [(i, t) for i, t in hits if i != -1]
        return min(hits) if hits else (None, None)

    def _held_back(self) -> int:
        """Characters at the end that could still turn into a tag once more arrive."""
        for tag in (_THINK_CLOSE, _THINK_OPEN):
            for n in range(len(tag) - 1, 0, -1):
                if self.text.endswith(tag[:n]):
                    return n
        return 0

    def _flush(self, final: bool = False) -> None:
        while self.cursor < len(self.text):
            index, tag = self._next_tag(self.cursor)
            if index is None:
                end = len(self.text) - (0 if final else self._held_back())
                if end > self.cursor:
                    self._body(self.text[self.cursor : end])
                    self.cursor = end
                return
            if index > self.cursor:
                self._body(self.text[self.cursor : index])
            self._write(tag, "blue")
            self.cursor = index + len(tag)
            self.in_think = tag == _THINK_OPEN
            if tag == _THINK_CLOSE:
                self._write("\n", None)


class ChatEngine:
    """The resident model, plus the users that share it.

    One :class:`DeepSeekV4Model` (weights, RoPE tables, decode traces) serves every
    user; each :class:`UserSession` owns a conversation and, on the traced path, a
    paged KV session on the model. Switching users is a page-table rewrite plus a
    swap of the small compressor window buffers, so it costs no cache copying and
    needs no second trace capture.
    """

    def __init__(self, mesh_device, args):
        from transformers import AutoTokenizer
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        loader = DeepseekV4WeightLoader(args.model_dir)
        config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
        config._attn_implementation = "eager"
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
        self.max_new_tokens = args.max_new_tokens
        self.traced = args.traced

        eos = config.eos_token_id
        self.eos_id = eos[0] if isinstance(eos, (list, tuple)) else eos

        # The traced caches are fixed-size and the RoPE tables are precomputed, so the
        # per-user context budget is chosen up front. Round it up so every
        # compressor's entry count tiles cleanly into whole blocks.
        crs = {int(v) for v in config.compress_rates.values()}
        self.max_seq = round_context(args.max_context, crs, _PAGE_BLOCK_SIZE)
        rope = _build_rope(config, self.max_seq)
        self.rope = rope

        max_layers = min(args.num_layers or config.num_hidden_layers, config.num_hidden_layers)
        cache_dir = args.cache_dir
        top_cache = WeightCache(os.path.join(cache_dir, "full_decode", "ttnn")) if cache_dir else None

        self.model = DeepSeekV4Model(
            config,
            loader,
            mesh_device,
            cache=top_cache,
            weight_dtype=_WEIGHT_DTYPE,
            max_layers=max_layers,
            use_submeshes=True,
        )
        self.lm_head = Linear(
            _w(loader, "lm_head.weight"),
            self.model.last_device,
            top_cache.file("lm_head") if top_cache else None,
            dtype=_WEIGHT_DTYPE,
        )
        logger.info(
            f"built DeepSeekV4Model with {self.model.num_layers}/{config.num_hidden_layers} layers, "
            f"context {self.max_seq} tokens/user"
        )

        if self.traced:
            # Traces are captured lazily on the first decode step and address these
            # buffers in place, so this happens exactly once per process. Every user's
            # paged session is allocated here too: allocating on a device that already
            # holds a trace is unsafe.
            self.model.prepare_static_decode(
                rope,
                self.max_seq,
                lm_head=self.lm_head,
                num_sessions=args.num_users,
                total_tokens=round_context(args.total_context, crs, _PAGE_BLOCK_SIZE),
                block_size=_PAGE_BLOCK_SIZE,
            )
        else:
            self.model.reset_caches(self.max_seq)

        self.users = [UserSession(self, i, args) for i in range(args.num_users)]
        self.active = 0
        self.users[0].activate()

    # -- users ----------------------------------------------------------------- #
    @property
    def user(self) -> "UserSession":
        return self.users[self.active]

    def switch_to(self, index: int) -> "UserSession":
        """Make user ``index`` the one the prompt talks to."""
        if not 0 <= index < len(self.users):
            raise KeyError(f"user {index} does not exist (0-{len(self.users) - 1})")
        self.active = index
        user = self.users[index]
        user.activate()
        return user

    def tokens_left(self) -> int:
        """Tokens the shared block pool can still admit across all users."""
        return self.model.session_tokens_left() if self.traced and self.model.paged else self.max_seq

    # -- decode ---------------------------------------------------------------- #
    def step(self, user: "UserSession", token_id: int, pos: int) -> int:
        """Feed ``token_id`` at absolute position ``pos`` of ``user``'s conversation;
        return the argmax of the resulting single-token logits (the device sync happens
        when the logits are read back)."""
        user.activate()
        if self.traced:
            # [1, 1, vocab], lm_head in-trace and read back off the D2H socket
            logits_host = self.model.decode_traced(token_id, pos)
        else:
            logits_host = ttnn.to_torch(self.lm_head(self.model.decode(token_id, pos, self.rope)))
        logits = logits_host.reshape(1, -1).float()
        return int(logits[0].argmax().item())

    def warmup(self) -> None:
        """Push one throwaway token through so the JIT compile and the trace capture
        are paid for before the user gets a prompt, then rewind.

        ``decode_traced`` captures *every* variant on its first call, so a single step
        is enough; the token it wrote into user 0's cache is dropped by the reset."""
        print("[warming up: compiling kernels and capturing decode traces...]", flush=True)
        t0 = time.perf_counter()
        self.step(self.users[0], self.tokenizer.bos_token_id or 0, 0)  # value irrelevant, state discarded
        self.users[0].reset()
        print(f"[warmup done in {time.perf_counter() - t0:.1f}s]", flush=True)


class UserSession:
    """One user's conversation against the shared :class:`ChatEngine`.

    ``pos`` is the number of tokens already written into this user's caches, i.e. the
    absolute position the next token is fed at. ``pending_id`` is the last token the
    model produced but that has *not* been fed back yet (its logits were the ones we
    sampled from), so the next feed always starts with it. ``_next_render`` is the
    first ``messages`` entry not yet encoded into the caches, so each turn only
    renders (and prefills) what is genuinely new.
    """

    def __init__(self, engine: ChatEngine, index: int, args):
        self.engine = engine
        self.index = index
        self.thinking_mode = "thinking" if args.think else "chat"
        self.reasoning_effort = args.reasoning_effort
        self.system_prompt: str | None = args.system_prompt or None
        # On the traced path each user is a paged session on the model; the eager path
        # has a single dense cache, so it only supports one user (enforced in
        # :func:`parse_args`).
        self.sid = engine.model.open_session() if engine.traced else None
        self.messages: list[dict] = []
        self.pos = 0
        self.pending_id: int | None = None
        self._next_render = 0
        self._seed_messages()

    # -- state ----------------------------------------------------------------- #
    def activate(self) -> None:
        """Point the decode traces at this user's KV blocks and window state."""
        if self.sid is not None:
            self.engine.model.activate_session(self.sid)

    def _seed_messages(self) -> None:
        """A system prompt is a plain message at index 0 (``encoding_dsv4`` renders it
        as bare text ahead of the first ``<|User|>`` token), so it is queued for the
        next turn's prefill rather than fed on its own."""
        self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        self._next_render = 0

    def reset(self) -> None:
        """Drop the conversation and rewind this user's caches to an empty sequence,
        keeping the system prompt (which is re-prefilled with the next turn) and
        returning its compressed blocks to the shared pool."""
        if self.sid is not None:
            self.engine.model.reset_session(self.sid)
        elif self.engine.traced:
            self.engine.model.reset_static_caches()
        else:
            self.engine.model.reset_caches(self.engine.max_seq)
        self.pos = 0
        self.pending_id = None
        self._seed_messages()

    def set_system_prompt(self, text: str | None) -> None:
        """Swap this user's system prompt. It lives at absolute position 0, so the
        caches have to be rewound; the conversation is dropped with it."""
        self.system_prompt = text or None
        self.reset()

    # -- decode ---------------------------------------------------------------- #
    def _feed(self, ids: list[int], progress=None) -> int:
        """Replay decode over ``ids`` at ascending positions (this is the prefill);
        return the token predicted after the last one. ``progress(done, total)`` is
        called after every token, since a long prompt spends a step per token with
        nothing to show for it yet."""
        engine = self.engine
        if self.pos + len(ids) >= engine.max_seq:
            raise ContextFull(f"user {self.index} needs {self.pos + len(ids)} of {engine.max_seq} tokens")
        if len(ids) > engine.tokens_left():
            raise ContextFull(
                f"the shared cache pool has room for about {engine.tokens_left()} more tokens, "
                f"this turn needs {len(ids)}"
            )
        next_id = engine.eos_id
        for done, token_id in enumerate(ids, start=1):
            next_id = engine.step(self, token_id, self.pos)
            self.pos += 1
            if progress is not None:
                progress(done, len(ids))
        return next_id

    def _turn_prompt_ids(self, text: str) -> list[int]:
        """Tokens to append to the cache for a new user turn: the tail of the
        previous assistant turn (its last generated token, plus the EOS that closes
        it if generation was cut short) followed by every message not yet encoded --
        the new user message, preceded by the system prompt on the first turn."""
        first_turn = self._next_render == 0
        self.messages.append({"role": "user", "content": text})
        # ``render_message`` renders one message against the whole conversation and
        # appends the assistant/thinking transition tokens, so concatenating the
        # not-yet-encoded messages gives exactly this turn's incremental prompt text.
        # Past assistant turns are skipped: the model's own tokens are in the cache
        # already (see the EOS handling below).
        rendered = "".join(
            render_message(i, self.messages, self.thinking_mode, reasoning_effort=self.reasoning_effort)
            for i in range(self._next_render, len(self.messages))
            if self.messages[i]["role"] != "assistant"
        )
        # Only the very first turn may carry the tokenizer's special prefix (BOS);
        # later turns continue an existing sequence.
        ids = list(self.engine.tokenizer(rendered, add_special_tokens=first_turn)["input_ids"])

        prefix: list[int] = []
        if self.pending_id is not None:
            prefix.append(self.pending_id)
            if self.pending_id != self.engine.eos_id:
                prefix.append(self.engine.eos_id)  # close an assistant turn that hit the token cap
        return prefix + ids

    def generate(self, text: str) -> None:
        """Prefill one user turn (showing progress) and stream the assistant reply."""
        engine = self.engine
        tokenizer = engine.tokenizer
        prompt_ids = self._turn_prompt_ids(text)
        t0 = time.perf_counter()
        progress = _PrefillProgress(t0)
        try:
            next_id = self._feed(prompt_ids, progress)
        except (ContextFull, PagedCacheFull) as e:
            self.messages.pop()  # nothing was fed: leave the conversation exactly as it was
            raise ContextFull(str(e)) from e
        finally:
            progress.done()
        self._next_render = len(self.messages)
        prefill_time = time.perf_counter() - t0

        print("bot> ", end="", flush=True)
        generated: list[int] = []
        stream = _ReplyStream(tokenizer, self.thinking_mode == "thinking")
        decode_time = 0.0
        try:
            for _ in range(engine.max_new_tokens):
                if next_id == engine.eos_id:
                    break
                if self.pos >= engine.max_seq - 1:
                    print("\n[context full -- use /reset]", flush=True)
                    break
                generated.append(next_id)
                stream.push(generated)
                t1 = time.perf_counter()
                next_id = engine.step(self, next_id, self.pos)
                self.pos += 1
                decode_time += time.perf_counter() - t1
            else:
                logger.info(f"stopped at the {engine.max_new_tokens}-token cap")
        except KeyboardInterrupt:
            # The step that was running finished, so the caches are consistent: keep
            # the partial reply as this turn's assistant message and carry on.
            print("\n[interrupted]", flush=True)
        except PagedCacheFull as e:
            # Another user holds the blocks this reply would need. The turn so far is
            # valid, so keep it and let the user free space with /reset.
            print(f"\n[cache pool full: {e} -- /reset a user]", flush=True)

        stream.close()
        # ``next_id`` was produced but never fed; the next turn starts with it.
        self.pending_id = next_id
        self.messages.append({"role": "assistant", "content": stream.text})
        self._next_render = len(self.messages)
        rate = f"{len(generated) / decode_time:.2f} tok/s" if decode_time else "n/a"
        logger.info(
            f"user {self.index}: prefill {len(prompt_ids)} tokens in {prefill_time:.2f}s | "
            f"decode {len(generated)} tokens at {rate} | context {self.pos}/{engine.max_seq}"
        )


_HELP = """commands:
  /user N         switch to user N (each user is an independent conversation with its
                  own KV cache; bare /user shows who you are talking to)
  /users          list the users, their token usage and the shared pool's usage
  /reset [N|all]  clear a conversation and rewind its KV cache (default: current user)
  /system TEXT    replace the current user's system prompt (implies a reset of that
                  user -- it sits at position 0); bare /system clears it
  /think [on|off] thinking mode for the current user's following turns (no argument
                  flips it); this only changes the template, so it needs no reset
  /context        tokens used of the current user's budget, and the shared pool
  /help           this message
  /exit           quit (also ctrl-D)
anything else is sent to the model as a user turn."""


def _print_users(engine: ChatEngine) -> None:
    for user in engine.users:
        marker = "*" if user.index == engine.active else " "
        think = " think" if user.thinking_mode == "thinking" else ""
        print(
            f" {marker} user {user.index}: {user.pos}/{engine.max_seq} tokens, " f"{len(user.messages)} messages{think}"
        )
    if engine.traced and engine.model.paged:
        usage = ", ".join(
            f"{name} {used}/{total} blocks" for name, (used, total) in engine.model.session_usage().items()
        )
        print(f"   shared pool: {usage}; room for ~{engine.tokens_left()} more tokens")


def repl(engine: ChatEngine) -> None:
    print(
        f"\nDeepSeek-V4-Flash chat ({engine.model.num_layers} layers, {len(engine.users)} users). "
        "/help for commands.\n"
    )
    if engine.user.thinking_mode == "thinking":
        print("[thinking mode: the reasoning block is streamed inline in gray before the reply]\n")
    while True:
        try:
            line = input(f"you[{engine.active}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith("/"):
            head, _, rest = line.partition(" ")
            cmd, rest = head.lower(), rest.strip()
            if cmd in ("/exit", "/quit"):
                return
            if cmd == "/user":
                if not rest:
                    print(f"[talking to user {engine.active} of 0-{len(engine.users) - 1}]")
                    continue
                try:
                    engine.switch_to(int(rest))
                except (ValueError, KeyError) as e:
                    print(f"[{e}]" if isinstance(e, KeyError) else "usage: /user N")
                    continue
                print(f"[switched to user {engine.active}]")
            elif cmd == "/users":
                _print_users(engine)
            elif cmd == "/reset":
                targets = engine.users if rest == "all" else None
                if targets is None and rest:
                    try:
                        targets = [engine.users[int(rest)]]
                    except (ValueError, IndexError):
                        print(f"usage: /reset [0-{len(engine.users) - 1}|all]")
                        continue
                for user in targets or [engine.user]:
                    user.reset()
                engine.user.activate()  # a reset of another user left it active
                names = "all users" if rest == "all" else f"user {(targets or [engine.user])[0].index}"
                print(f"[conversation reset: {names}]")
            elif cmd == "/system":
                engine.user.set_system_prompt(rest)
                print(f"[user {engine.active} system prompt: {engine.user.system_prompt!r}; conversation reset]")
            elif cmd == "/think":
                if rest not in ("on", "off", ""):
                    print("usage: /think [on|off]")
                    continue
                on = rest == "on" if rest else engine.user.thinking_mode != "thinking"
                engine.user.thinking_mode = "thinking" if on else "chat"
                print(f"[thinking {'on' if on else 'off'} for user {engine.active}]")
            elif cmd == "/context":
                print(
                    f"[user {engine.active}: {engine.user.pos}/{engine.max_seq} tokens, "
                    f"{len(engine.user.messages)} messages; pool room ~{engine.tokens_left()} tokens]"
                )
            elif cmd == "/help":
                print(_HELP)
            else:
                print(f"unknown command {cmd}; /help for the list")
            continue

        try:
            engine.user.generate(line)  # prints the prefill status, then "bot> " + the reply
        except ContextFull as e:
            print(f"[{e} -- use /reset]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        default=int(os.environ.get("DEEPSEEK_V4_NUM_USERS", "2")),
        help="independent conversations to allocate, switched with /user N (fixed at "
        "startup: their cache blocks cannot be allocated once the traces exist)",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_CONTEXT", "131072")),
        help="tokens (all turns) one user's caches are addressed for; rounded up",
    )
    p.add_argument(
        "--total-context",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_TOTAL_CONTEXT", "0")) or None,
        help="total tokens the shared block pool holds across all users "
        "(default: one --max-context, i.e. the users share a single context's worth)",
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
        help="system prompt prefixed to every user's conversation (see also /system)",
    )
    p.add_argument(
        "--system-prompt-file",
        help="read the system prompt from this file instead of --system-prompt",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help="thinking mode: the reply is preceded by a <think> reasoning block, which is streamed inline",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default=None,
        help="reasoning-effort hint, only meaningful with --think",
    )
    p.add_argument("--trace-region-size", type=int, default=int(os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE", "0")))
    p.add_argument("--no-trace", dest="traced", action="store_false", help="eager decode instead of traced decode")
    p.add_argument("--quiet", action="store_true", help="only warnings and above from the model logs")
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
        args.total_context = args.max_context
    return args


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The model emits plenty of non-ASCII (CJK, emoji, typographic punctuation), which
    # an ASCII/POSIX locale would turn into a UnicodeEncodeError mid-reply.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if not Path(args.model_dir).expanduser().exists():
        print(f"checkpoint not found: {args.model_dir}", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    with open_mesh_device(args.trace_region_size) as mesh_device:
        engine = ChatEngine(mesh_device, args)
        engine.warmup()
        repl(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())

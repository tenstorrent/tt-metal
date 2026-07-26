# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Interactive single-user chat CLI on the full ttnn ``DeepSeekV4Model``.

Same decode engine as
``models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py``
(whose helpers this reuses), turned into a multi-turn REPL: the model, the RoPE
tables and the traced-decode buffers are built once, then every turn only the
*new* tokens are fed into the already-populated caches. There is no prefill op --
a turn's prompt tokens are replayed one decode step each at ascending absolute
positions -- so a follow-up question costs one step per new prompt token and
never recomputes the earlier turns.

Because the caches (and the captured traces that address them) are fixed-size,
the whole conversation must fit ``--max-context`` tokens; ``/reset`` zeroes the
caches in place and restarts at position 0 without re-capturing any trace.

The chat template comes from ``encoding_dsv4.render_message``: ``--system-prompt``
(or ``--system-prompt-file``, or ``/system`` mid-session) puts a system message at
absolute position 0, and ``--think`` switches the template to thinking mode, where
the model opens its reply with a ``<think>`` reasoning block (streamed inline).

Commands at the prompt: ``/reset``, ``/system``, ``/think``, ``/context``, ``/help``,
``/exit``.

Run it (ttnn venv, from the repo root)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v4_flash/demo/chat_cli.py --max-context 2048 \\
        --system-prompt "You are a terse assistant." --think
"""

from __future__ import annotations

import argparse
import contextlib
import math
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
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# The decode demo owns the reference RoPE-table construction and the lazy
# dequantizing weight thunk; reuse them so the CLI cannot drift from it.
from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import (
    _DEFAULT_MODEL_DIR,
    _WEIGHT_DTYPE,
    _build_rope,
    _pad_to_tile,
    _w,
)


class ContextFull(RuntimeError):
    """The conversation ran past the fixed cache / RoPE capacity."""


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


class ChatSession:
    """One user's conversation against a single resident ``DeepSeekV4Model``.

    ``pos`` is the number of tokens already written into the caches, i.e. the
    absolute position the next token is fed at. ``pending_id`` is the last token
    the model produced but that has *not* been fed back yet (its logits were the
    ones we sampled from), so the next feed always starts with it.
    ``_next_render`` is the first ``messages`` entry not yet encoded into the
    caches, so each turn only renders (and prefills) what is genuinely new.
    """

    def __init__(self, mesh_device, args):
        from transformers import AutoTokenizer
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        loader = DeepseekV4WeightLoader(args.model_dir)
        config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
        config._attn_implementation = "eager"
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
        self.thinking_mode = "thinking" if args.think else "chat"
        self.reasoning_effort = args.reasoning_effort
        self.max_new_tokens = args.max_new_tokens
        self.traced = args.traced

        eos = config.eos_token_id
        self.eos_id = eos[0] if isinstance(eos, (list, tuple)) else eos

        # The traced caches are fixed-size and the RoPE tables are precomputed, so
        # the context budget is chosen up front. Round it up to a multiple of every
        # compress-rate (and of the tile) so each compressor's capacity tiles
        # cleanly into windows.
        crs = {int(v) for v in config.compress_rates.values()}
        step = math.lcm(32, *crs) if crs else 32
        self.max_seq = ((_pad_to_tile(args.max_context) + step - 1) // step) * step
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
            f"context {self.max_seq} tokens"
        )

        if self.traced:
            # Traces are captured lazily on the first decode step and address these
            # buffers in place, so this happens exactly once per process.
            self.model.prepare_static_decode(rope, self.max_seq, lm_head=self.lm_head)
        else:
            self.model.reset_caches(self.max_seq)

        self.system_prompt: str | None = args.system_prompt or None
        self.messages: list[dict] = []
        self.pos = 0
        self.pending_id: int | None = None
        self._next_render = 0
        self._seed_messages()

    # -- state ----------------------------------------------------------------- #
    def _seed_messages(self) -> None:
        """A system prompt is a plain message at index 0 (``encoding_dsv4`` renders it
        as bare text ahead of the first ``<|User|>`` token), so it is queued for the
        next turn's prefill rather than fed on its own."""
        self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        self._next_render = 0

    def reset(self) -> None:
        """Drop the conversation and rewind the caches to an empty sequence, keeping
        the system prompt (which is re-prefilled with the next turn)."""
        if self.traced:
            self.model.reset_static_caches()
        else:
            self.model.reset_caches(self.max_seq)
        self.pos = 0
        self.pending_id = None
        self._seed_messages()

    def set_system_prompt(self, text: str | None) -> None:
        """Swap the system prompt. It lives at absolute position 0, so the caches have
        to be rewound; the conversation is dropped with it."""
        self.system_prompt = text or None
        self.reset()

    # -- decode ---------------------------------------------------------------- #
    def _step(self, token_id: int, pos: int) -> int:
        """Feed ``token_id`` at absolute position ``pos``; return the argmax of the
        resulting single-token logits (the device sync happens in ``to_torch``)."""
        if self.traced:
            logits_tt = self.model.decode_traced(token_id, pos)  # [1, 1, vocab], lm_head in-trace
        else:
            logits_tt = self.lm_head(self.model.decode(token_id, pos, self.rope))
        logits = ttnn.to_torch(logits_tt).reshape(1, -1).float()
        return int(logits[0].argmax().item())

    def _feed(self, ids: list[int]) -> int:
        """Replay decode over ``ids`` at ascending positions (this is the prefill);
        return the token predicted after the last one."""
        if self.pos + len(ids) >= self.max_seq:
            raise ContextFull(f"conversation needs {self.pos + len(ids)} of {self.max_seq} tokens")
        next_id = self.eos_id
        for token_id in ids:
            next_id = self._step(token_id, self.pos)
            self.pos += 1
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
        ids = list(self.tokenizer(rendered, add_special_tokens=first_turn)["input_ids"])

        prefix: list[int] = []
        if self.pending_id is not None:
            prefix.append(self.pending_id)
            if self.pending_id != self.eos_id:
                prefix.append(self.eos_id)  # close an assistant turn that hit the token cap
        return prefix + ids

    def generate(self, text: str) -> None:
        """Prefill one user turn and stream the assistant reply to stdout."""
        prompt_ids = self._turn_prompt_ids(text)
        t0 = time.perf_counter()
        try:
            next_id = self._feed(prompt_ids)
        except ContextFull:
            self.messages.pop()  # nothing was fed: leave the session exactly as it was
            raise
        self._next_render = len(self.messages)
        prefill_time = time.perf_counter() - t0

        generated: list[int] = []
        shown = ""
        decode_time = 0.0
        try:
            for _ in range(self.max_new_tokens):
                if next_id == self.eos_id:
                    break
                if self.pos >= self.max_seq - 1:
                    print("\n[context full -- use /reset]", flush=True)
                    break
                generated.append(next_id)
                # Detokenize the reply as a whole each step and print only the new
                # text: a single token can be half a multi-byte character or word
                # piece.
                full = self.tokenizer.decode(generated, skip_special_tokens=True)
                if full.startswith(shown):
                    sys.stdout.write(full[len(shown) :])
                    sys.stdout.flush()
                    shown = full
                t1 = time.perf_counter()
                next_id = self._step(next_id, self.pos)
                self.pos += 1
                decode_time += time.perf_counter() - t1
            else:
                logger.info(f"stopped at the {self.max_new_tokens}-token cap")
        except KeyboardInterrupt:
            # The step that was running finished, so the caches are consistent: keep
            # the partial reply as this turn's assistant message and carry on.
            print("\n[interrupted]", flush=True)

        # ``next_id`` was produced but never fed; the next turn starts with it.
        self.pending_id = next_id
        self.messages.append({"role": "assistant", "content": shown})
        self._next_render = len(self.messages)
        print(flush=True)
        rate = f"{len(generated) / decode_time:.2f} tok/s" if decode_time else "n/a"
        logger.info(
            f"prefill {len(prompt_ids)} tokens in {prefill_time:.2f}s | "
            f"decode {len(generated)} tokens at {rate} | context {self.pos}/{self.max_seq}"
        )


_HELP = """commands:
  /reset          clear the conversation and rewind the KV caches to an empty sequence
  /system TEXT    replace the system prompt (implies /reset -- it sits at position 0);
                  bare /system clears it
  /think [on|off] thinking mode for the following turns (no argument flips it); this
                  only changes the template, so it needs no reset
  /context        tokens used of the fixed context budget
  /help           this message
  /exit           quit (also ctrl-D)
anything else is sent to the model as a user turn."""


def repl(session: ChatSession) -> None:
    print(f"\nDeepSeek-V4-Flash chat ({session.model.num_layers} layers). /help for commands.\n")
    if session.thinking_mode == "thinking":
        print("[thinking mode: the reasoning block is streamed inline before the reply]\n")
    while True:
        try:
            line = input("you> ").strip()
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
            if cmd == "/reset":
                session.reset()
                print("[conversation reset]")
            elif cmd == "/system":
                session.set_system_prompt(rest)
                print(f"[system prompt: {session.system_prompt!r}; conversation reset]")
            elif cmd == "/think":
                if rest not in ("on", "off", ""):
                    print("usage: /think [on|off]")
                    continue
                on = rest == "on" if rest else session.thinking_mode != "thinking"
                session.thinking_mode = "thinking" if on else "chat"
                print(f"[thinking {'on' if on else 'off'}]")
            elif cmd == "/context":
                print(f"[{session.pos}/{session.max_seq} tokens, {len(session.messages)} messages]")
            elif cmd == "/help":
                print(_HELP)
            else:
                print(f"unknown command {cmd}; /help for the list")
            continue

        print("bot> ", end="", flush=True)
        try:
            session.generate(line)
        except ContextFull as e:
            print(f"\n[{e} -- use /reset]")


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
        "--max-context",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_CONTEXT", "2048")),
        help="total tokens (all turns) the fixed caches are sized for; rounded up",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "512")),
        help="cap on the tokens generated per reply",
    )
    p.add_argument(
        "--system-prompt",
        default=os.environ.get("DEEPSEEK_V4_SYSTEM_PROMPT", ""),
        help="system prompt prefixed to the conversation (see also /system at the prompt)",
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
    return args


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if not Path(args.model_dir).expanduser().exists():
        print(f"checkpoint not found: {args.model_dir}", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    with open_mesh_device(args.trace_region_size) as mesh_device:
        session = ChatSession(mesh_device, args)
        repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())

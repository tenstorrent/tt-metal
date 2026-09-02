# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Text-generation engine over the ttnn :class:`DeepSeekV4Model`.

This is the layer between the raw model (which speaks single traced decode steps
against paged KV sessions) and its callers -- the chat CLI, the demo tests and the
vLLM wrapper in :mod:`.generator_vllm`. It owns the things every caller otherwise
has to rebuild by hand: the checkpoint/tokenizer load, the YaRN RoPE tables, the
context rounding, the traced-decode setup and the per-session decode step.

There is no prefill op: a prompt is replayed one decode step per token at ascending
absolute positions into the (empty) caches, which is what :meth:`prefill` does.

Positions are *not* tracked here. A step is always ``(session, token, absolute
position)`` so the caller (a chat turn, a vLLM slot) stays the owner of its own
sequence bookkeeping::

    gen = DeepSeekV4Generator.from_pretrained(mesh_device, max_seq_len=4096, num_sessions=2)
    sid = gen.open_session()
    next_id = gen.prefill(sid, gen.encode("hello"))
    for token_id in gen.generate(sid, next_id, start_pos=len(prompt_ids), max_new_tokens=64):
        print(gen.tokenizer.decode([token_id]), end="")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message
from models.experimental.deepseek_v4_flash.tt.common import _region
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.system_config import SystemConfig, load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731")


def weight_thunk(loader: DeepseekV4WeightLoader, name: str) -> Callable[[], torch.Tensor]:
    """Lazy (dequantized) fetch -> thunk (a populated tile cache skips the read)."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def build_rope(config, max_seq: int) -> dict:
    """YaRN RoPE tables (cos/sin halves) spanning ``max_seq`` for every layer family.

    ``win[cr]`` holds one windowed table per distinct compress-rate (CSA / HCA
    layers); decode slices the rows it needs from the max-length tables.
    """
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


def round_max_seq(config, max_seq_len: int, block_size: int = 32) -> int:
    """Round a requested context up to a length every compressor's entry count tiles
    into whole blocks (the traced caches and RoPE tables are sized to it)."""
    return round_context(max_seq_len, {int(v) for v in config.compress_rates.values()}, block_size)


class DeepSeekV4Generator:
    """A resident :class:`DeepSeekV4Model` plus the state a caller needs to decode.

    On the traced path (the default) the model owns a block pool per layer and each
    concurrent sequence is a *session*: one captured trace serves them all, and
    switching sessions rewrites page tables rather than caches. The eager path has a
    single dense cache per layer, so it supports one sequence at a time.
    """

    def __init__(self, model, lm_head, tokenizer, config, rope, max_seq: int, traced: bool = True):
        self.model = model
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.config = config
        self.rope = rope
        self.max_seq = max_seq
        self.traced = traced
        eos = config.eos_token_id
        self.eos_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    @property
    def system_config(self) -> SystemConfig:
        """The machine tuning profile the model was built with."""
        return self.model.system_config

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        *,
        model_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        max_seq_len: int | None = None,
        num_sessions: int | None = None,
        total_tokens: int | None = None,
        num_layers: int | None = None,
        block_size: int | None = None,
        weight_dtype=None,
        traced: bool | None = None,
        prepare: bool = True,
        system_config: SystemConfig | None = None,
        system_profile: str | None = None,
    ) -> "DeepSeekV4Generator":
        """Load the checkpoint, build the model across the mesh and allocate the
        decode state.

        Every sizing argument defaults to the machine's system profile (see
        :mod:`.system_config`), picked by ``mesh_device``'s device count unless
        ``system_profile`` names one or ``system_config`` hands one over. So on an
        8-chip host this reads its context and session counts from the ``p150x8``
        profile and on a 32-chip one from ``galaxy32``, with no caller change.

        ``max_seq_len`` is rounded up (see :func:`round_max_seq`); ``total_tokens`` is
        the *shared* budget the ``num_sessions`` sessions draw their blocks from and
        defaults to one full context. ``num_layers`` caps the decoder stack -- the full
        43-layer bf4 stack does not fit a single Blackhole. Everything the sessions
        need is allocated by :meth:`prepare_decode`, which this runs unless
        ``prepare=False`` -- pass that when the caller owns the KV memory and will call
        it itself (the vLLM wrapper allocates the pools in ``allocate_kv_cache``).
        Allocating on a device that already holds a trace is unsafe, so no device
        buffer may appear after that call.
        """
        from transformers import AutoTokenizer
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        sys_cfg = system_config or load_system_config(profile=system_profile, mesh_device=mesh_device).log()
        decode = sys_cfg.decode
        max_seq_len = max_seq_len if max_seq_len is not None else decode.max_context
        num_sessions = num_sessions if num_sessions is not None else decode.num_users
        block_size = block_size if block_size is not None else decode.block_size
        weight_dtype = weight_dtype if weight_dtype is not None else decode.ttnn_weight_dtype
        traced = traced if traced is not None else decode.traced

        model_dir = str(model_dir or os.environ.get("DEEPSEEK_V4_HF_MODEL", DEFAULT_MODEL_DIR))
        cache_dir = cache_dir if cache_dir is not None else os.environ.get("DEEPSEEK_V4_CACHE_DIR")

        loader = DeepseekV4WeightLoader(model_dir)
        config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
        config._attn_implementation = "eager"
        tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

        max_seq = round_max_seq(config, max_seq_len, block_size)
        rope = build_rope(config, max_seq)
        max_layers = num_layers or decode.resolve_num_layers(config.num_hidden_layers)
        max_layers = min(max_layers, config.num_hidden_layers)
        weight_cache = WeightCache(os.path.join(str(cache_dir), os.path.basename(model_dir))) if cache_dir else None

        model = DeepSeekV4Model(
            config,
            loader,
            mesh_device,
            cache=weight_cache,
            weight_dtype=weight_dtype,
            max_layers=max_layers,
            use_submeshes=True,
            system_config=sys_cfg,
        )
        lm_head = Linear(
            weight_thunk(loader, "lm_head.weight"),
            model.last_device,
            weight_cache.file("lm_head") if weight_cache else None,
            dtype=weight_dtype,
        )
        logger.info(
            f"built DeepSeekV4Model with {model.num_layers}/{config.num_hidden_layers} layers, "
            f"context {max_seq} tokens, {num_sessions} session(s)"
        )

        generator = cls(model, lm_head, tokenizer, config, rope, max_seq, traced)
        if prepare:
            generator.prepare_decode(num_sessions=num_sessions, total_tokens=total_tokens, block_size=block_size)
        return generator

    def prepare_decode(
        self,
        num_sessions: int | None = None,
        total_tokens: int | None = None,
        block_size: int | None = None,
        tokens_per_block: int | None = None,
        pools: dict[int, ttnn.Tensor] | None = None,
        batch: int | None = None,
    ) -> None:
        """Allocate the decode state: block pools, page tables and per-session buffers.

        Must run before the first step, since the traces are captured then and no
        device buffer may be allocated once one exists. ``pools`` hands in
        caller-owned block pools keyed by layer index (whose geometry
        ``tokens_per_block`` describes); their block count then replaces the internal
        pool plan.

        The unset arguments come from the system profile (``decode.num_users`` /
        ``block_size`` / ``total_context`` / ``batch``), so a machine's session and
        batch sizing lives in one file rather than at each call site.
        """
        decode = self.system_config.decode
        num_sessions = num_sessions if num_sessions is not None else decode.num_users
        block_size = block_size if block_size is not None else decode.block_size
        batch = batch if batch is not None else decode.batch
        if total_tokens is None:
            total_tokens = decode.resolve_total_context()
        if not self.traced:
            if num_sessions > 1:
                raise ValueError("the eager path decodes against one dense cache; use traced=True for several sessions")
            self.model.reset_caches(self.max_seq)
            return
        # ``lm_head`` is folded into the last submesh's trace, so a step returns logits
        # without a separately dispatched matmul.
        self.model.prepare_static_decode(
            self.rope,
            self.max_seq,
            lm_head=self.lm_head,
            num_sessions=num_sessions,
            total_tokens=round_max_seq(self.config, total_tokens or self.max_seq, block_size),
            block_size=block_size,
            tokens_per_block=tokens_per_block,
            pools=pools,
            batch=batch,
        )

    # -- properties ------------------------------------------------------------- #
    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    @property
    def paged(self) -> bool:
        return self.traced and self.model.paged

    # -- sessions --------------------------------------------------------------- #
    def open_session(self) -> int | None:
        """Claim a session slot, or ``None`` on the eager (single-sequence) path."""
        return self.model.open_session() if self.paged else None

    def close_session(self, sid: int | None) -> None:
        if sid is not None:
            self.model.close_session(sid)

    def activate_session(self, sid: int | None) -> None:
        """Point the decode traces at this session's KV blocks and window state."""
        if sid is not None:
            self.model.activate_session(sid)

    def reset_session(self, sid: int | None) -> None:
        """Rewind a session to position 0, returning its compressed blocks to the
        shared pool. A recycled slot must be reset before it is prefilled again."""
        if sid is not None:
            self.model.reset_session(sid)
        elif self.traced:
            self.model.reset_static_caches()
        else:
            self.model.reset_caches(self.max_seq)

    def tokens_left(self) -> int:
        """Tokens the shared block pool can still admit across all open sessions."""
        return self.model.session_tokens_left() if self.paged else self.max_seq

    def usage(self) -> dict:
        """Per-group ``(blocks used, pool size)``, for status reporting."""
        return self.model.session_usage() if self.paged else {}

    # -- tokenizer -------------------------------------------------------------- #
    def encode(self, text: str, thinking_mode: str = "chat", reasoning_effort: str | None = None) -> list[int]:
        """Tokenize one user turn wrapped in the V4 chat template."""
        return self.encode_messages([{"role": "user", "content": text}], thinking_mode, reasoning_effort)

    def encode_messages(
        self, messages: Sequence[dict], thinking_mode: str = "chat", reasoning_effort: str | None = None
    ) -> list[int]:
        rendered = "".join(
            render_message(i, list(messages), thinking_mode, reasoning_effort=reasoning_effort)
            for i in range(len(messages))
        )
        return list(self.tokenizer(rendered)["input_ids"])

    # -- decode ----------------------------------------------------------------- #
    def step(self, sid: int | None, token_id: int, pos: int) -> torch.Tensor:
        """Feed ``token_id`` at absolute position ``pos`` of ``sid``'s sequence and
        return the host logits ``[1, 1, vocab]``.

        Traced steps stream their logits back over the model's D2H socket, so this call
        is where the step synchronizes.
        """
        self.activate_session(sid)
        if self.traced:
            return self.model.decode_traced(int(token_id), int(pos))
        hidden = self.model.decode(int(token_id), int(pos), self.rope)
        with _region("LM_HEAD"):
            return ttnn.to_torch(self.lm_head(hidden))

    def logits(self, sid: int | None, token_id: int, pos: int) -> torch.Tensor:
        """:meth:`step`'s logits as a flat ``[vocab]`` float32 row."""
        return self.step(sid, token_id, pos).reshape(-1).float()

    def step_argmax(self, sid: int | None, token_id: int, pos: int) -> int:
        return int(self.logits(sid, token_id, pos).argmax().item())

    def prefill(
        self,
        sid: int | None,
        token_ids: Sequence[int],
        start_pos: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Replay one decode step per prompt token at ascending absolute positions,
        seeding the caches, and return the host logits ``[vocab]`` after the last one.

        ``progress(done, total)`` is called per token: a prompt costs a step each, so a
        long one is seconds of silence. Continues an existing sequence when
        ``start_pos > 0`` (a follow-up chat turn re-feeds only its new tokens).
        """
        if not token_ids:
            raise ValueError("prefill needs at least one token")
        end = start_pos + len(token_ids)
        if end > self.max_seq:
            raise ValueError(f"sequence needs {end} tokens of the {self.max_seq}-token context")
        out = None
        for offset, token_id in enumerate(token_ids):
            out = self.logits(sid, token_id, start_pos + offset)
            if progress is not None:
                progress(offset + 1, len(token_ids))
        return out

    def generate(
        self,
        sid: int | None,
        first_token_id: int,
        start_pos: int,
        max_new_tokens: int,
        stop_ids: Sequence[int] | None = None,
    ) -> Iterator[int]:
        """Greedily continue a prefilled sequence, yielding one token id per step.

        ``first_token_id`` is the token :meth:`prefill` predicted but has not fed back
        yet, and ``start_pos`` is the position it is fed at. Stops at ``max_new_tokens``,
        at a stop token (EOS by default, not yielded) or at the end of the context; the
        caller resumes from ``start_pos + <tokens yielded>``.
        """
        stops = set(stop_ids) if stop_ids is not None else {self.eos_id}
        token_id, pos = int(first_token_id), int(start_pos)
        for _ in range(max_new_tokens):
            if token_id in stops or pos >= self.max_seq - 1:
                return
            yield token_id
            token_id = self.step_argmax(sid, token_id, pos)
            pos += 1

    def warmup(self, sid: int | None = None) -> None:
        """Push one throwaway token through so the kernel compile and the trace capture
        -- minutes of it -- are paid before the first real step, then rewind.

        ``decode_traced`` captures *every* variant on its first call, so one step is
        enough; the token it wrote is dropped by the reset.
        """
        own = sid is None and self.paged
        if own:
            sid = self.open_session()
        self.logits(sid, self.tokenizer.bos_token_id or 0, 0)  # read back: the capture must finish here
        if own:
            self.close_session(sid)
        else:
            self.reset_session(sid)

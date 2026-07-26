# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-user paged-KV decode demo for ``DeepSeekV4Model`` (traced path).

Two independent conversations share one model, one block pool per layer and one set
of captured decode traces. Each is a *session* on the model: its KV lives in blocks
addressed through a per-session ``page_table``, so switching sessions rewrites that
table (plus the small compressor window buffers) instead of touching the caches, and
no second trace capture is needed.

Flow:
  1. Prefill two different prompts (one token at a time, per session).
  2. Generate in bursts of 64 tokens for user 0, then 64 for user 1, and repeat.

The assertion that matters is the interleaving one: a burst for user 1 must not
change what user 0 goes on to say. So user 0's *first* burst is compared against a
second, single-user run of the same prompt with no interleaving -- if the sessions
shared a block, or the compressor window state leaked between them, the two runs
diverge. (That the paged reads themselves match a dense cache is covered at the op
level by ``test_paged_kv_equivalence.py``.)

Run (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_multi_user_paged_decode_demo.py
"""

from __future__ import annotations

import os
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

_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
_PROMPT_A = "Tell me the name of the top 10 movies of all time. Also list out the top 10 worst movies of all time. Give me details of why you choose those movies. Try to make your response as humours as possible."
_PROMPT_B = "Tell me the name of the top 10 tv shows of all time. Also list out the top 10 worst tv shows of all time. Give me details of why you choose those tv shows. Try to make your response as humours as possible."
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")
_NUM_USERS = 2
_BURST_STEPS = int(os.environ.get("DEEPSEEK_V4_MULTI_USER_BURST", "64"))
_NUM_ROUNDS = int(os.environ.get("DEEPSEEK_V4_MULTI_USER_ROUNDS", "2"))
_PAGE_BLOCK_SIZE = 32


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
    user_id: int
    sid: int
    prompt_ids: list[int]
    generated: list[int] = field(default_factory=list)
    pos: int = 0
    next_token: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


def _tokenize_prompt(tokenizer, text: str) -> list[int]:
    prompt = render_message(0, [{"role": "user", "content": text}], "chat")
    return list(tokenizer(prompt)["input_ids"])


def _decode(model, session: UserSession, token_id: int, pos: int) -> int:
    """One traced step for ``session`` (``lm_head`` is folded into the trace)."""
    model.activate_session(session.sid)
    logits = ttnn.to_torch(model.decode_traced(int(token_id), int(pos))).reshape(1, -1).float()
    return int(logits[0].argmax().item())


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
    """Interleaved two-user decode over shared paged KV pools and one trace set."""
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    prompts = [_PROMPT_A, _PROMPT_B]
    prompt_ids = [_tokenize_prompt(tokenizer, p) for p in prompts]
    per_user = max(len(ids) for ids in prompt_ids) + _BURST_STEPS * _NUM_ROUNDS + 1
    crs = set(config.compress_rates.values())
    max_seq = round_context(per_user, crs, _PAGE_BLOCK_SIZE)

    rope = _build_rope(config, max_seq)
    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    top_cache = WeightCache(os.path.join(_CACHE_DIR, "full_decode", "ttnn")) if _CACHE_DIR else None

    model = DeepSeekV4Model(
        config,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    # One extra session slot for the isolation re-run at the end.
    model.prepare_static_decode(
        rope,
        max_seq,
        lm_head=lm_head,
        num_sessions=_NUM_USERS + 1,
        total_tokens=(_NUM_USERS + 1) * max_seq,
        block_size=_PAGE_BLOCK_SIZE,
    )

    sessions = [UserSession(user_id=u, sid=model.open_session(), prompt_ids=prompt_ids[u]) for u in range(_NUM_USERS)]
    logger.info(
        f"multi-user paged decode: users={_NUM_USERS} block_size={_PAGE_BLOCK_SIZE} max_seq={max_seq} "
        f"pool usage {model.session_usage()}"
    )

    # --- prefill each user's prompt ----------------------------------------- #
    for session in sessions:
        logger.info(f"prefill user {session.user_id}: {prompts[session.user_id]!r} ({session.prompt_len} tokens)")
        for pos in range(session.prompt_len):
            session.next_token = _decode(model, session, session.prompt_ids[pos], pos)
        session.pos = session.prompt_len
        session.generated.append(session.next_token)
        logger.info(
            f"user {session.user_id} prefill done -> first gen token "
            f"{session.next_token} {tokenizer.decode([session.next_token])!r}"
        )

    # --- interleaved generation: 64 steps per user, then switch -------------- #
    for round_idx in range(_NUM_ROUNDS):
        for session in sessions:
            logger.info(f"--- round {round_idx} user {session.user_id}: {_BURST_STEPS} decode steps ---")
            for step in range(_BURST_STEPS):
                pos = session.pos
                assert pos < max_seq, f"user {session.user_id} exceeded max_seq {max_seq}"
                session.next_token = _decode(model, session, session.next_token, pos)
                session.pos += 1
                session.generated.append(session.next_token)
                if step < 3 or step == _BURST_STEPS - 1:
                    logger.info(
                        f"  user {session.user_id} step {step:3d} pos {pos:4d}: "
                        f"id {session.next_token} {tokenizer.decode([session.next_token])!r}"
                    )

    for session in sessions:
        logger.info(f"USER {session.user_id} PROMPT    : {tokenizer.decode(session.prompt_ids)!r}")
        logger.info(
            f"USER {session.user_id} GENERATED : {tokenizer.decode(session.generated)!r} "
            f"({len(session.generated)} tokens, final pos {session.pos})"
        )
    logger.info(f"pool usage after generation: {model.session_usage()}")
    # Whether two prompts produce different text is a property of the *model*, not of
    # the paging: a stack truncated by ``DEEPSEEK_V4_DECODE_LAYERS`` emits much the same
    # gibberish for any prompt. So it is logged, not asserted.
    if sessions[0].generated == sessions[1].generated:
        logger.warning("both users produced identical tokens (expected on a heavily truncated stack)")

    # --- isolation: user 0's tokens must not depend on user 1 running -------- #
    # Same prompt and the same greedy decode, but alone in its own session, for as
    # many tokens as user 0 produced before user 1's first burst.
    solo_steps = _BURST_STEPS
    solo = UserSession(user_id=0, sid=model.open_session(), prompt_ids=prompt_ids[0])
    for pos in range(solo.prompt_len):
        solo.next_token = _decode(model, solo, solo.prompt_ids[pos], pos)
    solo.pos = solo.prompt_len
    solo.generated.append(solo.next_token)
    for _ in range(solo_steps):
        solo.next_token = _decode(model, solo, solo.next_token, solo.pos)
        solo.pos += 1
        solo.generated.append(solo.next_token)

    interleaved = sessions[0].generated[: len(solo.generated)]
    assert solo.generated == interleaved, (
        "user 0's tokens changed depending on whether user 1 was interleaved:\n"
        f"  interleaved: {tokenizer.decode(interleaved)!r}\n"
        f"  solo       : {tokenizer.decode(solo.generated)!r}"
    )

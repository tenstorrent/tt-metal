# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-user paged-KV decode demo for ``DeepSeekV4Model``.

Two independent chat sessions share one physical paged sliding-KV pool per
sliding-attention layer (block size 64, GPT-OSS / vLLM style). Each user has a
``page_table`` row mapping logical blocks to disjoint physical blocks, so
interleaved decode steps do not clobber the other user's cache.

Flow:
  1. Prefill two different prompts (one token at a time, per user).
  2. Generate in bursts of 64 tokens for user 0, then 64 for user 1, and repeat.

CSA/HCA layers keep per-user compressor caches; sliding-attention layers use
``paged_update_cache`` + ``paged_scaled_dot_product_attention_decode``.

Run (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_multi_user_paged_decode_demo.py
"""

from __future__ import annotations

import math
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
from models.experimental.deepseek_v4_flash.tt.paged_cache import PagedCacheConfig
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
_PAGE_BLOCK_SIZE = 64


def _pad_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


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
    """Interleaved two-user decode with shared paged sliding-KV pools."""
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    prompts = [_PROMPT_A, _PROMPT_B]
    sessions = [UserSession(user_id=u, prompt_ids=_tokenize_prompt(tokenizer, prompts[u])) for u in range(_NUM_USERS)]
    max_prompt = max(s.prompt_len for s in sessions)
    total_gen = _BURST_STEPS * _NUM_ROUNDS * _NUM_USERS
    max_seq = _pad_to_tile(max_prompt + total_gen)
    crs = {int(v) for v in config.compress_rates.values()}
    if crs:
        step = math.lcm(32, _PAGE_BLOCK_SIZE, *crs)
        max_seq = ((max_seq + step - 1) // step) * step

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
    model.reset_multi_user_paged_caches(max_seq, num_users=_NUM_USERS, block_size=_PAGE_BLOCK_SIZE)

    paged_cfg = model._paged_cfg
    assert isinstance(paged_cfg, PagedCacheConfig)
    sliding_layers = [li for li in range(model.num_layers) if config.layer_types[li] == "sliding_attention"]
    logger.info(
        f"multi-user paged decode: users={_NUM_USERS} block_size={_PAGE_BLOCK_SIZE} "
        f"blocks_per_user={paged_cfg.blocks_per_user} total_blocks={paged_cfg.total_blocks} "
        f"max_seq={max_seq} sliding_layers={sliding_layers}"
    )
    logger.info(f"page_table (host):\n{model._page_table_host}")

    # --- prefill each user's prompt ----------------------------------------- #
    for session in sessions:
        logger.info(f"prefill user {session.user_id}: {prompts[session.user_id]!r} ({session.prompt_len} tokens)")
        for pos in range(session.prompt_len):
            hidden = model.decode_user(session.user_id, session.prompt_ids[pos], pos, rope)
            logits = ttnn.to_torch(lm_head(hidden)).reshape(1, -1).float()
            session.next_token = int(logits[0].argmax().item())
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
                if pos >= max_seq:
                    pytest.fail(f"user {session.user_id} exceeded max_seq {max_seq} at pos {pos}")
                hidden = model.decode_user(session.user_id, session.next_token, pos, rope)
                logits = ttnn.to_torch(lm_head(hidden)).reshape(1, -1).float()
                session.next_token = int(logits[0].argmax().item())
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

    assert sessions[0].generated and sessions[1].generated
    assert sessions[0].generated != sessions[1].generated, "users should diverge after different prompts"

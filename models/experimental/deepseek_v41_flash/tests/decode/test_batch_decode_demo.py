# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""B=4 decode demo: four chat prompts answered together, one :meth:`~DeepSeekV41Model.decode_batch_block` step per
token of every user (``test_full_model_decode_demo``'s model, built with ``batch_block=True``).

The instance is one persistent TP4 stage (row 4) with every MoE on the persistent TP12 server below it. Each user is
a session of its own. Every step feeds each live row its next prompt token (while still in its prompt) or its last
pick, so the prompts are prefilled by the same batch steps that decode; a row leaves at EOS or after
``DEEPSEEK_V4_MAX_NEW_TOKENS`` new tokens, and the remaining rows' steps run with fewer users (padded with the scratch
session). The log reports every reply, each user's time to first token and decode rate, and the batch's tokens per
second.

``DEEPSEEK_V41_BATCH_PROMPTS``: a file with one prompt per line (the first four are used; default: four of
``test_multi_session_decode``'s texts). ``DEEPSEEK_V4_TEMPERATURE`` (default 0: greedy) samples on device. It needs the
TP12 expert cache (``tools/fill_expert_cache.py --all-layers --tp-size 12 --mesh-rows 3``)::

    DEEPSEEK_V4_MAX_NEW_TOKENS=256 pytest models/experimental/deepseek_v41_flash/tests/decode/test_batch_decode_demo.py -s
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _L1_SMALL_SIZE,
    _PAGE_BLOCK_SIZE,
    _SEED,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _stage_engram_or_skip,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_multi_session_decode import _TEXTS
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# DEEPSEEK_V41_BATCH_ROWS=2: B=2 batch steps (2-row blocks) instead of B=4.
_ROWS = int(os.environ.get("DEEPSEEK_V41_BATCH_ROWS", "4"))
assert _ROWS in (2, 4), f"DEEPSEEK_V41_BATCH_ROWS takes 2 or 4, got {_ROWS}"
_MAX_NEW_TOKENS = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "128"))
_TEMPERATURE = float(os.environ.get("DEEPSEEK_V4_TEMPERATURE", "0"))
_PROMPTS_FILE = os.environ.get("DEEPSEEK_V41_BATCH_PROMPTS")
# As test_batch_decode: the batch step's programs need L1_SMALL past test_full_model_decode_demo's.
_BATCH_L1_SMALL_SIZE = max(_L1_SMALL_SIZE, 16384)  # as demo/server.py under --persistent-moe (B=2 needs past 12 KB)


def _prompts() -> list[str]:
    if _PROMPTS_FILE:
        with open(_PROMPTS_FILE) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert lines, f"{_PROMPTS_FILE} holds no prompts"
        return lines[:_ROWS]
    return list(_TEXTS[1 : _ROWS + 1])


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": _BATCH_L1_SMALL_SIZE,
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_1stage_persistent_moe12"])
def test_batch_decode_demo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    _stage_engram_or_skip(loader, config)
    texts = _prompts()
    users = len(texts)
    prompts = [_tokenize_chat(tokenizer, text) for text in texts]
    longest = max(len(p) for p in prompts) + _MAX_NEW_TOKENS
    max_seq = _traced_max_seq(config, longest + 1, num_layers)
    eos_id = tokenizer.eos_token_id

    with contextlib.ExitStack() as stack:
        model, lm_head, _, _ = _construct_model(
            mesh_device,
            stack,
            loader=loader,
            config=config,
            num_stages=1,
            persistent_moe=True,
            verify_rows=_ROWS,
            batch_block=True,
        )
        rope = _build_rope(config, loader.snapshot_dir, max_seq)
        # One slot per user and the scratch slot that pads a short step.
        model.prepare_static_decode(
            rope,
            max_seq,
            lm_head=lm_head,
            num_sessions=_ROWS + 1,
            total_tokens=(_ROWS + 1) * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=True,
        )
        model.device_sampler.set_sampling(_TEMPERATURE, _SEED)
        model.prepare_batch_decode()
        sids = [model.open_session() for _ in range(users)]
        t0 = time.perf_counter()
        model.capture_batch_traces(prompts[0][0])
        logger.info(f"one-row and batch traces captured in {time.perf_counter() - t0:.0f}s")
        for sid in sids:
            model.ensure_session_capacity(max_seq - 1, sid)

        pos = [0] * users
        replies = [[] for _ in range(users)]
        done = [False] * users
        first_token_at = [None] * users
        finished_at = [None] * users
        steps = 0
        step_ms = []
        start = time.perf_counter()
        while not all(done):
            rows = [i for i in range(users) if not done[i]]
            # A row in its prompt is fed its next prompt token; past it, its last pick.
            feed = [prompts[i][pos[i]] if pos[i] < len(prompts[i]) else replies[i][-1] for i in rows]
            t_step = time.perf_counter()
            picks = model.decode_batch_block(feed, [sids[i] for i in rows], positions=[pos[i] for i in rows])
            now = time.perf_counter()
            step_ms.append(1e3 * (now - t_step))
            steps += 1
            for i, pick in zip(rows, picks):
                pos[i] += 1
                if pos[i] < len(prompts[i]):
                    continue  # still prefilling: the pick at a prompt position is not generated
                if first_token_at[i] is None:
                    first_token_at[i] = now
                replies[i].append(int(pick))
                if int(pick) == eos_id or len(replies[i]) >= _MAX_NEW_TOKENS:
                    done[i] = True
                    finished_at[i] = now
        elapsed = time.perf_counter() - start

        for i in range(users):
            text = tokenizer.decode([t for t in replies[i] if t != eos_id])
            gen = len(replies[i])
            ttft = first_token_at[i] - start
            decode_s = finished_at[i] - first_token_at[i]
            rate = (gen - 1) / decode_s if gen > 1 and decode_s > 0 else float("nan")
            logger.info(
                f"\n=== user {i} ({len(prompts[i])} prompt tokens) ===\n{texts[i]}\n--- reply ({gen} tokens"
                f"{', EOS' if replies[i] and replies[i][-1] == eos_id else ''}) ---\n{text}\n"
                f"time to first token {ttft:.2f}s, decode {rate:.1f} tok/s"
            )
        generated = sum(len(r) for r in replies)
        logger.info(
            f"{users} users, {steps} batch steps in {elapsed:.2f}s (median step {statistics.median(step_ms):.2f} ms): "
            f"{generated} generated tokens, {sum(len(p) for p in prompts) + generated} tokens processed, "
            f"{generated / elapsed:.1f} generated tok/s over the batch"
        )
        assert all(replies), "every user gets a reply"

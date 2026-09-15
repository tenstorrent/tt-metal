# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE, NUM_LAYERS, load_xtts_state_dict
from models.experimental.xtts.reference.xtts_conditioning import NUM_LATENTS
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN
from models.experimental.xtts.reference.xtts_gpt_model import NUM_TEXT_TOKENS
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32
TEXT_LEN = 96
MAX_NEW_TOKENS = 240


def _num_layers() -> int:
    """Read GPT layer count from XTTS_PERF_NUM_LAYERS or use the default."""
    raw = os.environ.get("XTTS_PERF_NUM_LAYERS")
    return NUM_LAYERS if raw is None or not raw.strip() else int(raw)


def _tracy_signpost_available() -> bool:
    """Return whether tracy.signpost is importable for profile markers."""
    try:
        pass

        return True
    except ImportError:
        return False


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_profile_single_step_decode(device):
    """Profile one GPT decode_on_device step with optional Tracy signposts."""
    num_layers = _num_layers()
    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    torch.manual_seed(0)
    sd = load_xtts_state_dict()
    gpt = TtXttsGptModel(sd, device, num_layers=num_layers)

    text_ids = torch.randint(0, NUM_TEXT_TOKENS, (1, TEXT_LEN), dtype=torch.long)
    cond = torch.randn(1, NUM_LATENTS, HIDDEN_SIZE) * 0.1
    cond_tt = ttnn.from_torch(cond, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    prompt_len = NUM_LATENTS + TEXT_LEN
    max_seq = -(-(prompt_len + MAX_NEW_TOKENS + 1) // TILE) * TILE

    kv = gpt.prefill(text_ids, cond_tt, max_seq)
    ttnn.synchronize_device(device)

    tok = gpt._pos_ids(START_AUDIO_TOKEN)
    mel_pos = gpt._pos_ids(0)
    cache_pos = gpt.cache_pos(prompt_len)

    gpt.decode_on_device(tok, mel_pos, cache_pos, kv, write_idx=None)
    ttnn.synchronize_device(device)

    ttnn.ReadDeviceProfiler(device)

    if use_signpost:
        from tracy import signpost

        signpost("start")

    logits, latent = gpt.decode_on_device(tok, mel_pos, cache_pos, kv, write_idx=None)

    ttnn.synchronize_device(device)
    if use_signpost:
        from tracy import signpost

        signpost("stop")
    ttnn.ReadDeviceProfiler(device)

    assert logits is not None and latent is not None
    logger.info(
        f"Profile workload complete: single decode layers={num_layers} prompt_len={prompt_len} "
        f"max_seq={max_seq} write=traced-onehot, signposts={'on' if use_signpost else 'off'}"
    )

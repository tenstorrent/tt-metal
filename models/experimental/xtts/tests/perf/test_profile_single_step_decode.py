# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-step GPT DECODE workload for Tracy / device-perf profiling.

Measures **one** ``TtXttsGptModel.decode_on_device`` call — the per-audio-code step the
model spends essentially all of its time in (~8 ms/code; see the README's Performance
section) — between Tracy ``start``/``stop`` signposts. Weight load, KV-cache allocation,
the untimed prefill that fills the cache, and a warmup decode all happen OUTSIDE the
window.

Two choices worth knowing:

  * ``write_idx=None`` — the **traced** cache-write path (a data-driven one-hot select
    over the whole cache), not the eager ``update_cache`` fast path. The demo and the
    perf baseline both run traced, so this is what production actually pays.
  * ``max_seq`` sized for the demo's 240-code budget. Decode attends over the ENTIRE
    fixed cache under an additive mask regardless of the current position, so decode
    cost is driven by ``max_seq`` rather than by how far into the sequence you are —
    profiling at a short cache would understate it.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run --op-support-count 100000 -m pytest \\
        models/experimental/xtts/tests/perf/test_profile_single_step_decode.py \\
        ::test_profile_single_step_decode -v

Device perf CSV/JSON dump (the outer driver spawns this under Tracy)::

    python models/experimental/xtts/tests/perf/test_device_perf_single_step_decode.py

Env:
  ``XTTS_PERF_NUM_LAYERS`` — GPT depth to profile (default 30, the full stack; set 1 to
  profile a single decoder layer in isolation).
"""

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
TEXT_LEN = 96  # the demo's wrapped-text length
MAX_NEW_TOKENS = 240  # the demo's audio-code budget; sizes the fixed KV cache


def _num_layers() -> int:
    raw = os.environ.get("XTTS_PERF_NUM_LAYERS")
    return NUM_LAYERS if raw is None or not raw.strip() else int(raw)


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_profile_single_step_decode(device):
    """One warm GPT decode step with Tracy start/stop around ``decode_on_device`` only."""
    num_layers = _num_layers()
    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    torch.manual_seed(0)
    sd = load_xtts_state_dict()
    gpt = TtXttsGptModel(sd, device, num_layers=num_layers)

    # Synthetic prompt: op timings depend on SHAPE, not values (see the prefill workload).
    text_ids = torch.randint(0, NUM_TEXT_TOKENS, (1, TEXT_LEN), dtype=torch.long)
    cond = torch.randn(1, NUM_LATENTS, HIDDEN_SIZE) * 0.1
    cond_tt = ttnn.from_torch(cond, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    prompt_len = NUM_LATENTS + TEXT_LEN
    max_seq = -(-(prompt_len + MAX_NEW_TOKENS + 1) // TILE) * TILE  # production sizing (xtts_generator)

    # Untimed prefill seeds the cache; the measured window is the decode step alone.
    kv = gpt.prefill(text_ids, cond_tt, max_seq)
    ttnn.synchronize_device(device)

    # Per-step index tensors are host->device writes — build them OUTSIDE the window, exactly as
    # the traced decode does (it binds them once and updates them on device).
    tok = gpt._pos_ids(START_AUDIO_TOKEN)
    mel_pos = gpt._pos_ids(0)
    cache_pos = gpt.cache_pos(prompt_len)

    # Untimed warmup decode at the same position, so the measured step hits a warm program cache.
    gpt.decode_on_device(tok, mel_pos, cache_pos, kv, write_idx=None)
    ttnn.synchronize_device(device)

    # Drain load/warmup/prefill markers so the signposted region is not dropped (Voxtral pattern).
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

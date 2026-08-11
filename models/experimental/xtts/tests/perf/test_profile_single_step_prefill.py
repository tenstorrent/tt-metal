# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-step GPT PREFILL workload for Tracy / device-perf profiling.

Measures **one** ``TtXttsGptModel.prefill_on_device`` call — embeddings ->
``concat([cond | text])`` -> 30 causal blocks -> ``fill_cache`` — between Tracy
``start``/``stop`` signposts. Everything that is not that call (weight load, KV-cache
allocation, the host->device text upload, and a full warmup pass to fill the program
cache) happens OUTSIDE the window.

The measured window is ``prefill_on_device``, not the ``prefill`` main, deliberately:
``prefill`` also calls ``alloc_static_kv``, which is a host->device write of zeroed
caches — allocation, not compute, and it would otherwise land inside the signposts.
``prefill_on_device`` is the same core the fully-traced path captures, so this profiles
production code.

Shapes match the demo's real configuration (see the README's Performance section):
96 wrapped text tokens + 32 conditioning latents = 128 prompt positions, and a KV cache
sized for the demo's 240-code budget (``max_seq`` 384). Decode attends over the whole
fixed cache, so ``max_seq`` is a real cost driver and is kept honest here.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run --op-support-count 100000 -m pytest \\
        models/experimental/xtts/tests/perf/test_profile_single_step_prefill.py \\
        ::test_profile_single_step_prefill -v

Device perf CSV/JSON dump (the outer driver spawns this under Tracy)::

    python models/experimental/xtts/tests/perf/test_device_perf_single_step_prefill.py

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
def test_profile_single_step_prefill(device):
    """One warm GPT prefill with Tracy start/stop around ``prefill_on_device`` only."""
    num_layers = _num_layers()
    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    torch.manual_seed(0)
    sd = load_xtts_state_dict()
    gpt = TtXttsGptModel(sd, device, num_layers=num_layers)

    # Synthetic prompt: prefill op timings depend on SHAPE, not on values, so this needs no
    # reference audio or conditioning encoder — keeping the GPT profile independent of them.
    text_ids = torch.randint(0, NUM_TEXT_TOKENS, (1, TEXT_LEN), dtype=torch.long)
    cond = torch.randn(1, NUM_LATENTS, HIDDEN_SIZE) * 0.1
    cond_tt = ttnn.from_torch(cond, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    prompt_len = NUM_LATENTS + TEXT_LEN
    max_seq = -(-(prompt_len + MAX_NEW_TOKENS + 1) // TILE) * TILE  # production sizing (xtts_generator)

    # Untimed warmup: same shape, so the measured pass hits a warm program cache.
    gpt.prefill(text_ids, cond_tt, max_seq)
    ttnn.synchronize_device(device)

    # Allocation + the host->device text upload stay OUTSIDE the measured window.
    gpt.alloc_static_kv(max_seq)
    text_dev = gpt.text_ids_to_device(text_ids)
    ttnn.synchronize_device(device)

    # Drain load/warmup markers so the signposted region is not dropped (Voxtral pattern).
    ttnn.ReadDeviceProfiler(device)

    if use_signpost:
        from tracy import signpost

        signpost("start")

    gpt.prefill_on_device(text_dev, cond_tt)

    ttnn.synchronize_device(device)
    if use_signpost:
        from tracy import signpost

        signpost("stop")
    ttnn.ReadDeviceProfiler(device)

    logger.info(
        f"Profile workload complete: single prefill layers={num_layers} text_len={TEXT_LEN} "
        f"prompt_len={prompt_len} max_seq={max_seq}, signposts={'on' if use_signpost else 'off'}"
    )

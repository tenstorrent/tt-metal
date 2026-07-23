# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-step LM prefill chunk workload for Tracy / device-perf profiling.

Runs **one** warm ``TTVibeVoiceLM.forward`` prefill chunk (default length
``PREFILL_CHUNK_SIZE`` = 256). Mask/embed/KV setup and warmup run outside the
Tracy ``start``/``stop`` window.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run \\
        pytest models/experimental/vibevoice/tests/perf/test_profile_single_step_prefill.py \\
        ::test_profile_single_step_prefill -v

Device perf CSV/JSON dump (outer driver spawns this under Tracy)::

    python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_prefill.py

Env:
  ``VV_PREFILL_PERF_SEQ_LEN`` — chunk length (clamped to ``PREFILL_CHUNK_SIZE``).
  ``VV_PREFILL_PERF_START_POS`` — chunk start (multiple of ``PREFILL_CHUNK_SIZE``).
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.vibevoice.tests.pcc.pcc_helpers import PREFILL_CHUNK_SIZE, build_tt_lm

NUM_WARMUP_ITERS = 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


def _run_single_prefill_chunk(
    mesh_device,
    lm_tt,
    inputs_embeds: ttnn.Tensor,
    kv_cache,
    *,
    start_pos: int,
    chunk_len: int,
    use_signpost: bool = False,
) -> None:
    hidden_dim = inputs_embeds.shape[-1]
    chunk = ttnn.slice(
        inputs_embeds,
        [0, 0, start_pos, 0],
        [1, 1, start_pos + chunk_len, hidden_dim],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if use_signpost:
        from tracy import signpost

        signpost("start")

    _, hidden = lm_tt.forward(
        chunk,
        start_pos=start_pos,
        kv_cache=kv_cache,
        return_last_hidden=True,
    )

    if use_signpost:
        ttnn.synchronize_device(mesh_device)
        signpost("stop")

    assert hidden is not None


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_profile_single_step_prefill(mesh_device, vv_config, lm_state):
    """One warm LM prefill chunk with Tracy start/stop around ``forward`` only."""
    start_pos = _env_int("VV_PREFILL_PERF_START_POS", 0)
    if start_pos % PREFILL_CHUNK_SIZE != 0:
        raise ValueError(
            f"VV_PREFILL_PERF_START_POS={start_pos} must be a multiple of PREFILL_CHUNK_SIZE={PREFILL_CHUNK_SIZE}"
        )
    chunk_len = min(_env_int("VV_PREFILL_PERF_SEQ_LEN", PREFILL_CHUNK_SIZE), PREFILL_CHUNK_SIZE)
    total_len = start_pos + chunk_len

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    torch.manual_seed(0)
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_len), dtype=torch.long)

    # Untimed full prefill: compile + fill program cache for this shape.
    kv_warm = lm_tt.alloc_kv_cache(total_len + 8)
    for _ in range(NUM_WARMUP_ITERS):
        lm_tt.prefill(input_ids, kv_cache=kv_warm, return_last_hidden=True)
        ttnn.synchronize_device(mesh_device)
    del kv_warm

    kv_cache = lm_tt.alloc_kv_cache(total_len + 8)
    inputs_embeds = lm_tt._embed(input_ids)
    hidden_dim = inputs_embeds.shape[-1]

    # Untimed prefix so the measured chunk sees a realistic KV prefix.
    if start_pos > 0:
        for s in range(0, start_pos, PREFILL_CHUNK_SIZE):
            e = min(s + PREFILL_CHUNK_SIZE, start_pos)
            prefix = ttnn.slice(
                inputs_embeds,
                [0, 0, s, 0],
                [1, 1, e, hidden_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            lm_tt.forward(prefix, start_pos=s, kv_cache=kv_cache, return_last_hidden=True)
        ttnn.synchronize_device(mesh_device)

    # Drain load/warmup markers so the signposted region is not dropped (Voxtral pattern).
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    _run_single_prefill_chunk(
        mesh_device,
        lm_tt,
        inputs_embeds,
        kv_cache,
        start_pos=start_pos,
        chunk_len=chunk_len,
        use_signpost=use_signpost,
    )
    if not use_signpost:
        ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    logger.info(
        f"Profile workload complete: single prefill chunk_len={chunk_len} start_pos={start_pos}, "
        f"signposts={'on' if use_signpost else 'off'}"
    )

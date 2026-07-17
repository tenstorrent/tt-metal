# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dump a Tracy op-level perf window for one VibeVoice LM prefill chunk.

The measured window is a **single** ``forward(...)`` chunk step (default chunk
size 256), not the full ``prefill_embeds`` multi-chunk loop. Warmup still runs a
full prefill so program cache is hot.

Run under Tracy (from tt-metal root)::

    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
    bash models/experimental/vibevoice/tests/perf/run_prefill_tracy.sh
    # or:
    VV_PREFILL_PERF_SEQ_LEN=256 python -m tracy -v -r -p --op-support-count 100000 \\
      -m "pytest models/experimental/vibevoice/tests/perf/test_prefill_perf_dump.py \\
             -k test_lm_prefill_tracy_signposts -s"
    CSV=$(ls -td generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    tt-perf-report "$CSV" --start-signpost start --end-signpost stop \\
      > models/experimental/vibevoice/lm/prefill_expN.txt

``VV_PREFILL_PERF_SEQ_LEN`` is the chunk length profiled (clamped to the LM
chunk size, default 256). For a later chunk attending a longer prefix, set
``VV_PREFILL_PERF_START_POS`` (e.g. 256 → second chunk after an untimed first).
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.vibevoice.tests.pcc.lm_pcc_common import PREFILL_CHUNK_SIZE, build_tt_lm


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _prefill_perf_chunk_lens() -> list[int]:
    """Chunk lengths to profile; override via ``VV_PREFILL_PERF_SEQ_LEN``."""
    env = os.environ.get("VV_PREFILL_PERF_SEQ_LEN")
    if env is not None and env.strip():
        return [min(int(env), PREFILL_CHUNK_SIZE)]
    return [32, 256]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("chunk_len", _prefill_perf_chunk_lens())
def test_lm_prefill_tracy_signposts(mesh_device, vv_config, lm_state, chunk_len):
    """Warm **single-chunk** LM prefill with Tracy start/stop around one forward.

    Does **not** profile the full multi-chunk ``prefill_embeds`` loop. Optional
    ``VV_PREFILL_PERF_START_POS`` selects which chunk (0 = first; 256 = second
    after an untimed prefix fill).
    """
    start_pos = _env_int("VV_PREFILL_PERF_START_POS", 0)
    if start_pos % PREFILL_CHUNK_SIZE != 0:
        raise ValueError(
            f"VV_PREFILL_PERF_START_POS={start_pos} must be a multiple of " f"PREFILL_CHUNK_SIZE={PREFILL_CHUNK_SIZE}"
        )
    chunk_len = min(int(chunk_len), PREFILL_CHUNK_SIZE)
    total_len = start_pos + chunk_len

    torch.manual_seed(0)
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_len), dtype=torch.long)

    # Untimed full prefill: compile + fill program cache for this shape.
    kv_warm = lm_tt.alloc_kv_cache(total_len + 8)
    lm_tt.prefill(input_ids, kv_cache=kv_warm, return_last_hidden=True)
    ttnn.synchronize_device(mesh_device)
    del kv_warm

    kv_cache = lm_tt.alloc_kv_cache(total_len + 8)
    inputs_embeds = lm_tt._embed(input_ids)
    hidden_dim = inputs_embeds.shape[-1]

    # Untimed prefix chunks so the measured step sees a realistic KV prefix.
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

    chunk = ttnn.slice(
        inputs_embeds,
        [0, 0, start_pos, 0],
        [1, 1, start_pos + chunk_len, hidden_dim],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)

    import tracy

    tracy.signpost("start")
    _, hidden = lm_tt.forward(
        chunk,
        start_pos=start_pos,
        kv_cache=kv_cache,
        return_last_hidden=True,
    )
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("stop")

    assert hidden is not None
    print(
        f"[test_prefill_perf_dump] Tracy start/stop around ONE prefill chunk: "
        f"chunk_len={chunk_len} start_pos={start_pos} "
        f"(not the full chunked prefill loop)",
        flush=True,
    )

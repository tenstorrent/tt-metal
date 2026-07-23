# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf inference workload: one LM prefill chunk + a few decode steps.

Mirrors Voxtral ``test_voxtral_tts_perf_inference.py``:

  * Weight load / warmup dispatch many device ops and can overflow the Tracy
    DRAM profiler buffer. ``ttnn.ReadDeviceProfiler()`` is called after warmup
    to clear that buffer before the measured region.
  * ``signpost("start"/"stop")`` brackets only the measured eager LM pass so
    ``test_vibevoice_device_perf.py`` can aggregate with ``has_signposts=True``.

Correctness is covered by the PCC suite. No metal trace (Tracy per-op records
do not reconcile cleanly across trace replays).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.vibevoice.tests.pcc.pcc_helpers import PREFILL_CHUNK_SIZE, build_tt_lm

try:
    from tracy import signpost as _signpost
except ImportError:

    def _signpost(name: str) -> None:
        pass


# Tiny measured window (same idea as Voxtral max_tokens=2): one prefill chunk + 2 decode steps.
_PREFILL_LEN = PREFILL_CHUNK_SIZE
_NUM_DECODE_STEPS = 2


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm(mesh_device, vv_config, lm_state):
    """Eager LM prefill + decode, Tracy-measured between start/stop only."""
    torch.manual_seed(0)
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)

    prefill_len = _PREFILL_LEN
    total_len = prefill_len + _NUM_DECODE_STEPS
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_len), dtype=torch.long)

    # Untimed warmup: compile + fill program cache (outside the measured window).
    kv_warm = lm_tt.alloc_kv_cache(total_len + 8)
    lm_tt.prefill(input_ids[:, :prefill_len], kv_cache=kv_warm, return_last_hidden=True)
    for i in range(_NUM_DECODE_STEPS):
        lm_tt.decode_step(input_ids[:, prefill_len + i : prefill_len + i + 1], prefill_len + i, kv_warm)
    ttnn.synchronize_device(mesh_device)
    del kv_warm

    # Drain load/warmup markers so they do not overflow the buffer / drop measured ops.
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    kv_cache = lm_tt.alloc_kv_cache(total_len + 8)

    _signpost("start")
    lm_tt.prefill(input_ids[:, :prefill_len], kv_cache=kv_cache, return_last_hidden=True)
    for i in range(_NUM_DECODE_STEPS):
        lm_tt.decode_step(input_ids[:, prefill_len + i : prefill_len + i + 1], prefill_len + i, kv_cache)
    ttnn.synchronize_device(mesh_device)
    _signpost("stop")

    # Drain measured ops before teardown (avoids profiler cleanup issues on overflow-marked buffers).
    ttnn.ReadDeviceProfiler(mesh_device)

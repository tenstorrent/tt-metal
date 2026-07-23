# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-step LM decode workload for Tracy / device-perf profiling.

Prefills ``PREFILL_LEN`` tokens **outside** the Tracy window, then measures
**one** ``decode_step`` between ``start``/``stop`` signposts.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run \\
        pytest models/experimental/vibevoice/tests/perf/test_profile_single_step_decode.py \\
        ::test_profile_single_step_decode -v

Device perf CSV/JSON dump (outer driver spawns this under Tracy)::

    python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

Env:
  ``VV_DECODE_PERF_PREFILL_LEN`` — KV prefix length before the measured decode (default 256).
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


def _run_single_decode_step(
    mesh_device,
    lm_tt,
    input_id: torch.Tensor,
    start_pos: int,
    kv_cache,
    *,
    use_signpost: bool = False,
) -> None:
    if use_signpost:
        from tracy import signpost

        signpost("start")

    logits = lm_tt.decode_step(input_id, start_pos, kv_cache, return_last_hidden=False)

    if use_signpost:
        ttnn.synchronize_device(mesh_device)
        signpost("stop")

    assert logits is not None


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_profile_single_step_decode(mesh_device, vv_config, lm_state):
    """One warm LM decode step with Tracy start/stop around ``decode_step`` only."""
    prefill_len = _env_int("VV_DECODE_PERF_PREFILL_LEN", PREFILL_CHUNK_SIZE)
    decode_pos = prefill_len
    total_len = prefill_len + 1

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    torch.manual_seed(0)
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, total_len), dtype=torch.long)

    # Untimed warmup: same prefill+decode shape to fill program cache.
    kv_warm = lm_tt.alloc_kv_cache(total_len + 8)
    for _ in range(NUM_WARMUP_ITERS):
        lm_tt.prefill(input_ids[:, :prefill_len], kv_cache=kv_warm, return_last_hidden=True)
        lm_tt.decode_step(input_ids[:, decode_pos : decode_pos + 1], decode_pos, kv_warm)
        ttnn.synchronize_device(mesh_device)
    del kv_warm

    # Fresh cache: untimed prefill fills KV; measured window is decode only.
    kv_cache = lm_tt.alloc_kv_cache(total_len + 8)
    lm_tt.prefill(input_ids[:, :prefill_len], kv_cache=kv_cache, return_last_hidden=True)
    ttnn.synchronize_device(mesh_device)

    # Drain load/warmup/prefill markers so the signposted decode is not dropped (Voxtral pattern).
    ttnn.ReadDeviceProfiler(mesh_device)

    _run_single_decode_step(
        mesh_device,
        lm_tt,
        input_ids[:, decode_pos : decode_pos + 1],
        decode_pos,
        kv_cache,
        use_signpost=use_signpost,
    )
    if not use_signpost:
        ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    logger.info(
        f"Profile workload complete: single decode at pos={decode_pos} "
        f"(prefill_len={prefill_len}), signposts={'on' if use_signpost else 'off'}"
    )

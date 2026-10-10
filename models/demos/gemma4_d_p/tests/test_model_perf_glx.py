# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLX prefill perf gate for Gemma4 disaggregated prefill.

For each chunk size, builds the model and runs a traced 256k prefill with ``_measure_traced``
(real text, 8x4 = CP8/TP4), then gates the cumulative wall and device time to reach each context
length in REPORT_CONTEXT_KS. All chunk sizes run in one process under ``_shared_device_weights``, so
each cached weight is written to the device once, as in
``text_demo_prefill.py::test_prefill_chunk_sweep_traced``. Wall time covers host staging,
trace dispatch and waiting for the device, but not compile or capture. Device time is the sum of
execute_trace + synchronize per chunk, without host staging, so it varies less and gets a tighter
margin. ``_measure_traced`` logs both as the ``[traced_perf] context wall and device times`` table.
"""

import gc

import pytest
import torch
from loguru import logger

from models.demos.gemma4_d_p.demo.text_demo_prefill import (
    REPORT_CONTEXT_KS,
    TRACE_REGION_SIZE,
    _build_prefill_model,
    _hf_model_id,
    _measure_traced,
    _mesh_config,
    _shared_device_weights,
)
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric

CONTEXT_LEN = 262144
CHUNK_SIZES = (2048, 4096, 8192)
# Baselines in ms to reach each context, keyed by (chunk_size, context_k), 1k = 1024 tokens.
WALL_MS_BASELINES = {
    (2048, 1): 65.4,
    (2048, 10): 334.4,
    (2048, 100): 3829.6,
    (2048, 256): 11928.6,
    (4096, 1): 83.8,
    (4096, 10): 256.6,
    (4096, 100): 2457.5,
    (4096, 256): 7768.7,
    (8192, 1): 122.9,
    (8192, 10): 252.3,
    (8192, 100): 2080.2,
    (8192, 256): 6893.6,
}
DEVICE_MS_BASELINES = {
    (2048, 1): 62.9,
    (2048, 10): 321.0,
    (2048, 100): 3678.0,
    (2048, 256): 11541.7,
    (4096, 1): 81.6,
    (4096, 10): 248.6,
    (4096, 100): 2387.6,
    (4096, 256): 7585.6,
    (8192, 1): 121.5,
    (8192, 10): 248.3,
    (8192, 100): 2002.8,
    (8192, 256): 6742.9,
}
# Allowed run-to-run variation either side of the baseline, keyed by context_k.
# Device time excludes host staging, so its margins are tighter.
WALL_MS_MARGINS = {1: 0.02, 10: 0.02, 100: 0.015, 256: 0.01}
DEVICE_MS_MARGINS = {1: 0.01, 10: 0.01, 100: 0.01, 256: 0.01}


def _ranges(baselines, margins):
    """Return {(chunk_size, context_k): (min ms, max ms)}."""
    return {
        (chunk_size, context_k): (ms * (1 - margins[context_k]), ms * (1 + margins[context_k]))
        for (chunk_size, context_k), ms in baselines.items()
    }


WALL_MS_RANGES = _ranges(WALL_MS_BASELINES, WALL_MS_MARGINS)
DEVICE_MS_RANGES = _ranges(DEVICE_MS_BASELINES, DEVICE_MS_MARGINS)
RESULT_HEADER = (
    f"{'Context':>8} | {'Chunk':>6} | {'Metric':>6} | {'Time (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | Result"
)


@torch.no_grad()
@pytest.mark.timeout(540)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_model_perf_glx(mesh_device, reset_seeds):
    """Check the 1k/10k/100k/256k wall and device times of a traced 256k prefill against their baselines.

    Runs every chunk size in one process, writing weights once. Fails if a time is outside its
    range on either side: too slow is a regression, too fast means the baseline needs updating.
    """
    for metric, ranges in (("wall", WALL_MS_RANGES), ("device", DEVICE_MS_RANGES)):
        if unset := [(c, k) for c in CHUNK_SIZES for k in REPORT_CONTEXT_KS if (c, k) not in ranges]:
            pytest.fail(f"no {metric}-time baseline for (chunk_size, context_k) {unset} (context in units of 1k)")

    mesh_config = _mesh_config(mesh_device)
    hf_model_id = _hf_model_id()

    rows, failures = [], []
    with _shared_device_weights():
        for chunk_size in CHUNK_SIZES:
            logger.info(f"[perf_gate] ===== chunk_size={chunk_size} context_len={CONTEXT_LEN} =====")
            model_args, model, _kv_cache = _build_prefill_model(
                mesh_config=mesh_config,
                hf_model_id=hf_model_id,
                chunk_size=chunk_size,
                context_len=CONTEXT_LEN,
            )
            wall_ms, device_ms = _measure_traced(
                mesh_device, mesh_config, model_args, model, hf_model_id, CONTEXT_LEN, chunk_size, token_source="text"
            )
            del model_args, model, _kv_cache
            gc.collect()
            # Cached programs hold op-allocated L1 semaphores that fragment L1 for the next model's circular buffers.
            mesh_device.clear_program_cache()

            for context_k in REPORT_CONTEXT_KS:
                for metric, measured, ranges in (
                    ("wall", wall_ms, WALL_MS_RANGES),
                    ("device", device_ms, DEVICE_MS_RANGES),
                ):
                    ms = measured[context_k]
                    min_ms, max_ms = ranges[(chunk_size, context_k)]
                    if ms > max_ms:
                        result = "SLOW"
                        failures.append(f"{context_k}k@{chunk_size} {metric}: {ms:.1f}ms > max {max_ms:.1f}ms")
                    elif ms < min_ms:
                        result = "FAST"
                        failures.append(
                            f"{context_k}k@{chunk_size} {metric}: {ms:.1f}ms < min {min_ms:.1f}ms, update the baseline"
                        )
                    else:
                        result = "PASS"
                    rows.append(
                        f"{str(context_k) + 'k':>8} | {chunk_size:>6} | {metric:>6} | {ms:>10.1f} | {min_ms:>10.1f} | {max_ms:>10.1f} | {result}"
                    )

    logger.info("[perf_gate] wall and device times vs baseline ranges:\n" + "\n".join([RESULT_HEADER, *rows]))
    assert not failures, "prefill time outside baseline range: " + ", ".join(failures)

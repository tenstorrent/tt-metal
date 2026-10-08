# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLX prefill perf gate for Gemma4 disaggregated prefill.

For each chunk size, builds the model and runs a traced 256k prefill with ``_measure_traced``
(real text, 8x4 = CP8/TP4), then gates the cumulative wall time to reach each context length
in REPORT_CONTEXT_KS. All chunk sizes run in one process under ``_shared_device_weights``, so
each cached weight is written to the device once, as in
``text_demo_prefill.py::test_prefill_chunk_sweep_traced``. Wall time covers host staging,
trace dispatch and waiting for the device, but not compile or capture. ``_measure_traced``
logs it as the ``[traced_perf] context wall times`` table.
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
# Baseline wall times in ms to reach each context, keyed by (chunk_size, context_k), 1k = 1024 tokens.
WALL_MS_BASELINES = {
    (2048, 1): 66.1,
    (2048, 10): 338.9,
    (2048, 100): 3846.4,
    (2048, 256): 11993.1,
    (4096, 1): 84.6,
    (4096, 10): 261.6,
    (4096, 100): 2490.2,
    (4096, 256): 7985.6,
    (8192, 1): 124.6,
    (8192, 10): 258.6,
    (8192, 100): 2117.7,
    (8192, 256): 7016.7,
}
# Allowance for run-to-run variation, keyed by context_k. 1k and 10k take only
# 1-5 chunks, so host jitter is a larger share of their wall time.
WALL_MS_MARGINS = {1: 0.03, 10: 0.03, 100: 0.02, 256: 0.02}
WALL_MS_THRESHOLDS = {
    (chunk_size, context_k): ms * (1 + WALL_MS_MARGINS[context_k])
    for (chunk_size, context_k), ms in WALL_MS_BASELINES.items()
}
RESULT_HEADER = f"{'Context':>8} | {'Chunk':>6} | {'Wall (ms)':>10} | {'Max (ms)':>10} | Result"


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_perf_glx(mesh_device, reset_seeds):
    """Gate the 1k/10k/100k/256k wall times of a traced 256k prefill at each chunk size, writing weights once."""
    if unset := [(c, k) for c in CHUNK_SIZES for k in REPORT_CONTEXT_KS if (c, k) not in WALL_MS_THRESHOLDS]:
        pytest.fail(f"no wall-time threshold for (chunk_size, context_k) {unset} (context in units of 1k)")

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
            measured_ms = _measure_traced(
                mesh_device, mesh_config, model_args, model, hf_model_id, CONTEXT_LEN, chunk_size, token_source="text"
            )
            del model_args, model, _kv_cache
            gc.collect()
            # Cached programs hold op-allocated L1 semaphores that fragment L1 for the next model's circular buffers.
            mesh_device.clear_program_cache()

            for context_k in REPORT_CONTEXT_KS:
                wall_ms = measured_ms[context_k]
                max_ms = WALL_MS_THRESHOLDS[(chunk_size, context_k)]
                passed = wall_ms <= max_ms
                rows.append(
                    f"{str(context_k) + 'k':>8} | {chunk_size:>6} | {wall_ms:>10.1f} | {max_ms:>10.1f} | {'PASS' if passed else 'FAIL'}"
                )
                if not passed:
                    failures.append(f"{context_k}k@{chunk_size}: {wall_ms:.1f}ms > {max_ms:.1f}ms")

    logger.info("[perf_gate] wall times vs thresholds:\n" + "\n".join([RESULT_HEADER, *rows]))
    assert not failures, "prefill wall time over threshold: " + ", ".join(failures)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Incremental text prefill sweep to max context (65536) on one KV cache.

Fills tale tokens **once** on a single model, timing each segment between
checkpoints. This is the practical way to perf-sweep to ``max_seq_len`` without
re-building the model or re-prefilling from scratch at every ISL.

  - Checkpoints: 256 → 512 → … → 65536 (paged KV, per-token prefill)
  - After each checkpoint: one decode step at that position
  - Final row at 65536 = full KV budget exercised

**Verification (PCC, not perf):**

  - ``test_text_model_decode_tail_context_multistep_pcc`` — 65504 prefill + 32 decode
    steps at tail positions (logits PCC vs HF).
  - Run locally: ``pytest models/experimental/voxtraltts/tests/test_text_model.py::
    test_text_model_decode_tail_context_multistep_pcc -s``

Skipped in CI (multi-hour on bare metal).

Run perf::

    pytest models/experimental/voxtraltts/tests/perf/test_text_prefill_max_context_sweep_perf.py -q -s
"""

from __future__ import annotations

import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import (
    build_voxtral_text_page_table_tt,
    create_real_voxtral_text_model_or_skip,
    tale_continuation_tokens,
    tale_prompt_tokens,
)
from models.experimental.voxtraltts.tests.perf.isl_sweep_reporting import log_text_prefill_isl_sweep_summary
from models.experimental.voxtraltts.tests.perf.text_prefill_perf_utils import (
    decode_one_step,
    prefill_tokens,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

# Log-spaced checkpoints up to production KV (256 is first paged checkpoint).
_MAX_ISL = DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
_INCREMENTAL_CHECKPOINTS = (256, 512, 1024, 4096, 8192, 16384, 32768, _MAX_ISL)
_MODEL_NAME = "voxtral_text_4B"


@torch.no_grad()
@pytest.mark.timeout(0)
@pytest.mark.models_performance_bare_metal
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Max-context prefill sweep is multi-hour; bare metal only")
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_text_prefill_max_context_incremental_sweep_perf(device, reset_seeds):
    max_seq_len = _MAX_ISL
    full_tokens = tale_prompt_tokens(max_seq_len)
    page_table_tt = build_voxtral_text_page_table_tt(device, max_seq_len=max_seq_len)

    warmup_model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat16,
        use_paged_kv_cache=True,
    )
    prefill_tokens(
        warmup_model,
        tale_prompt_tokens(256),
        page_table_tt=page_table_tt,
        paged_kv=True,
    )
    ttnn.synchronize_device(device)
    del warmup_model
    gc.collect()

    model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat16,
        use_paged_kv_cache=True,
    )

    summary_rows: list[dict] = []
    prev_pos = 0

    try:
        for isl in _INCREMENTAL_CHECKPOINTS:
            if isl > max_seq_len:
                break
            segment_tokens = isl - prev_pos
            logger.info(f"Incremental prefill: filling positions [{prev_pos}, {isl}) ({segment_tokens} tokens)")

            t_prefill = time.time()
            prefill_tokens(
                model,
                full_tokens,
                page_table_tt=page_table_tt,
                paged_kv=True,
                start=prev_pos,
            )
            ttnn.synchronize_device(device)
            segment_s = time.time() - t_prefill

            decode_token = tale_continuation_tokens(isl, 1)[:, 0]
            t_decode = time.time()
            decode_one_step(model, decode_token, isl)
            ttnn.synchronize_device(device)
            decode_s = time.time() - t_decode

            segment_ms_per_token = (segment_s / max(segment_tokens, 1)) * 1000.0
            cumulative_ms_per_token = (segment_s / max(isl - prev_pos, 1)) * 1000.0

            row = {
                "isl": isl,
                "prefill_mode": "per_token_incr",
                "paged_kv": True,
                "segment_tokens": segment_tokens,
                "segment_prefill_s": segment_s,
                "segment_ms_per_token": segment_ms_per_token,
                "prefill_total_s": segment_s,
                "prefill_ms_per_token": cumulative_ms_per_token,
                "prefill_tokens_per_s": segment_tokens / segment_s if segment_s > 0 else 0.0,
                "decode_ms": decode_s * 1000.0,
            }
            summary_rows.append(row)

            logger.info(
                f"ISL={isl}: segment {segment_tokens} tok in {segment_s:.1f}s "
                f"({segment_ms_per_token:.2f} ms/tok), decode@pos={isl} {decode_s*1000:.1f} ms"
            )
            prev_pos = isl

            profiler = BenchmarkProfiler()
            profiler.start("run")
            profiler.end("run")
            step = "voxtral_text_prefill_max_ctx"
            profiler.start(step)
            profiler.end(step)
            benchmark_data = BenchmarkData()
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    benchmark_data.add_measurement(profiler, 0, step, key, float(value))
                elif isinstance(value, bool):
                    benchmark_data.add_measurement(profiler, 0, step, key, float(value))
            benchmark_data.save_partial_run_json(
                profiler,
                run_type="text_prefill_max_context_sweep",
                ml_model_name=_MODEL_NAME,
                batch_size=1,
                input_sequence_length=isl,
                output_sequence_length=1,
                config_params={"incremental": True, "paged_kv": True},
            )

        log_text_prefill_isl_sweep_summary(summary_rows)
        logger.info(
            f"Full {max_seq_len}-token KV fill complete. "
            "PCC verification at tail: pytest models/experimental/voxtraltts/tests/"
            "test_text_model.py::test_text_model_decode_tail_context_multistep_pcc -s"
        )
    finally:
        del model
        gc.collect()

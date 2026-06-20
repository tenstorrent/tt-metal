# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock text prefill ISL sweep on TTNN (Voxtral text backbone).

Sweeps input sequence length (ISL) with *A Tale of Two Cities* prompts, production
``max_seq_len`` KV (65536), and reports per-token prefill + one decode step.

For a **single run** filling KV incrementally to 65536, use
``test_text_prefill_max_context_sweep_perf.py`` instead (much faster than
re-prefilling from scratch at each checkpoint).

Run::

    pytest models/experimental/voxtraltts/tests/perf/test_text_prefill_isl_sweep_perf.py -q -s
"""

from __future__ import annotations

import gc
import os
import time

import pytest
import torch

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
    can_bulk_prefill,
    decode_one_step,
    prefill_tokens,
    use_paged_kv_for_isl,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.perf_utils import prep_perf_report

_ISL_SWEEP = (128, 256, 512, 1024, 4096, 8192, 16384, 32768)
_WARMUP_ISL = 128
_MODEL_NAME = "voxtral_text_4B"


@torch.no_grad()
@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("expected_compile_time, expected_inference_time", [(120.0, 0.01)])
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_text_prefill_isl_sweep_perf(
    device,
    reset_seeds,
    expected_compile_time,
    expected_inference_time,
):
    max_seq_len = DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
    isl_values = list(_ISL_SWEEP)
    if os.environ.get("CI") == "true":
        isl_values = [isl for isl in isl_values if isl <= 1024]

    summary_rows: list[dict] = []

    for isl in isl_values:
        use_paged = use_paged_kv_for_isl(isl)
        prompt_tokens = tale_prompt_tokens(isl)
        decode_token = tale_continuation_tokens(isl, 1)[:, 0:1]
        prefill_mode = "bulk" if can_bulk_prefill(isl, paged_kv=use_paged) else "per_token"

        warmup_model = create_real_voxtral_text_model_or_skip(
            device,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat16,
            use_paged_kv_cache=use_paged,
        )
        page_table_tt = build_voxtral_text_page_table_tt(device, max_seq_len=max_seq_len) if use_paged else None
        prefill_tokens(
            warmup_model,
            tale_prompt_tokens(_WARMUP_ISL),
            page_table_tt=page_table_tt,
            paged_kv=use_paged,
        )
        ttnn.synchronize_device(device)
        del warmup_model
        gc.collect()

        timed_model = create_real_voxtral_text_model_or_skip(
            device,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat16,
            use_paged_kv_cache=use_paged,
        )
        if use_paged and page_table_tt is None:
            page_table_tt = build_voxtral_text_page_table_tt(device, max_seq_len=max_seq_len)

        t_prefill = time.time()
        prefill_tokens(timed_model, prompt_tokens, page_table_tt=page_table_tt, paged_kv=use_paged)
        ttnn.synchronize_device(device)
        prefill_s = time.time() - t_prefill

        t_decode = time.time()
        decode_one_step(timed_model, decode_token[:, 0], isl)
        ttnn.synchronize_device(device)
        decode_s = time.time() - t_decode

        row = {
            "isl": isl,
            "prefill_mode": prefill_mode,
            "paged_kv": use_paged,
            "prefill_total_s": prefill_s,
            "prefill_ms_per_token": (prefill_s / max(isl, 1)) * 1000.0,
            "prefill_tokens_per_s": isl / prefill_s if prefill_s > 0 else 0.0,
            "decode_ms": decode_s * 1000.0,
        }
        summary_rows.append(row)

        prep_perf_report(
            model_name=_MODEL_NAME,
            batch_size=1,
            inference_and_compile_time=prefill_s,
            inference_time=prefill_s / max(isl, 1),
            expected_compile_time=expected_compile_time,
            expected_inference_time=expected_inference_time,
            comments=f"prefill_isl{isl}_{prefill_mode}",
        )

        profiler = BenchmarkProfiler()
        profiler.start("run")
        profiler.end("run")
        step = "voxtral_text_prefill_isl"
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
            run_type="text_prefill_isl_sweep",
            ml_model_name=_MODEL_NAME,
            batch_size=1,
            input_sequence_length=isl,
            output_sequence_length=1,
            config_params={"prefill_mode": prefill_mode, "paged_kv": use_paged},
        )

        del timed_model
        gc.collect()

    log_text_prefill_isl_sweep_summary(summary_rows)

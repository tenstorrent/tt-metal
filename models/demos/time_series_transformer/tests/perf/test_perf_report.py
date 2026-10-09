# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Emit the repository-standard performance report for this model.

``prep_perf_report`` writes the CSV that the shared tooling consumes, with the same columns
every other demo produces. The bounty asks for the standard header rather than a bespoke table,
so this exists alongside PERF.md rather than replacing it.

    pytest models/demos/time_series_transformer/tests/perf/test_perf_report.py -q -s
"""

import time
from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.time_series_transformer.reference.torch_reference import make_inputs
from models.demos.time_series_transformer.tests.perf.perf_common import TARGET_LATENCY_MS, build_model, run_benchmark
from models.perf.perf_utils import prep_perf_report

MODEL_NAME = "time_series_transformer"
BATCH = 1

# Generous ceilings: this report is for tracking, and the hard gates live in test_perf.py.
EXPECTED_COMPILE_SECONDS = 120.0
EXPECTED_INFERENCE_SECONDS = TARGET_LATENCY_MS / 1000.0


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("profile", ["accuracy", "performance"])
def test_perf_report(device, config, hf_state, hf_model, profile):
    overrides = {} if profile == "accuracy" else {"dtype": "bfloat16", "use_sdpa": True, "use_exact_softmax": False}
    inputs = make_inputs(hf_model.config, batch=BATCH)

    model = build_model(replace(config, use_trace=True, **overrides), hf_state, device=device)
    try:
        # First call carries kernel compilation and trace capture; the report wants both.
        start = time.perf_counter()
        model.generate(num_parallel_samples=1, mode="mean", **inputs)
        inference_and_compile_time = time.perf_counter() - start

        result = run_benchmark(model, inputs, batch=BATCH, num_samples=1, mode="mean")
        inference_time = result.latency_ms / 1000.0
    finally:
        model.release_traces()
        ttnn.synchronize_device(device)

    prep_perf_report(
        model_name=f"{MODEL_NAME}_{profile}",
        batch_size=BATCH,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=EXPECTED_COMPILE_SECONDS,
        expected_inference_time=EXPECTED_INFERENCE_SECONDS,
        comments=profile,
    )

    logger.info(
        f"{profile}: compile {inference_and_compile_time - inference_time:.2f} s, "
        f"inference {inference_time * 1000:.2f} ms, {result.throughput:.1f} seq/s"
    )
    assert torch.isfinite(torch.tensor(inference_time))
    assert (
        inference_time < EXPECTED_INFERENCE_SECONDS
    ), f"{profile} inference {inference_time * 1000:.2f} ms exceeds {EXPECTED_INFERENCE_SECONDS * 1000:.0f} ms"

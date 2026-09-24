# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import statistics
import time

import pytest
import torch

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    A10G_SERIES_PER_S,
    A10G_WALL_S,
    BATCH,
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    NUM_QUANTILES,
    PREDICTION_LENGTH,
    _load_reference,
)


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 100_000_000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_paper_forward_trace_perf(mesh_device):
    ttnn = pytest.importorskip("ttnn")

    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    reference, weight_source = _load_reference()
    model = TtChronos.from_torch_model(mesh_device, reference)
    torch.manual_seed(0)
    context = torch.randn(BATCH, CONTEXT)

    preprocess_start = time.perf_counter()
    prepared = model.prepare_inputs(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
    preprocess_s = time.perf_counter() - preprocess_start

    runner = TtChronosTraceRunner(model, prepared)
    try:
        runner.capture()
        for _ in range(3):
            runner.execute(blocking=True, readback=False)

        trace_times = []
        for index in range(20):
            start = time.perf_counter()
            runner.execute(blocking=True, readback=False)
            duration = time.perf_counter() - start
            trace_times.append(duration)
            print(f"[TRACE PERF] replay {index + 1:2d}/20 {duration:.6f}s")

        e2e_times = []
        result = None
        for index in range(20):
            start = time.perf_counter()
            prepared_iteration = model.prepare_inputs(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
            result = runner.execute_pipelined(prepared_iteration, readback=True)
            duration = time.perf_counter() - start
            e2e_times.append(duration)
            print(f"[TRACE PERF] e2e    {index + 1:2d}/20 {duration:.6f}s")
    finally:
        runner.release()

    expected_shape = (BATCH, NUM_QUANTILES, PREDICTION_LENGTH)
    assert result is not None
    assert result.quantile_preds.shape == expected_shape
    replay_median = statistics.median(trace_times)
    replay_p95 = _percentile(trace_times, 0.95)
    e2e_median = statistics.median(e2e_times)
    e2e_p95 = _percentile(e2e_times, 0.95)
    print(
        "\n[TRACE PERF] Chronos paper shape"
        f"\n  weights:              {weight_source}"
        f"\n  host_preprocess_s:    {preprocess_s:.6f}"
        f"\n  replay_median_s:      {replay_median:.6f}"
        f"\n  replay_p95_s:         {replay_p95:.6f}"
        f"\n  replay_series_per_s:  {BATCH / replay_median:.2f}"
        f"\n  e2e_median_s:         {e2e_median:.6f}"
        f"\n  e2e_p95_s:            {e2e_p95:.6f}"
        f"\n  e2e_series_per_s:     {BATCH / e2e_median:.2f}"
        f"\n  a10g_wall_s:          {A10G_WALL_S:.3f}"
        f"\n  a10g_series_per_s:    {A10G_SERIES_PER_S:.0f}"
    )

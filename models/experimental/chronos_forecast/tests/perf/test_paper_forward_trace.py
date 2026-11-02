# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import statistics
import time

import pytest
import torch

from models.experimental.chronos_forecast.tests.mesh_params import MULTI_CHIP_MESHES
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

REPLAY_ITERS = 20
STAGE_ITERS = 10

# Scaling baselines: one p150a, performance precision, L1-resident; keyed by (group_size, batch).
SINGLE_CHIP_REPLAY_S = {
    (None, BATCH): 0.1535,
    (None, BATCH // 4): 0.0349,
    (None, BATCH // 8): 0.0175,
    (None, BATCH // 16): 0.00878,
    (None, BATCH // 32): 0.00609,
    (4, BATCH): 0.2691,
}
SINGLE_CHIP_E2E_S = {
    (None, BATCH): 0.1992,
    (None, BATCH // 4): 0.0527,
    (None, BATCH // 8): 0.0267,
    (None, BATCH // 16): 0.0138,
    (None, BATCH // 32): 0.00983,
    (4, BATCH): 0.3300,
}

TRACE_DEVICE_PARAMS = [{"trace_region_size": 200_000_000, "num_command_queues": 2}]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _run_trace_perf(mesh_device, precision_name, l1_resident, group_size, batch) -> dict[str, float]:
    ttnn = pytest.importorskip("ttnn")

    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision
    from models.experimental.chronos_forecast.tt.trace_runner import TtChronosTraceRunner

    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    l1_chunk_tokens = precision.l1_chunk_tokens() if l1_resident else None
    reference, weight_source = _load_reference()
    model = TtChronos.from_torch_model(mesh_device, reference, precision, l1_chunk_tokens=l1_chunk_tokens)
    torch.manual_seed(0)
    context = torch.randn(batch, CONTEXT)
    group_ids = None if group_size is None else torch.arange(batch) // group_size

    def prepare():
        return model.prepare_inputs(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES)

    prepared, preprocess_s = _timed(prepare)

    runner = TtChronosTraceRunner(model, prepared)
    try:
        runner.capture()
        for _ in range(3):
            runner.execute(blocking=True, readback=False)

        trace_times = []
        for index in range(REPLAY_ITERS):
            _, duration = _timed(lambda: runner.execute(blocking=True, readback=False))
            trace_times.append(duration)
            print(f"[TRACE PERF] replay {index + 1:2d}/{REPLAY_ITERS} {duration:.6f}s")

        # Serial stages, so each host step is timed on its own.
        stages = {"prepare": [], "upload": [], "replay": [], "readback": []}
        for _ in range(STAGE_ITERS):
            prepared_iteration, duration = _timed(prepare)
            stages["prepare"].append(duration)

            def upload():
                runner.update_inputs(prepared_iteration)
                ttnn.synchronize_device(mesh_device)

            stages["upload"].append(_timed(upload)[1])
            stages["replay"].append(_timed(lambda: runner.execute(blocking=True, readback=False))[1])
            stages["readback"].append(_timed(runner.read_output)[1])

        e2e_times = []
        result = None
        for index in range(REPLAY_ITERS):
            start = time.perf_counter()
            prepared_iteration = prepare()
            result = runner.execute_pipelined(prepared_iteration, readback=True)
            duration = time.perf_counter() - start
            e2e_times.append(duration)
            print(f"[TRACE PERF] e2e    {index + 1:2d}/{REPLAY_ITERS} {duration:.6f}s")

        # Back-to-back batches with host prepare overlapping the previous replay.
        start = time.perf_counter()
        for result in runner.stream(range(REPLAY_ITERS), lambda _: prepare()):
            pass
        stream_s = (time.perf_counter() - start) / REPLAY_ITERS
    finally:
        runner.release()

    expected_shape = (batch, NUM_QUANTILES, PREDICTION_LENGTH)
    assert result is not None
    assert result.quantile_preds.shape == expected_shape
    replay_median = statistics.median(trace_times)
    e2e_median = statistics.median(e2e_times)
    grid = mesh_device.compute_with_storage_grid_size()
    stage_medians = {name: statistics.median(times) for name, times in stages.items()}
    stage_lines = "".join(f"\n  stage_{name + '_s:':17s} {value:.6f}" for name, value in stage_medians.items())
    print(
        "\n[TRACE PERF] Chronos paper shape"
        f"\n  weights:              {weight_source}"
        f"\n  chips:                {model.num_devices} (worker grid {grid.x}x{grid.y})"
        f"\n  precision:            {precision_name}"
        f"\n  batch:                {batch} ({batch // model.num_devices} per chip)"
        f"\n  l1_chunk_tokens:      {l1_chunk_tokens}"
        f"\n  group_size:           {group_size or 'unique'} (block {prepared.group_block})"
        f"\n  host_preprocess_s:    {preprocess_s:.6f}"
        f"\n  replay_median_s:      {replay_median:.6f}"
        f"\n  replay_p95_s:         {_percentile(trace_times, 0.95):.6f}"
        f"\n  replay_series_per_s:  {batch / replay_median:.2f}"
        f"{stage_lines}"
        f"\n  e2e_median_s:         {e2e_median:.6f}"
        f"\n  e2e_p95_s:            {_percentile(e2e_times, 0.95):.6f}"
        f"\n  e2e_series_per_s:     {batch / e2e_median:.2f}"
        f"\n  stream_per_batch_s:   {stream_s:.6f}"
        f"\n  stream_series_per_s:  {batch / stream_s:.2f}"
        f"\n  a10g_wall_s:          {A10G_WALL_S:.3f}"
        f"\n  a10g_series_per_s:    {A10G_SERIES_PER_S:.0f}"
    )
    return {"replay": replay_median, "e2e": e2e_median}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "precision_name, l1_resident, group_size, batch",
    [
        pytest.param("default", False, None, BATCH, id="default_dram"),
        pytest.param("default", True, None, BATCH, id="default_l1"),
        pytest.param("performance", True, None, BATCH, id="performance_l1"),
        pytest.param("default", False, 4, BATCH, id="default_dram_groups_of_4"),
        pytest.param("performance", True, 4, BATCH, id="performance_l1_groups_of_4"),
        # Single-chip runs at the per-chip batch of 4 / 8 / 16 / 32-chip data parallel.
        pytest.param("performance", True, None, BATCH // 4, id="performance_l1_b256"),
        pytest.param("performance", True, None, BATCH // 8, id="performance_l1_b128"),
        pytest.param("performance", True, None, BATCH // 16, id="performance_l1_b64"),
        pytest.param("performance", True, None, BATCH // 32, id="performance_l1_b32"),
    ],
)
@pytest.mark.parametrize("device_params", TRACE_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_paper_forward_trace_perf(mesh_device, precision_name, l1_resident, group_size, batch):
    _run_trace_perf(mesh_device, precision_name, l1_resident, group_size, batch)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("group_size", [pytest.param(None, id="unique"), pytest.param(4, id="groups_of_4")])
@pytest.mark.parametrize("device_params", TRACE_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MULTI_CHIP_MESHES, indirect=True)
def test_paper_forward_trace_data_parallel(mesh_device, group_size):
    """Full paper batch split across the mesh; scaling is against one p150a.

    Speedup is against one chip running the full batch; "same per-chip batch" compares
    with one chip running only this chip's share (1.0 means no data-parallel overhead).
    """
    stats = _run_trace_perf(mesh_device, "performance", True, group_size, BATCH)
    chips = mesh_device.get_num_devices()
    for name, baseline in (("replay", SINGLE_CHIP_REPLAY_S), ("e2e", SINGLE_CHIP_E2E_S)):
        speedup = baseline[(group_size, BATCH)] / stats[name]
        line = f"  {name + '_speedup:':21s} {speedup:.2f}x (efficiency {speedup / chips:.1%})"
        per_chip = baseline.get((group_size, BATCH // chips))
        if per_chip is not None:
            line += f", vs one chip at the same per-chip batch {per_chip / stats[name]:.1%}"
        print(line)

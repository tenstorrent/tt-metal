# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device profile of the device-resident Chronos-2 forward at the paper shape.

Run under Tracy and summarize the signposted region:

    python -m tracy -v -r -p -m pytest \
        "models/experimental/chronos_forecast/tests/perf/test_device_forward_profile.py" -k default_dram
    tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
        --start-signpost chronos_device_forward_start --end-signpost chronos_device_forward_stop

The ``*_l1`` variants dispatch ~3.5k ops per forward, which overflows the
device profiler's marker buffers at batch 1024; run them without Tracy for
the eager timing, or profile a smaller batch (per-chunk costs are identical).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    BATCH,
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    _load_reference,
)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "precision_name, l1_resident",
    [
        pytest.param("default", False, id="default_dram"),
        pytest.param("default", True, id="default_l1"),
        pytest.param("performance", False, id="performance_dram"),
        pytest.param("performance", True, id="performance_l1"),
    ],
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_device_forward_profile(mesh_device, precision_name, l1_resident):
    ttnn = pytest.importorskip("ttnn")

    from models.experimental.chronos_forecast.tt.model import TtChronos
    from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision
    from tracy import signpost

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")
    mesh_device.enable_program_cache()

    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    reference, weight_source = _load_reference()
    l1_chunk_tokens = precision.l1_chunk_tokens() if l1_resident else None
    model = TtChronos.from_torch_model(mesh_device, reference, precision, l1_chunk_tokens=l1_chunk_tokens)
    torch.manual_seed(0)
    context = torch.randn(BATCH, CONTEXT)
    prepared = model.prepare_inputs(context=context, num_output_patches=NUM_OUTPUT_PATCHES)
    inputs = model.upload_inputs(prepared)

    try:
        for _ in range(2):
            output = model.forward_device(inputs)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        device_profiler_enabled = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if device_profiler_enabled:
            ttnn.ReadDeviceProfiler(mesh_device)

        signpost("chronos_device_forward_start")
        start = time.perf_counter()
        output = model.forward_device(inputs)
        ttnn.synchronize_device(mesh_device)
        device_s = time.perf_counter() - start
        signpost("chronos_device_forward_stop")
        if device_profiler_enabled:
            ttnn.ReadDeviceProfiler(mesh_device)
        ttnn.deallocate(output)
    finally:
        model.deallocate_inputs(inputs)

    print(
        "\n[DEVICE PROFILE] Chronos paper shape, device-resident forward"
        f"\n  weights:        {weight_source}"
        f"\n  precision:      {precision_name}"
        f"\n  l1_chunk_tokens: {l1_chunk_tokens}"
        f"\n  unique_groups:  {prepared.unique_groups}"
        f"\n  eager_device_s: {device_s:.6f}"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 perf: end-to-end latency of the device-resident Ideogram4Pipeline.
# Warms the program cache, then times encode / denoise / decode via the standard
# tt_dit event -> BenchmarkProfiler path, emits BenchmarkData (in CI), and asserts
# each section stays under a conservative ceiling so a gross regression fails the
# test. Single-block device perf comes from the PCC test (test_transformer_ideogram4.py).
# =============================================================================

import os

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from ....pipelines.events import profiler_event_callback
from ....pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ....utils.test import line_params, ring_params

# The pipeline needs the gated fp8 checkpoint (create_pipeline raises ValueError without it);
# skip (don't error) when it isn't configured, matching the component tests.
_NEEDS_WEIGHTS = pytest.mark.skipif(
    not os.environ.get("IDEOGRAM4_WEIGHTS"), reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)"
)

PROMPT = "a watercolor painting of a red panda reading a book under a cherry tree, soft morning light"

_LINE_PIPE = {**line_params, "l1_small_size": 32768, "trace_region_size": 60000000}
_RING_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": 60000000}

# Conservative upper-bound targets (seconds), program-cache warm, 512px V4_TURBO_12. Generous
# placeholders that catch a gross (>~2x) regression without flaking; tighten once stable CI
# baselines exist. Keyed by mesh shape.
_PERF_TARGETS = {
    (2, 4): {"encoder": 30.0, "denoising": 90.0, "vae": 30.0, "total": 130.0},  # BH LoudBox 2x4 SP4xTP2
    (4, 8): {"encoder": 30.0, "denoising": 60.0, "vae": 30.0, "total": 100.0},  # BH Galaxy TP4xSP8
}


# Parallel config (tp/sp/num_links/topology) is DISCOVERED from the mesh shape via
# Ideogram4Pipeline._PRESETS, so the perf test only picks a mesh shape + its device_params.
@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param((2, 4), _LINE_PIPE, id="sp4tp2"),  # BH loudbox 2x4: SP4 x TP2
        pytest.param((4, 8), _RING_PIPE, id="bh_galaxy_sp8tp4"),  # BH Galaxy: TP4 x SP8 (Ring)
    ],
    indirect=True,
)
@pytest.mark.parametrize(("height", "width", "preset"), [(512, 512, "V4_TURBO_12")], ids=["512px_turbo12"])
def test_latency(*, mesh_device, is_ci_env, height, width, preset) -> None:
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=height, width=width)

    pipe(prompts=[PROMPT], preset=preset, seed=1)  # warmup (compile + cache)
    profiler = BenchmarkProfiler()
    pipe(prompts=[PROMPT], preset=preset, seed=2, on_event=profiler_event_callback(profiler, 0))  # timed (warm)

    steps = ("encoder", "denoising", "vae", "total")
    measurements = {s: profiler.get_duration(s) for s in steps}
    logger.info(f"LATENCY {height}px {preset}: " + " | ".join(f"{s} {measurements[s]:.2f}s" for s in steps))

    targets = _PERF_TARGETS[tuple(mesh_device.shape)]
    if is_ci_env:
        benchmark_data = BenchmarkData()
        for s in steps:
            benchmark_data.add_measurement(
                profiler=profiler, iteration=0, step_name=s, name=s, value=measurements[s], target=targets[s]
            )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="BH_LB" if tuple(mesh_device.shape) == (2, 4) else "BH_GLX",
            ml_model_name="Ideogram4",
            batch_size=1,
            config_params={"width": width, "height": height, "preset": preset},
        )

    regressions = [f"{s}: {measurements[s]:.2f}s > {targets[s]:.1f}s" for s in steps if measurements[s] > targets[s]]
    assert not regressions, "perf regression: " + "; ".join(regressions)

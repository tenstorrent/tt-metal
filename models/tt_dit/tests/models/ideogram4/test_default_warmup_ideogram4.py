# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4 as a server uses it: ``create_pipeline`` with its defaults and no explicit ``warmup``.

On a BH Galaxy the default warms the four 2K serving sizes (``SERVING_SIZES``) inside
``create_pipeline``, on a mesh opened with the published ``required_trace_region_size``. Then one
traced request per size must produce an image without compiling or capturing anything, and a size
that was not warmed must raise. Reports the load time, how much of it the default warmup takes, and
the per-request times. ``test_trace_region_too_small`` checks that a mesh with a smaller trace
region fails at once, before the weights load, with a clear error.

Env: IDEOGRAM4_WEIGHTS (fp8 checkpoint dir or HF repo id), IDEOGRAM_SERVE_OUT (report dir).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

import ttnn

from ....pipelines.ideogram4 import pipeline_ideogram4
from ....pipelines.ideogram4.pipeline_ideogram4 import (
    SERVING_SIZES,
    Ideogram4Pipeline,
    Ideogram4PipelineConfig,
    required_trace_region_size,
)
from ....utils.test import ring_params
from ....utils.tracing import Tracer

_NEEDS_WEIGHTS = pytest.mark.skipif(
    not os.environ.get("IDEOGRAM4_WEIGHTS"), reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)"
)
# The trace region a server must open the mesh with for the default warmup, as published.
_SERVE_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": required_trace_region_size((4, 8))}
# What test_pipeline_class opens the mesh with: too small for the default warmup's kept traces.
_SMALL_TRACE_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": 60000000}

PROMPT = "A red apple on a wooden table."
SEED = 123


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _SERVE_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_default_warmup(*, mesh_device, expect_error, monkeypatch) -> None:
    out = Path(os.environ.get("IDEOGRAM_SERVE_OUT", "ideogram4_default_warmup"))
    (out / "images").mkdir(parents=True, exist_ok=True)

    # Time the warmup that create_pipeline runs, without changing what it does.
    warmup_s = []
    orig_warmup = Ideogram4Pipeline.warmup

    def timed_warmup(self, *args, **kwargs):
        start = time.perf_counter()
        orig_warmup(self, *args, **kwargs)
        warmup_s.append(time.perf_counter() - start)

    monkeypatch.setattr(Ideogram4Pipeline, "warmup", timed_warmup)

    assert Ideogram4PipelineConfig.default(mesh_shape=mesh_device.shape).resolutions == SERVING_SIZES
    trace_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
    trace_reserved = trace_view.total_bytes_per_bank * trace_view.num_banks
    assert trace_reserved >= required_trace_region_size(mesh_device.shape) == 128_000_000

    start = time.perf_counter()
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device)
    load_s = time.perf_counter() - start
    dev = pipe.mesh_device  # sync the pipeline's submesh, never the parent mesh

    assert pipe.resolutions == pipe.warmed_sizes == SERVING_SIZES
    assert pipe.num_trace_captures == len(SERVING_SIZES)
    assert len(warmup_s) == 1

    report = {
        "required_trace_region_size": required_trace_region_size(mesh_device.shape),
        "trace_region_reserved": trace_reserved,
        "create_pipeline_s": round(load_s, 1),
        "default_warmup_s": round(warmup_s[0], 1),
        "load_without_warmup_s": round(load_s - warmup_s[0], 1),
        "trace_region_used_per_bank": ttnn.get_memory_view(dev, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank,
        "dram_free_per_bank_after_warmup": ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_free_per_bank,
        "dram_bank_size": ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_per_bank,
        "requests": [],
    }
    logger.info(f"DEFAULT-WARMUP load {json.dumps({k: v for k, v in report.items() if k != 'requests'})}")

    programs = dev.num_program_cache_entries()
    captures = pipe.num_trace_captures
    traces_live = Tracer._traces_live[dev.id()]
    failures = []

    for width, height in SERVING_SIZES:
        start = time.perf_counter()
        image = pipe(prompts=[PROMPT], seed=SEED, width=width, height=height)[0]  # traced by default
        ttnn.synchronize_device(dev)
        row = {
            "size": f"{width}x{height}",
            "wall_s": round(time.perf_counter() - start, 3),
            "image_size": list(image.size),
            "image_std": round(float(np.asarray(image, dtype=np.float32).std()), 1),
            "new_programs": dev.num_program_cache_entries() - programs,
            "new_captures": pipe.num_trace_captures - captures,
            "traces_live_delta": Tracer._traces_live[dev.id()] - traces_live,
        }
        logger.info(f"DEFAULT-WARMUP {json.dumps(row)}")
        report["requests"].append(row)
        image.save(out / "images" / f"{width}x{height}.png")
        if tuple(image.size) != (width, height):
            failures.append(f"{row['size']}: image size {row['image_size']}")
        if row["image_std"] < 10:
            failures.append(f"{row['size']}: flat image")
        for key in ("new_programs", "new_captures", "traces_live_delta"):
            if row[key]:
                failures.append(f"{row['size']}: {key} = {row[key]}")

    # A call that names no size runs at the first resolution, from its kept trace.
    image = pipe(prompts=[PROMPT], seed=SEED)[0]
    ttnn.synchronize_device(dev)
    report["default_size"] = list(image.size)
    if tuple(image.size) != SERVING_SIZES[0]:
        failures.append(f"default call size {image.size} != {SERVING_SIZES[0]}")
    if dev.num_program_cache_entries() != programs or pipe.num_trace_captures != captures:
        failures.append("default call compiled or captured")

    # A size that was not warmed raises before touching the device.
    with expect_error(ValueError, "was not warmed up"):
        pipe(prompts=[PROMPT], width=1024, height=1024)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    assert not failures, "\n".join(failures)


class _WeightsLoad(Exception):
    pass


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _SMALL_TRACE_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_trace_region_too_small(*, mesh_device, expect_error, monkeypatch) -> None:
    def no_load(*args, **kwargs):
        raise _WeightsLoad

    # The check runs first in __init__, before the submesh and the weights: stop at the submesh
    # instead of loading them.
    monkeypatch.setattr(pipeline_ideogram4, "create_submeshes", no_load)
    with expect_error(ValueError, "needs 128 MB; open the mesh with trace_region_size >= 128000000"):
        Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device)
    # Without the default warmup, the same mesh gets past the check to the load.
    with expect_error(_WeightsLoad, ""):
        Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, run_warmup=False)

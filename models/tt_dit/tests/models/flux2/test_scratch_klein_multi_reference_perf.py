# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: Klein 4B performance with 0 to N reference images at one resolution, with the GPU benchmark's
inputs (gpu-benchmark/inputs/flux2-klein.json) and runs, as the scratch commit b4ccd2a5 does for 0 and 1.

Run once per resolution (KLEIN_SIZE, 1024 or 2048), in two steps:

- KLEIN_FIND_LIMIT=1 runs 0, 1, 2, ... reference images untraced up to KLEIN_MAX_REFERENCES (default 10),
  stops at the first failure and writes the highest count that worked to KLEIN_LIMIT_FILE.
- Then, with KLEIN_MAX_REFERENCES set to that count, every count runs untraced once (as create_pipeline's
  warmup does for 0 and 1 image), and then, per count, NUM_UNMEASURED_RUNS traced runs, the first of
  which captures the trace, followed by NUM_PERF_RUNS timed traced runs.

A failure that leaves the device out of memory spoils the rest of the process, hence the two steps.
"""

import os
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.events import profiler_event_callback

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .device_params import REQ_EXACT, line_params_flux2

NUM_UNMEASURED_RUNS = 2
NUM_PERF_RUNS = 5
_SEED = 123
_GUIDANCE = 1.0
_PROMPT = "A cat holding a sign that says hello world"
_EDIT_PROMPT = (
    "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park "
    "square."
)
_IMAGE_DIR = Path(__file__).resolve().parent.parent / "gpu_benchmark_inputs"
_IMAGES = [_IMAGE_DIR / name for name in ("farmers-market.jpg", "mount-hood.jpg", "city-street.jpg")]
_SIZE = int(os.environ.get("KLEIN_SIZE", "1024"))
_MAX_COUNT = int(os.environ.get("KLEIN_MAX_REFERENCES", "10"))
_FIND_LIMIT = os.environ.get("KLEIN_FIND_LIMIT") == "1"
_LIMIT_FILE = os.environ.get("KLEIN_LIMIT_FILE", f"/tmp/klein_limit_{_SIZE}")
_TRACE_REGION_SIZE = int(os.environ.get("KLEIN_TRACE_REGION_SIZE", "200000000"))
_DEVICE_PARAMS = {**line_params_flux2, **REQ_EXACT, "trace_region_size": _TRACE_REGION_SIZE}


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 2), _DEVICE_PARAMS], [(2, 4), _DEVICE_PARAMS], [(4, 8), _DEVICE_PARAMS]],
    ids=["2x2", "2x4", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_multi_reference_performance(
    *, mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    if not is_blackhole():
        pytest.skip("Blackhole only")
    if _MAX_COUNT < 0:
        pytest.skip(f"no reference count worked at {_SIZE}x{_SIZE}")

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        width=_SIZE,
        height=_SIZE,
        resolutions=[(_SIZE, _SIZE)],
        trace_warmup=False,
    )
    references = [Image.open(path).convert("RGB") for path in _IMAGES]
    profiler = BenchmarkProfiler()

    def run(count: int, *, traced: bool, iteration: int | None = None) -> np.ndarray:
        images = pipeline(
            prompts=[_EDIT_PROMPT if count else _PROMPT],
            images=[references[i % len(references)] for i in range(count)] or None,
            num_inference_steps=4,
            guidance_scale=_GUIDANCE,
            seed=_SEED,
            width=_SIZE,
            height=_SIZE,
            traced=traced,
            on_event=None if iteration is None else profiler_event_callback(profiler, iteration),
        )
        ttnn.synchronize_device(mesh_device)
        return np.asarray(images[0].convert("RGB")).astype(np.float64)

    def first_line(error: Exception) -> str:
        return (str(error).splitlines() or [repr(error)])[0][:300]

    if _FIND_LIMIT:
        limit = -1
        for count in range(_MAX_COUNT + 1):
            try:
                run(count, traced=False)
            except Exception as error:  # noqa: BLE001
                logger.error(f"LIMIT {_SIZE}x{_SIZE}, {count} input images failed untraced: {first_line(error)}")
                break
            limit = count
            logger.info(f"LIMIT {_SIZE}x{_SIZE}, {count} input images worked untraced")
        with open(_LIMIT_FILE, "w") as f:
            f.write(str(limit))
        logger.info(f"LIMIT {_SIZE}x{_SIZE} on {mesh_shape}: up to {limit} input images")
        return

    untraced = {}
    for count in range(_MAX_COUNT + 1):
        with profiler("untraced", iteration=10_000 + count):
            untraced[count] = run(count, traced=False)

    lines = []
    failures = []
    for count in range(_MAX_COUNT + 1):
        label = f"{_SIZE}x{_SIZE}, {count} input images"
        try:
            for _ in range(NUM_UNMEASURED_RUNS):  # the first captures this count's traces
                run(count, traced=True)
            its = []
            for i in range(NUM_PERF_RUNS):
                it = count * NUM_PERF_RUNS + i
                with profiler("run", iteration=it):
                    image = run(count, traced=True, iteration=it)
                its.append(it)
                logger.info(f"{label} run {i + 1}: {profiler.get_duration('run', it):.3f}s")
        except Exception as error:  # noqa: BLE001
            failures.append(f"{label}: {first_line(error)}")
            logger.error(failures[-1])
            break

        mse = np.mean((image - untraced[count]) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else np.inf
        totals = [profiler.get_duration("run", it) for it in its]
        lines.append(f"Case: {label}  (traced against untraced {psnr:.1f} dB)")
        lines.append(
            f"  {'Total':14} | Mean {statistics.mean(totals):8.4f}s | Median {statistics.median(totals):8.4f}s | "
            f"Min {min(totals):8.4f}s | Max {max(totals):8.4f}s | Runs {[round(t, 4) for t in totals]}"
        )
        for name, step in (
            ("Encoding", "encoder"),
            ("Image encoder", "image_encoder"),
            ("Denoising", "denoising"),
            ("VAE", "vae"),
        ):
            times = [profiler.get_duration(step, it) for it in its if profiler.contains_step(step, it)]
            if times:
                lines.append(
                    f"  {name:14} | Mean {statistics.mean(times):8.4f}s | Min {min(times):8.4f}s | "
                    f"Max {max(times):8.4f}s"
                )
        lines.append(f"  {'Untraced':14} | {profiler.get_duration('untraced', 10_000 + count):8.4f}s (first run)")

    print("\n" + "=" * 100)
    print("FLUX2 KLEIN MULTI-REFERENCE PERFORMANCE RESULTS")
    print("=" * 100)
    print(f"Mesh Shape: {mesh_shape}, size: {_SIZE}x{_SIZE}, steps: 4, timed runs per case: {NUM_PERF_RUNS}")
    print("\n".join(lines))
    for failure in failures:
        print(f"FAILED {failure}")
    print("=" * 100)
    assert not failures, "\n".join(failures)

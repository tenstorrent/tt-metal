# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.fibo.pipeline_fibo import FiboPipeline

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 30
NUM_MEASURED_RUNS = 8
CFG_SCALE = 5.0

PROMPT = "A red bicycle leaning against a stone wall at sunset."

_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "l1_small_size": 32_768,
    "trace_region_size": 256_000_000,
}


# Use -s to print the table.
#
#     pytest models/tt_dit/tests/models/fibo/test_performance_fibo.py -k 4x8 -v -s --timeout=5400
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 2), id="2x2", marks=run_for_blackhole()),
        pytest.param((2, 4), id="2x4", marks=run_for_wormhole_b0()),
        pytest.param((4, 8), id="4x8", marks=run_for_blackhole()),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
def test_fibo_pipeline_perf_breakdown(
    *,
    mesh_device: ttnn.MeshDevice,
    model_location_generator,
) -> None:
    """Time each stage of a text-prompt generation, VLM included, then log medians over the runs.

    Production settings: traced, CFG on at gs=5.0, so two transformer forwards per step.
    """
    pipeline = FiboPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=HEIGHT,
        width=WIDTH,
        checkpoint_name=model_location_generator("briaai/FIBO"),
        vlm_checkpoint_name=model_location_generator("briaai/FIBO-vlm"),
    )
    profiler = BenchmarkProfiler()

    # A warmup and an untimed settle run: the first replay after a capture pays one-time costs the
    # steady state does not, and one such outlier moves the median.
    untimed = 2
    images: list[Image.Image] = []
    for iteration in range(untimed + NUM_MEASURED_RUNS):
        phase = "untimed" if iteration < untimed else f"measured {iteration - untimed + 1}/{NUM_MEASURED_RUNS}"
        logger.info(f"perf run {iteration + 1}/{untimed + NUM_MEASURED_RUNS} ({phase})...")
        images = pipeline(
            prompts=[PROMPT],
            num_inference_steps=NUM_INFERENCE_STEPS,
            seed=0,
            cfg_scale=CFG_SCALE,
            on_event=profiler_event_callback(profiler, iteration),
        )

    runs = []
    totals = []
    for iteration in range(untimed, untimed + NUM_MEASURED_RUNS):
        total = profiler.get_duration("total", iteration)
        vlm = profiler.get_duration("vlm", iteration)
        encoder = profiler.get_duration("encoder", iteration)
        denoising = profiler.get_duration("denoising", iteration)
        vae = profiler.get_duration("vae", iteration)
        totals.append(total)
        # The pipeline brackets four sections; "prepare" is the span between the encoder and the
        # denoising loop: scheduler setup, latent sampling, the RoPE build and the prompt upload.
        runs.append(
            {
                "vlm": vlm,
                "encoder": encoder,
                "prepare": total - vlm - encoder - denoising - vae,
                "denoising": denoising,
                "vae": vae,
            }
        )

    # Saved before the asserts, so a degenerate frame still lands on disk to be looked at.
    out_path = Path.cwd() / f"fibo_perf_{WIDTH}x{HEIGHT}_{NUM_INFERENCE_STEPS}steps.png"
    images[0].save(out_path)
    logger.info(f"saved last image -> {out_path}")

    array = np.asarray(images[0])
    assert array.shape == (HEIGHT, WIDTH, 3), f"unexpected image shape {array.shape}"
    assert array.std() > 1.0, f"image looks degenerate (std={array.std():.4f})"
    assert np.unique(array).size > 16, f"image looks degenerate ({np.unique(array).size} unique values)"

    median = {stage: float(np.median([run[stage] for run in runs])) for stage in runs[0]}
    lowest = {stage: min(run[stage] for run in runs) for stage in runs[0]}
    highest = {stage: max(run[stage] for run in runs) for stage in runs[0]}

    # The headline is the median of the measured per-run totals. Summing the per-stage medians
    # would describe a generation that never happened, since each stage's slowest run is a
    # different one; that sum is only used below as the denominator for the percentage shares.
    median_total = float(np.median(totals))
    best_total = min(totals)
    median_sum = sum(median.values())

    lines = [
        f"\nFIBO perf breakdown — {WIDTH}x{HEIGHT}, {NUM_INFERENCE_STEPS} steps, gs={CFG_SCALE} "
        f"[CFG on (2 fwd/step), traced], median of {NUM_MEASURED_RUNS} runs (after {untimed} untimed)"
    ]
    for stage in median:
        share = 100.0 * median[stage] / median_sum if median_sum else 0.0
        note = f"-> {NUM_INFERENCE_STEPS / median[stage]:.2f} it/s" if stage == "denoising" and median[stage] else ""
        lines.append(
            f"  {stage:<9} {median[stage]:7.2f} s  ({share:4.1f}%)  "
            f"[min {lowest[stage]:6.2f} / max {highest[stage]:6.2f}]  {note}"
        )
    lines.append("  " + "-" * 62)
    lines.append(
        f"  {'total':<9} {median_total:7.2f} s  (median full gen)  -> {1.0 / median_total:.4f} images/s"
        f"   |  best {best_total:.2f} s -> {1.0 / best_total:.4f} images/s"
    )
    logger.info("\n".join(lines))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Traced, stage-timed performance harness for the Wan2.2 lightx2v distill (4-step I2V).

This mirrors ``models/tt_dit/tests/models/wan2_2/test_performance_wan.py`` but uses
:class:`WanDistillPipelineI2V` (4 inference steps, CFG baked in) instead of the base
40-step pipeline.

Key differences vs the functional test (``test_pipeline_wan_distill_i2v.py``):

* Device trace is captured/replayed on the 4x32 quad config (``traced=True`` +
  ``trace_region_size``), which is what makes the per-step denoise cost comparable to
  the documented ~20s-class numbers. The functional test runs untraced, so each denoise
  step pays full host-dispatch overhead.
* A traced warmup iteration is run (and excluded from timing); only the subsequent
  iteration is measured, via per-stage profiler events.
* Traces are released at the end.

The expected-metric targets below are provisional upper bounds for a first capture and
should be tightened once real quad numbers land.
"""
from __future__ import annotations

import statistics

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.utils.test import ring_params, ring_params_8k
from models.tt_dit.utils.video import export_to_video

# Trace region needed for trace capture/replay on quad. The distill 4-step graph
# captures ~124 MB of trace buffers, so allocate generous headroom (200 MB).
DEVICE_PARAMS = {"trace_region_size": 200000000}

NUM_FRAMES = 81
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0  # CFG baked into the distill weights.


def create_fractal_image(width: int, height: int) -> Image.Image:
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


def distill_metrics(mesh_device, height):
    """Provisional perf targets (upper bounds) for the distill 4-step I2V pipeline.

    These are intentionally loose so a first capture run passes; tighten after the
    real numbers are known.
    """
    if tuple(mesh_device.shape) == (4, 32):
        assert is_blackhole(), "4x32 is only supported for blackhole"
        assert height == 720, "4x32 is only supported for 720p"
        return {
            "encoder": 2.0,
            "denoising": 30.0,
            "vae": 25.0,
            "total": 60.0,
        }
    if tuple(mesh_device.shape) == (4, 8):
        if height == 720:
            return {
                "encoder": 2.0,
                "denoising": 120.0,
                "vae": 20.0,
                "total": 150.0,
            }
        return {
            "encoder": 2.0,
            "denoising": 60.0,
            "vae": 20.0,
            "total": 90.0,
        }
    assert False, f"Unknown mesh device for distill performance comparison: {mesh_device}"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        # BH Galaxy 4x8 Ring (single galaxy) — untraced baseline.
        [(4, 8), (4, 8), 2, False, ring_params, ttnn.Topology.Ring, False],
        # BH Quad Galaxy 4x32 Ring (multi-host) — traced. Uses the 8K fabric-router config
        # plus a trace region.
        [(4, 32), (4, 32), 2, False, {**DEVICE_PARAMS, **ring_params_8k}, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x8sp1tp0_ring", "bh_4x32sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
        (1280, 720),
    ],
    ids=[
        "resolution_480p",
        "resolution_720p",
    ],
)
def test_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    width: int,
    height: int,
    is_ci_env: bool,
    galaxy_type: str,
    is_fsdp: bool,
) -> None:
    """Performance test for the Wan2.2 distill (4-step) I2V pipeline with stage timing."""

    benchmark_profiler = BenchmarkProfiler()
    traced = mesh_shape == (4, 32)  # Trace only for quad (4x32), matching the base perf harness.

    # Skip 4U.
    if galaxy_type == "4U":
        # NOTE: Pipelines fail if a performance test is skipped without providing a benchmark output.
        if is_ci_env:
            with benchmark_profiler("run", iteration=0):
                pass
            BenchmarkData().save_partial_run_json(
                benchmark_profiler,
                run_type="empty_run",
                ml_model_name="empty_run",
            )
        pytest.skip("4U is not supported for this test")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    expected_metrics = distill_metrics(mesh_device, height)
    image_prompt = create_fractal_image(width, height)

    print(f"Parameters: {height}x{width}, {NUM_FRAMES} frames, {NUM_INFERENCE_STEPS} steps (distill), traced={traced}")

    pipeline = WanDistillPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
    )

    prompts = [
        "The cat in the hat runs up the hill to the house.",
        "A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.",
    ]

    # Warmup run (not timed). Captures the device trace when traced=True.
    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        if traced:
            with torch.no_grad():
                pipeline(
                    prompts=[prompts[0]],
                    image_prompt=image_prompt,
                    num_inference_steps=2,  # Small step count to reduce warmup time; warms both experts.
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE,
                    traced=traced,
                )
    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    # Performance measurement run.
    logger.info("Running performance measurement iteration...")
    num_perf_runs = 1

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    frames = None
    for i in range(num_perf_runs):
        logger.info(f"Performance run {i + 1}/{num_perf_runs}...")
        prompt_idx = (i + 1) % len(prompts)
        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                frames = pipeline(
                    prompts=[prompts[prompt_idx]],
                    image_prompt=image_prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    on_event=profiler_event_callback(benchmark_profiler, i),
                    seed=42,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE,
                    traced=traced,
                    output_type="uint8",
                )
                ttnn.synchronize_device(mesh_device)
        logger.info(f"  Run {i + 1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    pipeline.release_traces()

    print("✓ Inference completed successfully")
    print(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
    print(f"  Output type: {type(frames)}")
    if isinstance(frames, np.ndarray):
        print(f"  Video dtype:      {frames.dtype}")
        print(f"  Video data range: [{frames.min()}, {frames.max()}]")
    elif isinstance(frames, torch.Tensor):
        print(f"  Video dtype:      {frames.dtype}")
        print(f"  Video data range: [{frames.min().item()}, {frames.max().item()}]")

    # Save video (skip in CI).
    frames = frames[0]
    if not is_ci_env and int(ttnn.distributed_context_get_rank()) == 0:
        output_path = f"wan_distill_i2v_{width}x{height}{'_traced' if traced else ''}.mp4"
        try:
            export_to_video(frames, output_path, fps=16)
            print(f"✓ Saved video to: {output_path}")
        except ImportError:
            print("Could not export video - imageio_ffmpeg not available")

    # Calculate statistics.
    text_encoder_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    prepare_latents_times = [benchmark_profiler.get_duration("prepare_latents", i) for i in range(num_perf_runs)]
    denoising_times = [benchmark_profiler.get_duration("denoising", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    print("\n" + "=" * 80)
    print("WAN DISTILL PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height}")
    print(f"Inference Steps: {NUM_INFERENCE_STEPS}")
    print(f"Num Frames: {NUM_FRAMES}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {topology}")
    print(f"Traced: {traced}")
    print("-" * 80)

    def print_stats(name, times):
        if not times:
            print(f"{name:25} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min(times):8.4f}s | Max: {max(times):8.4f}s"
        )

    print_stats("Text Encoding", text_encoder_times)
    print_stats("Image Encoding (total)", prepare_latents_times)
    print_stats("Denoising", denoising_times)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)
    print("-" * 80)

    measurements = {
        "encoder": statistics.mean(text_encoder_times),
        "denoising": statistics.mean(denoising_times),
        "vae": statistics.mean(vae_times),
        "total": statistics.mean(total_times),
    }

    if is_ci_env:
        benchmark_data = BenchmarkData()
        for iteration in range(num_perf_runs):
            for step_name in ["encoder", "denoising", "vae", "run"]:
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=expected_metrics["total" if step_name == "run" else step_name],
                )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="BH_QGLX" if mesh_shape == (4, 32) else "BH_GLX",
            ml_model_name="Wan2.2-Distill",
            batch_size=1,
            config_params={
                "width": width,
                "height": height,
                "num_frames": NUM_FRAMES,
                "num_steps": NUM_INFERENCE_STEPS,
                "topology": str(topology),
                "num_links": num_links,
                "fsdp": is_fsdp,
                "traced": traced,
            },
        )

    pass_perf_check = True
    assert_msgs = []
    for k in expected_metrics:
        if measurements[k] > expected_metrics[k]:
            assert_msgs.append(
                f"Warning: {k} is outside of the tolerance range. "
                f"Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False

    assert pass_perf_check, "\n".join(assert_msgs)

    logger.info("Distill performance test completed successfully!")

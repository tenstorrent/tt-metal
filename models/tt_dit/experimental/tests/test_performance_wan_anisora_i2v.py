# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Traced, stage-timed performance harness for the AniSora V3.2 I2V pipeline.

This mirrors ``test_performance_wan_distill_i2v.py`` but uses
:class:`AniSoraPipeline`. Unlike the distill (4-step, CFG baked in), AniSora is a
full anime fine-tune: it runs *real* classifier-free guidance
(``guidance_scale=3.5`` on both experts) and a configurable step count
(upstream default 40). Step count therefore dominates cost and there is no
CFG-baked shortcut — each step runs cond + uncond forwards.

Step count is taken from the ``NUM_STEPS`` env var (default 8) so the same
harness can time 8 / 16 / 40 steps. As with the distill/base harnesses, the
device trace is captured/replayed only on the 4x32 quad; a warmup iteration is
run and excluded, and only the subsequent iteration is measured via per-stage
profiler events.

The image-encode optimizations developed for the distill (truncation, swept
conv3d blockings, on-device conditioning) are now available to AniSora via
``FastImageEncodeMixin`` and are gated behind the ``WAN_ANISORA_FAST_VAE_ENCODER``
/ ``WAN_ANISORA_ENCODER_T_OUT_1`` / ``WAN_ANISORA_ONDEVICE_COND`` env flags. With
all three set, "Image Encoding (total)" drops from ~7.5s to ~0.35s; unset, this
harness reflects the full 81-frame encode path.
"""
from __future__ import annotations

import os
import statistics

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.experimental.pipelines.pipeline_anisora import AniSoraPipeline
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import ring_params, ring_params_8k
from models.tt_dit.utils.video import export_to_video

# Trace region for capture/replay on quad. AniSora shares the base Wan2.2 I2V
# transformer (base harness uses 150 MB); allocate generous headroom (200 MB).
DEVICE_PARAMS = {"trace_region_size": 200000000}

NUM_FRAMES = 81
# Overridable so one harness can time 8 / 16 / 40 steps.
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_STEPS", "8"))
GUIDANCE_SCALE = 3.5  # AniSora upstream config — real CFG on both experts.
GUIDANCE_SCALE_2 = 3.5


def create_fractal_image(width: int, height: int) -> Image.Image:
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


def _load_condition_image(width: int, height: int) -> Image.Image:
    """Use ``PROMPT_IMAGE`` if set (sized to WxH), else a synthetic fractal.

    Image content does not affect timing, but allowing a real image keeps the
    saved perf video meaningful when sanity-checking quality alongside perf.
    """
    path = os.environ.get("PROMPT_IMAGE")
    if path:
        return Image.open(path).convert("RGB").resize((width, height))
    return create_fractal_image(width, height)


def anisora_metrics(mesh_device, height, num_steps):
    """Provisional perf targets (loose upper bounds), scaled by step count.

    AniSora pays cond+uncond per step, so the denoise bound scales with steps.
    Intentionally generous so a first capture passes; tighten once real numbers
    land.
    """
    if tuple(mesh_device.shape) == (4, 32):
        assert is_blackhole(), "4x32 is only supported for blackhole"
        assert height == 720, "4x32 is only supported for 720p"
        denoising = num_steps * 8.0
        return {
            "encoder": 20.0,
            "denoising": denoising,
            "vae": 30.0,
            "total": denoising + 70.0,
        }
    if tuple(mesh_device.shape) == (4, 8):
        denoising = num_steps * 20.0
        return {
            "encoder": 20.0,
            "denoising": denoising,
            "vae": 30.0,
            "total": denoising + 80.0,
        }
    assert False, f"Unknown mesh device for AniSora performance comparison: {mesh_device}"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        # BH Galaxy 4x8 Ring (single galaxy) — untraced baseline.
        [(4, 8), (4, 8), 2, False, ring_params, ttnn.Topology.Ring, False],
        # BH Quad Galaxy 4x32 Ring (multi-host) — traced. 8K fabric-router config + trace region.
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
    """Performance test for the AniSora V3.2 I2V pipeline with stage timing."""

    benchmark_profiler = BenchmarkProfiler()
    traced = mesh_shape == (4, 32)  # Trace only for quad (4x32), matching the base/distill harnesses.

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

    expected_metrics = anisora_metrics(mesh_device, height, NUM_INFERENCE_STEPS)
    image_prompt = [ImagePrompt(image=_load_condition_image(width, height), frame_pos=0)]

    print(
        f"Parameters: {height}x{width}, {NUM_FRAMES} frames, {NUM_INFERENCE_STEPS} steps "
        f"(AniSora, CFG={GUIDANCE_SCALE}), traced={traced}"
    )

    pipeline = AniSoraPipeline.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
    )

    prompt = os.environ.get(
        "PROMPT",
        "The anime girl smiles gently as cherry blossom petals drift past her face, "
        "her purple hair flowing softly in a spring breeze, sparkling light, anime style",
    )
    negative_prompt = ""

    # Warmup run (not timed). Captures the device trace when traced=True.
    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        if traced:
            with torch.no_grad():
                pipeline(
                    prompts=[prompt],
                    image_prompt=image_prompt,
                    negative_prompts=[negative_prompt],
                    num_inference_steps=2,  # Small step count to reduce warmup time; warms both experts.
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE_2,
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
        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                frames = pipeline(
                    prompts=[prompt],
                    image_prompt=image_prompt,
                    negative_prompts=[negative_prompt],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    on_event=profiler_event_callback(benchmark_profiler, i),
                    seed=42,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE_2,
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
        output_path = f"wan_anisora_i2v_{width}x{height}_{NUM_INFERENCE_STEPS}steps{'_traced' if traced else ''}.mp4"
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
    print("WAN ANISORA PERFORMANCE RESULTS")
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
            ml_model_name="Wan2.2-AniSora",
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

    logger.info("AniSora performance test completed successfully!")

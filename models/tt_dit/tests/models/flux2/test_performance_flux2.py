# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import statistics

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .test_pipeline_flux2 import line_params_8k_flux2, line_params_flux2, ring_params_8k_flux2

NUM_INFERENCE_STEPS = 50
NUM_PERF_RUNS = 3


@pytest.mark.parametrize(
    "width, height",
    [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ],
    ids=[
        "1024x1024",
        "2048x2048",
        "4096x4096",
        "8192x8192",
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, encoder_tp_axis, vae_tp_axis, topology, num_links, is_fsdp, dynamic_load, device_params",
    [
        [(2, 2), 0, 1, 1, 1, ttnn.Topology.Linear, 2, True, False, line_params_flux2],
        [(2, 4), 0, 1, 1, 1, ttnn.Topology.Linear, 2, False, False, line_params_flux2],
        [(4, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 2, False, False, line_params_8k_flux2],
        [(4, 8), 0, 1, 1, 0, ttnn.Topology.Ring, 2, False, False, ring_params_8k_flux2],
        [(4, 8), 1, 0, 1, 0, ttnn.Topology.Ring, 2, False, False, ring_params_8k_flux2],
        [(4, 8), 0, 1, 1, 0, ttnn.Topology.Ring, 2, True, False, ring_params_8k_flux2],
    ],
    ids=[
        "bh_qb",
        "bh_lb",
        "bh_glx_linear",
        "bh_glx_ring_sp0tp1_nofsdp",
        "bh_glx_ring_sp1tp0",
        "bh_glx_ring_sp0tp1_fsdp",
    ],
    indirect=["mesh_device", "device_params"],
)
def test_flux2_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    encoder_tp_axis: int,
    vae_tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    is_fsdp: bool,
    dynamic_load: bool,
    width: int,
    height: int,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    """Performance test for Flux2 pipeline across machine types and resolutions."""

    benchmark_profiler = BenchmarkProfiler()

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    logger.info(
        f"Config: {width}x{height}, mesh={tuple(mesh_device.shape)}, "
        f"sp={sp_factor}, tp={tp_factor}, topology={topology}"
    )

    # Pipeline creation includes internal warmup (2 steps at target resolution).
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        encoder_tp_axis=encoder_tp_axis,
        vae_tp_axis=vae_tp_axis,
        vae_h_axis=1 - vae_tp_axis,
        vae_w_axis=None,  # 1-vae_tp_axis,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        dynamic_load=dynamic_load,
        is_fsdp=is_fsdp,
        trace_warmup=True,
        shard_prompt=True,
    )

    # Performance measurement runs
    logger.info(f"Running {NUM_PERF_RUNS} timed iterations...")

    for i in range(NUM_PERF_RUNS):
        logger.info(f"Performance run {i + 1}/{NUM_PERF_RUNS}...")

        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()

        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                images = pipeline(
                    prompts=["A photo of a cat sitting on a windowsill at sunset"],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    seed=42,
                    traced=True,
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )
                ttnn.synchronize_device(mesh_device)

        logger.info(f"  Run {i + 1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    # Collect timing data across all runs
    encoder_times = [benchmark_profiler.get_duration("encoder", i) for i in range(NUM_PERF_RUNS)]
    denoising_times = [benchmark_profiler.get_duration("denoising", i) for i in range(NUM_PERF_RUNS)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(NUM_PERF_RUNS)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(NUM_PERF_RUNS)]

    all_denoising_steps = []
    for i in range(NUM_PERF_RUNS):
        for j in range(NUM_INFERENCE_STEPS):
            if benchmark_profiler.contains_step(f"denoising_step_{j}", i):
                all_denoising_steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", i))

    # Save output image on rank 0 (from last run)
    if not is_ci_env:
        rank = int(ttnn.distributed_context_get_rank())
        if rank == 0:
            mesh_tag = "x".join(str(s) for s in mesh_device.shape)
            output_path = f"flux2_{mesh_tag}_{width}x{height}.png"
            images[0].save(output_path)
            logger.info(f"Image saved as {output_path}")

    # Report results
    print("\n" + "=" * 80)
    print("FLUX2 PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height}")
    print(f"Inference Steps: {NUM_INFERENCE_STEPS}")
    print(f"Performance Runs: {NUM_PERF_RUNS}")
    print(f"DiT Configuration: sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {tuple(mesh_device.shape)}")
    print(f"Topology: {topology}")
    print("-" * 80)

    def print_stats(name, times):
        if not times:
            print(f"{name:25} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    print_stats("Encoding", encoder_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("Denoising (total)", denoising_times)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)
    print("-" * 80)

    # Additional metrics
    avg_total = statistics.mean(total_times)
    avg_encoding = statistics.mean(encoder_times)
    avg_denoising = statistics.mean(denoising_times)
    avg_vae = statistics.mean(vae_times)

    if all_denoising_steps:
        print(f"Average total denoising time: {avg_denoising:.4f}s")
        print(f"Denoising throughput: {NUM_INFERENCE_STEPS / avg_denoising:.2f} steps/second")

    print(f"Overall throughput: {1 / avg_total:.4f} images/second")

    print(f"\nTime breakdown:")
    print(f"  Encoding:  {avg_encoding / avg_total * 100:5.1f}%")
    print(f"  Denoising: {avg_denoising / avg_total * 100:5.1f}%")
    print(f"  VAE:       {avg_vae / avg_total * 100:5.1f}%")

    print("=" * 80)

    if is_ci_env:
        device_name_map = {
            (2, 2): "BH_QB",
            (2, 4): "BH_LB" if is_blackhole() else "WH_T3K",
            (4, 8): "BH_GLX" if is_blackhole() else "WH_GLX",
        }

        benchmark_data = BenchmarkData()
        for i in range(NUM_PERF_RUNS):
            for step_name in ["encoder", "denoising", "vae", "run"]:
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=i,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, i),
                    target=benchmark_profiler.get_duration(step_name, i),  # No baseline targets yet
                )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map.get(tuple(mesh_device.shape), "UNKNOWN"),
            ml_model_name=os.path.basename(pipeline.checkpoint_name),
            batch_size=1,
            config_params={
                "width": width,
                "height": height,
                "num_steps": NUM_INFERENCE_STEPS,
                "num_perf_runs": NUM_PERF_RUNS,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
            },
        )

    logger.info("Performance test completed successfully!")


@pytest.mark.parametrize(
    "width, height",
    [
        (1024, 1024),
    ],
    ids=[
        "1024x1024",
    ],
)
@pytest.mark.parametrize(
    "cond_width, cond_height",
    [
        (512, 512),
    ],
    ids=[
        "cond512x512",
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, encoder_tp_axis, vae_tp_axis, topology, num_links, is_fsdp, dynamic_load, device_params",
    [
        [(4, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 4, True, False, line_params_flux2],
    ],
    ids=[
        "wh_glx_linear",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(1800)  # full e2e img2img perf run (encoder load + trace warmups + 3x50-step runs)
def test_flux2_img2img_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    encoder_tp_axis: int,
    vae_tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    is_fsdp: bool,
    dynamic_load: bool,
    width: int,
    height: int,
    cond_width: int,
    cond_height: int,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    """Image-to-image performance test for the Flux2 pipeline on a 4x8 Wormhole galaxy.

    Reports the on-device VAE-encode duration (the img2img input path added by this work) alongside
    the full image-to-image pipeline breakdown (prompt encode, denoising, VAE decode, total).
    """
    benchmark_profiler = BenchmarkProfiler()

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    logger.info(
        f"img2img config: {width}x{height} (cond {cond_width}x{cond_height}), "
        f"mesh={tuple(mesh_device.shape)}, sp={sp_factor}, tp={tp_factor}, topology={topology}"
    )

    # Deterministic condition image for the img2img reference path. Built before the pipeline so it
    # can be passed as warmup_image, ensuring the traced latents/rope buffers are sized for the
    # combined noise+image sequence (the timed traced runs reuse those buffers).
    torch.manual_seed(42)
    cond_array = (torch.rand(cond_height, cond_width, 3) * 255).to(torch.uint8).numpy()
    condition_image = Image.fromarray(cond_array, mode="RGB")

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        encoder_tp_axis=encoder_tp_axis,
        vae_tp_axis=vae_tp_axis,
        vae_h_axis=1 - vae_tp_axis,
        vae_w_axis=None,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        dynamic_load=dynamic_load,
        is_fsdp=is_fsdp,
        trace_warmup=True,
        shard_prompt=True,
        warmup_image=condition_image,
    )

    logger.info(f"Running {NUM_PERF_RUNS} timed img2img iterations...")

    for i in range(NUM_PERF_RUNS):
        logger.info(f"img2img performance run {i + 1}/{NUM_PERF_RUNS}...")

        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()

        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                images = pipeline(
                    prompts=["A photo of a cat sitting on a windowsill at sunset"],
                    image=condition_image,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    seed=42,
                    traced=True,
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )
                ttnn.synchronize_device(mesh_device)

        logger.info(f"  Run {i + 1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    def _durations(step_name: str) -> list[float]:
        out = []
        for i in range(NUM_PERF_RUNS):
            if benchmark_profiler.contains_step(step_name, i):
                out.append(benchmark_profiler.get_duration(step_name, i))
        return out

    encoder_times = _durations("encoder")
    vae_encode_times = _durations("vae_encode")
    denoising_times = _durations("denoising")
    vae_times = _durations("vae")
    total_times = _durations("run")

    if not is_ci_env:
        rank = int(ttnn.distributed_context_get_rank())
        if rank == 0:
            mesh_tag = "x".join(str(s) for s in mesh_device.shape)
            output_path = f"flux2_img2img_{mesh_tag}_{width}x{height}.png"
            images[0].save(output_path)
            logger.info(f"Image saved as {output_path}")

    print("\n" + "=" * 80)
    print("FLUX2 IMAGE-TO-IMAGE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height} | Condition Image: {cond_width}x{cond_height}")
    print(f"Inference Steps: {NUM_INFERENCE_STEPS}")
    print(f"Performance Runs: {NUM_PERF_RUNS}")
    print(f"DiT Configuration: sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {tuple(mesh_device.shape)}")
    print(f"Topology: {topology}")
    print("-" * 80)

    def print_stats(name, times):
        if not times:
            print(f"{name:28} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        print(
            f"{name:28} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min(times):8.4f}s | Max: {max(times):8.4f}s"
        )

    print_stats("Prompt encoding", encoder_times)
    print_stats("VAE encode (img2img)", vae_encode_times)
    print_stats("Denoising (total)", denoising_times)
    print_stats("VAE decoding", vae_times)
    print_stats("Total img2img pipeline", total_times)
    print("-" * 80)

    if total_times:
        avg_total = statistics.mean(total_times)
        print(f"Overall throughput: {1 / avg_total:.4f} images/second")
        print("\nTime breakdown:")
        if vae_encode_times:
            print(f"  VAE encode (img2img): {statistics.mean(vae_encode_times) / avg_total * 100:5.1f}%")
        if encoder_times:
            print(f"  Prompt encoding:      {statistics.mean(encoder_times) / avg_total * 100:5.1f}%")
        if denoising_times:
            print(f"  Denoising:            {statistics.mean(denoising_times) / avg_total * 100:5.1f}%")
        if vae_times:
            print(f"  VAE decoding:         {statistics.mean(vae_times) / avg_total * 100:5.1f}%")
    print("=" * 80)

    if is_ci_env:
        benchmark_data = BenchmarkData()
        for i in range(NUM_PERF_RUNS):
            for step_name in ["encoder", "vae_encode", "denoising", "vae", "run"]:
                if benchmark_profiler.contains_step(step_name, i):
                    benchmark_data.add_measurement(
                        profiler=benchmark_profiler,
                        iteration=i,
                        step_name=step_name,
                        name=step_name,
                        value=benchmark_profiler.get_duration(step_name, i),
                        target=benchmark_profiler.get_duration(step_name, i),
                    )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="BH_GLX" if is_blackhole() else "WH_GLX",
            ml_model_name=os.path.basename(pipeline.checkpoint_name),
            batch_size=1,
            config_params={
                "width": width,
                "height": height,
                "cond_width": cond_width,
                "cond_height": cond_height,
                "num_steps": NUM_INFERENCE_STEPS,
                "num_perf_runs": NUM_PERF_RUNS,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
                "mode": "img2img",
            },
        )

    logger.info("img2img performance test completed successfully!")

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import torch
import ttnn
import numpy as np
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from models.common.utility_functions import is_blackhole
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.utils import export_to_video
from ....parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from ....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        [(1, 8), (1, 8), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "1x8sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
    ],
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
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: dict,
    topology: ttnn.Topology,
    width: int,
    height: int,
    is_ci_env: bool,
    galaxy_type: str,
    is_fsdp: bool,
) -> None:
    """Performance test for Wan pipeline with detailed timing analysis."""

    benchmark_profiler = BenchmarkProfiler()

    # Skip 4U.
    if galaxy_type == "4U":
        # NOTE: Pipelines fail if a performance test is skipped without providing a benchmark output.
        if is_ci_env:
            with benchmark_profiler("run", iteration=0):
                pass

            benchmark_data = BenchmarkData()
            benchmark_data.save_partial_run_json(
                benchmark_profiler,
                run_type="empty_run",
                ml_model_name="empty_run",
            )
        pytest.skip("4U is not supported for this test")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
    )

    # Test prompts
    prompts = [
        """Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.""",
        """A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.""",
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic depth and warmth.""",
    ]

    num_frames = 81
    num_inference_steps = 40

    print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    pipeline = WanPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
    )

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")

    with benchmark_profiler("run", iteration=0):
        with torch.no_grad():
            result = pipeline(
                prompt=prompts[0],
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,  # Small number of steps to reduce test time.
            )

    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    # Check output
    if hasattr(result, "frames"):
        frames = result.frames
    else:
        frames = result[0] if isinstance(result, tuple) else result

    print(f"✓ Inference completed successfully")
    print(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
    print(f"  Output type: {type(frames)}")

    # Basic validation
    if isinstance(frames, np.ndarray):
        print(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
    elif isinstance(frames, torch.Tensor):
        print(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

    # Save video using diffusers utility
    # Remove batch dimension
    frames = frames[0]
    try:
        export_to_video(frames, "wan_output_video.mp4", fps=16)
    except AttributeError as e:
        logger.info(f"AttributeError: {e}")
    print("✓ Saved video to: wan_output_video.mp4")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    num_perf_runs = 1  # For now use 1 prompt to minimize test time.

    for i in range(num_perf_runs):
        logger.info(f"Performance run {i+1}/{num_perf_runs}...")

        # Run pipeline with different prompt
        prompt_idx = (i + 1) % len(prompts)
        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                pipeline(
                    prompt=prompts[prompt_idx],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )

        logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    # Calculate statistics
    text_encoder_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    denoising_times = [benchmark_profiler.get_duration("denoising", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    # Report results
    print("\n" + "=" * 80)
    print("WAN PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Num Frames: {num_frames}")
    print(f"DiT Configuration: sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {mesh_device.shape}")
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
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    print_stats("Text Encoding", text_encoder_times)
    print_stats("Denoising", denoising_times)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)
    print("-" * 80)

    # Validate performance
    measurements = {
        "encoder": statistics.mean(text_encoder_times),
        "denoising": statistics.mean(denoising_times),
        "vae": statistics.mean(vae_times),
        "total": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 4) and height == 480:
        expected_metrics = {
            "encoder": 19.0,
            "denoising": 800.0,
            "vae": 9.0,
            "total": 850.0,
        }
    elif tuple(mesh_device.shape) == (4, 8) and height == 480:
        expected_metrics = {
            "encoder": 15.0,
            "denoising": 163.0,
            "vae": 18.2,
            "total": 192.0,
        }
    elif tuple(mesh_device.shape) == (4, 8) and height == 720:
        if is_blackhole():
            expected_metrics = {
                "encoder": 15.0,
                "denoising": 185.0,
                "vae": 8.0,
                "total": 208.0,
            }
        else:
            expected_metrics = {
                "encoder": 15.0,
                "denoising": 440.0,
                "vae": 8.0,
                "total": 463.0,
            }
    elif tuple(mesh_device.shape) == (2, 2):
        assert height == 480, "2x2 is only supported for 480p"
        assert is_blackhole(), "2x2 is only supported for blackhole"
        expected_metrics = {
            "encoder": 27.0,
            "denoising": 680.0,
            "vae": 60.0,
            "total": 760.0,
        }
    elif tuple(mesh_device.shape) == (1, 8) and height == 480:
        assert is_blackhole(), "1x8 is only supported for blackhole"
        expected_metrics = {
            "encoder": 23.0,
            "denoising": 426.6,
            "vae": 10.0,
            "total": 449.3,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
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
        device_name_map = {
            (2, 2): "BH_QB",
            (2, 4): "WH_T3K",
            (1, 8): "BH_LB",
            (4, 8): "BH_GLX" if is_blackhole() else "WH_GLX",
        }
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map[mesh_shape],
            ml_model_name="Wan2.2",
            batch_size=1,
            config_params={
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_steps": num_inference_steps,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
                "num_links": num_links,
                "fsdp": is_fsdp,
            },
        )

    pass_perf_check = True
    assert_msgs = []
    for k in expected_metrics.keys():
        if measurements[k] > expected_metrics[k]:
            assert_msgs.append(
                f"Warning: {k} is outside of the tolerance range. Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False

    assert pass_perf_check, "\n".join(assert_msgs)

    logger.info("Performance test completed successfully!")

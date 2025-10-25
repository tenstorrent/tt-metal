# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import torch
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData

from ....pipelines.mochi.pipeline_mochi import MochiPipeline as TTMochiPipeline
from ....parallel.config import DiTParallelConfig, MochiVAEParallelConfig, ParallelFactor


@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps, num_frames",
    [
        ("genmo/mochi-1-preview", 848, 480, 3.5, 50, 168),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, vae_mesh_shape, vae_sp_axis, vae_tp_axis, topology, num_links",
    [
        # VAE mesh shape = (1, 8) is more memory efficient.
        [(2, 4), 0, 1, (1, 8), 0, 1, ttnn.Topology.Linear, 1],
        [(4, 8), 1, 0, (4, 8), 0, 1, ttnn.Topology.Linear, 4],  # note sp <-> tp switch for VAE for memory efficiency.
    ],
    ids=[
        "dit_2x4sp0tp1_vae_1x8sp0tp1",
        "dit_4x8sp1tp0_vae_4x8sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("use_cache", [True, False], ids=["yes_use_cache", "no_use_cache"])
def test_mochi_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name: str,
    image_w: int,
    image_h: int,
    guidance_scale: float,
    num_inference_steps: int,
    num_frames: int,
    sp_axis: int,
    tp_axis: int,
    vae_mesh_shape: tuple,
    vae_sp_axis: int,
    vae_tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    use_cache: bool,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """Performance test for Mochi pipeline with detailed timing analysis."""

    benchmark_profiler = BenchmarkProfiler()

    # Process skips
    if is_ci_env and use_cache:
        pytest.skip("use_cache not necessary for performance test in CI. See pipeline test.")

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

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Guidance scale: {guidance_scale}")
    logger.info(f"  Inference steps: {num_inference_steps}")
    logger.info(f"  Number frames: {num_frames}")

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    logger.info(
        f"Creating TT Mochi pipeline with DiT mesh device shape {mesh_device.shape}, VAE mesh device shape {vae_mesh_shape}"
    )
    logger.info(f"DiT SP axis: {sp_axis}, TP axis: {tp_axis}")
    logger.info(f"VAE SP axis: {vae_sp_axis}, TP axis: {tp_axis}")

    # Create parallel config
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    if vae_mesh_shape[vae_sp_axis] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=vae_mesh_shape[vae_tp_axis], mesh_axis=vae_tp_axis),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=vae_sp_axis),
        h_parallel=ParallelFactor(factor=vae_mesh_shape[vae_sp_axis] // w_parallel_factor, mesh_axis=vae_sp_axis),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == vae_mesh_shape[vae_sp_axis]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    # Create the TT Mochi pipeline
    tt_pipe = TTMochiPipeline(
        mesh_device=mesh_device,
        vae_mesh_shape=vae_mesh_shape,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=use_cache,
        use_reference_vae=False,
        model_name=model_name,
    )

    # Use a generator for deterministic results.
    generator = torch.Generator("cpu").manual_seed(0)

    # Test prompts
    prompts = [
        """A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.""",
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic depth and warmth.""",
    ]

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")

    frames = tt_pipe(
        prompts[0],
        num_inference_steps=2,  # Small number of steps to reduce test time.
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        height=image_h,
        width=image_w,
        generator=generator,
    ).frames[0]

    logger.info(f"Warmup completed in {tt_pipe.timing_data['total']:.2f}s")

    # Validate output
    assert frames is not None, "No frames were generated by the TT pipeline"
    assert len(frames) > 0, "Empty frames list generated by the TT pipeline"

    # Check frame properties
    first_frame = frames[0]
    logger.info(f"TT Pipeline generated {len(frames)} frames, first frame size: {first_frame.size}")

    # Optional: Export to video file
    try:
        from diffusers.utils import export_to_video

        export_to_video(frames, "tt_mochi_test_output.mp4", fps=30)
        logger.info("TT Pipeline video exported to tt_mochi_test_output.mp4")
    except ImportError:
        logger.info("Could not export video - diffusers.utils.export_to_video not available")
    except AttributeError as e:
        logger.info(f"AttributeError: {e}")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    all_timings = []
    num_perf_runs = 1  # For now use 1 prompt to minimize test time.

    # Optional Tracy profiling (if available)
    profiler = None
    try:
        from tracy import Profiler

        profiler = Profiler()
        profiler.enable()
        logger.info("Tracy profiling enabled")
    except ImportError:
        logger.info("Tracy profiler not available, continuing without profiling")

    try:
        for i in range(num_perf_runs):
            logger.info(f"Performance run {i+1}/{num_perf_runs}...")

            # Run pipeline with different prompt
            prompt_idx = (i + 1) % len(prompts)
            with benchmark_profiler("run", iteration=i):
                frames = tt_pipe(
                    prompts[prompt_idx],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    height=image_h,
                    width=image_w,
                    generator=generator,
                ).frames[0]

            # Collect timing data
            all_timings.append(tt_pipe.timing_data)
            logger.info(f"  Run {i+1} completed in {tt_pipe.timing_data['total']:.2f}s")

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # Calculate statistics
    text_encoder_times = [t["text_encoder"] for t in all_timings]
    denoising_times = [t["denoising"] for t in all_timings]
    vae_times = [t["vae"] for t in all_timings]
    total_times = [t["total"] for t in all_timings]

    # Report results
    cfg_factor = tt_pipe.parallel_config.cfg_parallel.factor
    sp_factor = tt_pipe.parallel_config.sequence_parallel.factor
    tp_factor = tt_pipe.parallel_config.tensor_parallel.factor

    print("\n" + "=" * 80)
    print("MOCHI PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"DiT Configuration: cfg={cfg_factor}, sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"VAE Mesh Shape: {vae_mesh_shape}")
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

    # Validate that we got reasonable results
    assert len(all_timings) == num_perf_runs, f"Expected {num_perf_runs} timing results, got {len(all_timings)}"
    assert all(t["total"] > 0 for t in all_timings), "All runs should have positive total time"

    # Validate performance
    measurements = {
        "text_encoding_time": statistics.mean(text_encoder_times),
        "denoising_time": statistics.mean(denoising_times),
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 4) and vae_mesh_shape == (1, 8):
        expected_metrics = {
            "text_encoding_time": 4.07,
            "denoising_time": 1320,
            "vae_decoding_time": 55,
            "total_time": 1385,
        }
    elif tuple(mesh_device.shape) == (4, 8) and vae_mesh_shape == (4, 8):
        expected_metrics = {
            "text_encoding_time": 4.43,
            "denoising_time": 400,
            "vae_decoding_time": 22,
            "total_time": 430,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
        profiler_model_name = (
            f"mochi_{'t3k' if tuple(mesh_device.shape) == (2, 4) else 'tg'}_sp{sp_factor}_tp{tp_factor}"
        )
        benchmark_data = BenchmarkData()
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="mochi_traced",
            ml_model_name=profiler_model_name,
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

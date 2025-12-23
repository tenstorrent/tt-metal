# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from models.common.utility_functions import is_blackhole

from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline


def get_expected_metrics(mesh_device):
    if tuple(mesh_device.shape) == (2, 4):
        return {
            "clip_encoding_time": 0.15,
            "t5_encoding_time": 0.1,
            "total_encoding_time": 0.25,
            "denoising_steps_time": 11.3,
            "vae_decoding_time": 1.6,
            "total_time": 13.2,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        return {
            "clip_encoding_time": 0.2,
            "t5_encoding_time": 0.12,
            "total_encoding_time": 0.6,
            "denoising_steps_time": 4.2,
            "vae_decoding_time": 1.35,
            "total_time": 5.9,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"


@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("large", 1024, 1024, 3.5, 20),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4cfg1sp0tp1",
        "4x8cfg1sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
def test_sd35_new_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg,
    sp,
    tp,
    topology,
    num_links,
    model_location_generator,
    is_ci_env,
    galaxy_type,
) -> None:
    """Performance test for new SD35 pipeline with detailed timing analysis."""

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

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Guidance scale: {guidance_scale}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    pipeline = StableDiffusion3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        batch_size=1,
        image_w=image_w,
        image_h=image_h,
        guidance_scale=guidance_scale,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        num_links=num_links,
        checkpoint_name=model_location_generator(
            f"stabilityai/stable-diffusion-3.5-{model_name}", model_subdir="StableDiffusion_35_Large"
        ),
    )

    # Test prompts - diverse set for comprehensive performance testing
    prompts = [
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """A tiny, fluffy dragon curled up in a teacup, warm cozy lighting, big expressive eyes, intricate scale patterns, surrounded by books and potions — high detail, Studio Ghibli meets Pixar""",
        """An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic depth and warmth.""",
    ]
    negative_prompt = ""

    # Warmup run
    logger.info("Running warmup iteration...")

    with benchmark_profiler("run", iteration=0):
        images = pipeline(
            prompt_1=[prompts[0]],
            prompt_2=[prompts[0]],
            prompt_3=[prompts[0]],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )
    images[0].save(f"sd35_new_{image_w}_{image_h}_warmup.png")

    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    num_perf_runs = 4  # Use 4 different prompts for performance testing

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
                images = pipeline(
                    prompt_1=[prompts[prompt_idx]],
                    prompt_2=[prompts[prompt_idx]],
                    prompt_3=[prompts[prompt_idx]],
                    negative_prompt_1=[negative_prompt],
                    negative_prompt_2=[negative_prompt],
                    negative_prompt_3=[negative_prompt],
                    num_inference_steps=num_inference_steps,
                    seed=0,  # Different seed for each run
                    traced=True,
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )
            images[0].save(f"sd35_new_{image_w}_{image_h}_perf_run{i}.png")

            # Collect timing data
            logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # Calculate statistics
    clip_times = [benchmark_profiler.get_duration("clip_encoding", i) for i in range(num_perf_runs)]
    t5_times = [benchmark_profiler.get_duration("t5_encoding", i) for i in range(num_perf_runs)]
    total_encoding_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    # Calculate per-step denoising times
    all_denoising_steps = []
    for i in range(num_perf_runs):
        for j in range(num_inference_steps):
            assert benchmark_profiler.contains_step(
                f"denoising_step_{j}", i
            ), f"All runs should have {num_inference_steps} denoising steps"
            all_denoising_steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", i))

    # Report results
    cfg_factor = pipeline.dit_parallel_config.cfg_parallel.factor
    sp_factor = pipeline.dit_parallel_config.sequence_parallel.factor
    tp_factor = pipeline.dit_parallel_config.tensor_parallel.factor
    enable_t5_text_encoder = pipeline.t5_enabled()

    print("\n" + "=" * 80)
    print("STABLE DIFFUSION 3.5 NEW PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: cfg={cfg_factor}, sp={sp_factor}, tp={tp_factor}")
    print(f"T5 Text Encoder: {'Enabled' if enable_t5_text_encoder else 'Disabled'}")
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

    print_stats("CLIP Encoding", clip_times)
    print_stats("T5 Encoding", t5_times)
    print_stats("Total Encoding", total_encoding_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)

    print("-" * 80)

    # Additional metrics
    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * num_inference_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {num_inference_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} images/second")

        # Breakdown percentages
        avg_encoding_time = statistics.mean(total_encoding_times)
        avg_vae_time = statistics.mean(vae_times)

        print(f"\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE: {avg_vae_time/avg_total_time*100:.1f}%")

        # Performance benchmarks (set reasonable targets)
        print(f"\nPerformance Analysis:")
        if avg_total_time < 60:  # Less than 1 minute for 1024x1024
            print(f"  ✅ Excellent performance: {avg_total_time:.1f}s per image")
        elif avg_total_time < 120:  # Less than 2 minutes
            print(f"  ✅ Good performance: {avg_total_time:.1f}s per image")
        elif avg_total_time < 180:  # Less than 3 minutes
            print(f"  ⚠️  Acceptable performance: {avg_total_time:.1f}s per image")
        else:
            print(f"  ❌ Performance may need optimization: {avg_total_time:.1f}s per image")

        # Memory and efficiency notes
        print(f"\nConfiguration Notes:")
        print(f"  Parallel Efficiency: {cfg_factor}x CFG, {sp_factor}x SP, {tp_factor}x TP")
        print(f"  Expected Speedup: ~{cfg_factor * max(sp_factor, tp_factor):.1f}x theoretical")
        if enable_t5_text_encoder:
            print(f"  T5 Text Encoder: Enabled (higher quality, slower encoding)")
        else:
            print(f"  T5 Text Encoder: Disabled (faster encoding, may affect quality)")

    print("=" * 80)

    # Validate performance
    measurements = {
        "clip_encoding_time": statistics.mean(clip_times),
        "t5_encoding_time": statistics.mean(t5_times),
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }

    expected_metrics = get_expected_metrics(mesh_device)
    if is_ci_env:
        # In CI, dump a performance report
        benchmark_data = BenchmarkData()
        for iteration in range(num_perf_runs):
            for step_name, target in zip(
                ["encoder", "denoising", "vae", "run"],
                [
                    None,
                    expected_metrics["denoising_steps_time"],
                    expected_metrics["vae_decoding_time"],
                    expected_metrics["total_time"],
                ],
            ):
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=target,
                )
        device_name_map = {
            (2, 4): "WH_T3K",
            (4, 8): "BH_GLX" if is_blackhole() else "WH_GLX",
        }
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map[tuple(mesh_device.shape)],
            ml_model_name="SD35",
            batch_size=1,
            config_params={
                "width": image_w,
                "height": image_h,
                "num_frames": 1,
                "num_steps": num_inference_steps,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
                "num_links": num_links,
                "fsdp": False,
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

    # Synchronize all devices
    for submesh_device in pipeline.submesh_devices:
        ttnn.synchronize_device(submesh_device)

    logger.info("Performance test completed successfully!")

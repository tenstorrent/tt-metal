# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from ....pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import TimingCollector


# TODO: Factor out commonalities with sd35
@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, num_inference_steps",
    [
        (1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(2, 4), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (4, 0), (8, 1), (4, 0), (4, 0), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4sp0tp1",
        "4x8sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 34000000}],
    indirect=True,
)
def test_flux1_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    sp,
    tp,
    encoder_tp,
    vae_tp,
    topology,
    num_links,
    model_location_generator,
    is_ci_env,
    galaxy_type,
    monkeypatch,
) -> None:
    """Performance test for Flux.1 pipeline with detailed timing analysis. We use the dev variant"""

    benchmark_profiler = BenchmarkProfiler()

    # use new hugging face cache location in CI. Remove after complete migration.
    if is_ci_env:
        monkeypatch.setenv("HF_HUB_CACHE", "/mnt/MLPerf/huggingface/hub")

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

    pipeline = Flux1Pipeline.create_pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-dev"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        topology=topology,
        num_links=num_links,
    )

    # Test prompts - diverse set for comprehensive performance testing
    prompts = [
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """A tiny, fluffy dragon curled up in a teacup, warm cozy lighting, big expressive eyes, intricate scale patterns, surrounded by books and potions — high detail, Studio Ghibli meets Pixar""",
        """An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic depth and warmth.""",
    ]

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")
    # timer_warmup = TimingCollector()
    timer = TimingCollector()
    pipeline.timing_collector = timer

    images = pipeline.run_single_prompt(
        width=image_w, height=image_h, prompt=prompts[0], num_inference_steps=num_inference_steps, seed=0, traced=True
    )
    images[0].save(f"flux1_dev_{image_w}_{image_h}_warmup.png")

    warmup_timing = timer.get_timing_data()
    logger.info(f"Warmup completed in {warmup_timing.total_time:.2f}s")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    all_timings = []
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
                images = pipeline.run_single_prompt(
                    width=image_w,
                    height=image_h,
                    prompt=prompts[prompt_idx],
                    num_inference_steps=num_inference_steps,
                    seed=0,
                    traced=True,
                )
            images[0].save(f"flux1_dev_{image_w}_{image_h}_perf_run{i}.png")

            # Collect timing data
            timing_data = timer.get_timing_data()
            all_timings.append(timing_data)
            logger.info(f"  Run {i+1} completed in {timing_data.total_time:.2f}s")

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # Calculate statistics
    clip_times = [t.clip_encoding_time for t in all_timings]
    t5_times = [t.t5_encoding_time for t in all_timings]
    total_encoding_times = [t.total_encoding_time for t in all_timings]
    vae_times = [t.vae_decoding_time for t in all_timings]
    total_times = [t.total_time for t in all_timings]

    # Calculate per-step denoising times
    all_denoising_steps = []
    for timing in all_timings:
        all_denoising_steps.extend(timing.denoising_step_times)

    # Report results
    sp_factor = sp[0]  # pipeline.dit_parallel_config.sequence_parallel.factor
    tp_factor = tp[0]  # pipeline.dit_parallel_config.tensor_parallel.factor
    encoder_tp_factor = encoder_tp[0]  # pipeline.encoder_parallel_config.tensor_parallel.factor
    vae_tp_factor = vae_tp[0]  # pipeline.vae_parallel_config.tensor_parallel.factor

    print("\n" + "=" * 80)
    print("FLUX.1 DEV PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: FLUX.1-dev")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: sp={sp_factor}, tp={tp_factor}, encoder_tp={encoder_tp_factor}, vae_tp={vae_tp_factor}")
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

    print("=" * 80)

    # Validate that we got reasonable results
    assert len(all_timings) == num_perf_runs, f"Expected {num_perf_runs} timing results, got {len(all_timings)}"
    assert all(t.total_time > 0 for t in all_timings), "All runs should have positive total time"
    assert all(
        len(t.denoising_step_times) == num_inference_steps for t in all_timings
    ), f"All runs should have {num_inference_steps} denoising steps"

    # Clean up
    pipeline.timing_collector = None

    # Validate performance
    measurements = {
        "clip_encoding_time": statistics.mean(clip_times),
        "t5_encoding_time": statistics.mean(t5_times),
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 4):
        expected_metrics = {
            "clip_encoding_time": 0.1,
            "t5_encoding_time": 0.25,
            "total_encoding_time": 0.3,
            "denoising_steps_time": 0.75 * num_inference_steps,
            "vae_decoding_time": 1.6,
            "total_time": 22,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        expected_metrics = {
            "clip_encoding_time": 0.1,
            "t5_encoding_time": 0.21,
            "total_encoding_time": 0.3,
            "denoising_steps_time": 0.3 * num_inference_steps,
            "vae_decoding_time": 1.3,
            "total_time": 8.2,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
        profiler_model_name = (
            f"flux1_dev_{'t3k' if tuple(mesh_device.shape) == (2, 4) else 'tg'}_sp{sp_factor}_tp{tp_factor}"
        )
        benchmark_data = BenchmarkData()
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="flux1_dev_traced",
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

    # Synchronize all devices
    pipeline.synchronize_devices()

    logger.info("Performance test completed successfully!")

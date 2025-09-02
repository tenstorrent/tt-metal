# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from loguru import logger

from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    create_pipeline,
    TimingCollector,
)


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
@pytest.mark.parametrize("use_cache", [True, False], ids=["yes_use_cache", "no_use_cache"])
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
    use_cache,
) -> None:
    """Performance test for new SD35 pipeline with detailed timing analysis."""

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Guidance scale: {guidance_scale}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    pipeline = create_pipeline(
        mesh_device=mesh_device,
        batch_size=1,
        image_w=image_w,
        image_h=image_h,
        guidance_scale=guidance_scale,
        cfg_config=cfg,
        sp_config=sp,
        tp_config=tp,
        num_links=num_links,
        model_checkpoint_path=model_location_generator(
            f"stabilityai/stable-diffusion-3.5-{model_name}", model_subdir="StableDiffusion_35_Large"
        ),
        use_cache=use_cache,
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

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")
    timer_warmup = TimingCollector()
    pipeline.timing_collector = timer_warmup

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

    warmup_timing = timer_warmup.get_timing_data()
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

            # Create timing collector for this run
            timer = TimingCollector()
            pipeline.timing_collector = timer

            # Run pipeline with different prompt
            prompt_idx = (i + 1) % len(prompts)
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
            )
            images[0].save(f"sd35_new_{image_w}_{image_h}_perf_run{i}.png")

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

    # Validate that we got reasonable results
    assert len(all_timings) == num_perf_runs, f"Expected {num_perf_runs} timing results, got {len(all_timings)}"
    assert all(t.total_time > 0 for t in all_timings), "All runs should have positive total time"
    assert all(
        len(t.denoising_step_times) == num_inference_steps for t in all_timings
    ), f"All runs should have {num_inference_steps} denoising steps"

    # Clean up
    pipeline.timing_collector = None

    # Synchronize all devices
    for submesh_device in pipeline.submesh_devices:
        ttnn.synchronize_device(submesh_device)

    logger.info("Performance test completed successfully!")

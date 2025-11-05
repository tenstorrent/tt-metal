# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
from loguru import logger
import ttnn
from ..tt.fun_pipeline import TtStableDiffusion3Pipeline, TimingCollector
from ..tt.parallel_config import StableDiffusionParallelManager, EncoderParallelManager, create_vae_parallel_config
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData


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
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
def test_sd35_performance(
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
    """Performance test for SD35 pipeline with detailed timing analysis."""

    if galaxy_type == "4U":
        # NOTE: Pipelines fail if a performance test is skipped without providing a benchmark output.
        if is_ci_env:
            profiler = BenchmarkProfiler()
            with profiler("run", iteration=0):
                pass

            benchmark_data = BenchmarkData()
            benchmark_data.save_partial_run_json(
                profiler,
                run_type="empty_run",
                ml_model_name="empty_run",
            )
        pytest.skip("4U is not supported for this test")

    # Setup parallel manager
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )

    # HACK: reshape submesh device 0 from 2D to 1D
    encoder_device = parallel_manager.submesh_devices[0]
    if parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape[1] != 4:
        # If reshaping, vae_device must be on submesh 0. That means T5 can't fit, so disable it.
        vae_device = parallel_manager.submesh_devices[0]
        enable_t5_text_encoder = False

        cfg_shape = parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape
        assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
        print(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP + T5")
        encoder_device.reshape(ttnn.MeshShape(1, 4))
    else:
        # vae_device can only be on submesh 1 if submesh is not getting reshaped.
        vae_device = parallel_manager.submesh_devices[1]
        enable_t5_text_encoder = True

    print(f"T5 enabled: {enable_t5_text_encoder}")

    encoder_parallel_manager = EncoderParallelManager(
        encoder_device,
        topology,
        mesh_axis=1,  # 1x4 submesh, parallel on axis 1
        num_links=num_links,
    )
    vae_parallel_manager = create_vae_parallel_config(vae_device, parallel_manager)
    # HACK: reshape submesh device 0 from 1D to 2D
    encoder_device.reshape(ttnn.MeshShape(*parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape))

    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    # Create pipeline
    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        guidance_cond=guidance_cond,
        parallel_manager=parallel_manager,
        encoder_parallel_manager=encoder_parallel_manager,
        vae_parallel_manager=vae_parallel_manager,
        height=image_h,
        width=image_w,
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    # Test prompts
    # prompts = (
    #     "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
    #     "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
    #     "snow-covered pines and delicate falling snowflakes - captured in a rich, "
    #     "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    # )
    prompts = [
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """A tiny, fluffy dragon curled up in a teacup, warm cozy lighting, big expressive eyes, intricate scale patterns, surrounded by books and potions — high detail, Studio Ghibli meets Pixar""",
    ]
    negative_prompt = ""

    # Warmup run (not timed)
    print("Running warmup iteration...")
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
    images[0].save(f"sd35_{image_w}_{image_h}_warmup.png")

    # Performance measurement runs
    print("Running performance measurement iterations...")
    all_timings = []
    profiler = BenchmarkProfiler()

    for i in range(3):
        print(f"Performance run {i+1}/3...")

        # Create timing collector for this run
        timer = TimingCollector()
        pipeline.timing_collector = timer

        # Run pipeline
        with profiler("run", iteration=i):
            images = pipeline(
                prompt_1=[prompts[i + 1]],
                prompt_2=[prompts[i + 1]],
                prompt_3=[prompts[i + 1]],
                negative_prompt_1=[negative_prompt],
                negative_prompt_2=[negative_prompt],
                negative_prompt_3=[negative_prompt],
                num_inference_steps=num_inference_steps,
                seed=0,
                traced=True,
            )
        images[0].save(f"sd35_{image_w}_{image_h}_run{i}.png")
        # Collect timing data
        timing_data = timer.get_timing_data()
        all_timings.append(timing_data)

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
    print("\n" + "=" * 80)
    print("STABLE DIFFUSION 3.5 PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: cfg={cfg_factor}, sp={sp_factor}, tp={tp_factor}")
    print("-" * 80)

    def print_stats(name, times):
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
            "clip_encoding_time": 0.09,
            "t5_encoding_time": 0.1,
            "total_encoding_time": 0.2,
            "denoising_steps_time": 11,
            "vae_decoding_time": 1.6,
            "total_time": 12.6,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        expected_metrics = {
            "clip_encoding_time": 0.17,
            "t5_encoding_time": 0.13,
            "total_encoding_time": 0.6,
            "denoising_steps_time": 4,
            "vae_decoding_time": 1.65,
            "total_time": 6.2,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
        profiler_model_name = (
            f"sd35_{'t3k' if tuple(mesh_device.shape) == (2, 4) else 'tg'}_cfg{cfg_factor}_sp{sp_factor}_tp{tp_factor}"
        )
        benchmark_data = BenchmarkData()
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="sd35_traced",
            ml_model_name=profiler_model_name,
        )

    pass_perf_check = True
    for k in expected_metrics.keys():
        if measurements[k] > expected_metrics[k]:
            logger.warning(
                f"Warning: {k} is outside of the tolerance range. Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False
    if pass_perf_check:
        logger.info("Perf check passed!")
    else:
        logger.warning("Perf check failed!")
        assert False, "Perf check failed!"

    for submesh_device in parallel_manager.submesh_devices:
        ttnn.synchronize_device(submesh_device)

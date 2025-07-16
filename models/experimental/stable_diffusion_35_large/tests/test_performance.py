# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from ..tt.fun_pipeline import TtStableDiffusion3Pipeline, TimingCollector
from ..tt.parallel_config import StableDiffusionParallelManager


@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("large", 1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 20000000}],
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
) -> None:
    """Performance test for SD35 pipeline with detailed timing analysis."""

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

    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    # Create pipeline
    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=False,
        guidance_cond=guidance_cond,
        parallel_manager=parallel_manager,
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
    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )
    negative_prompt = ""

    # Warmup run (not timed)
    print("Running warmup iteration...")
    pipeline(
        prompt_1=[prompt],
        prompt_2=[prompt],
        prompt_3=[prompt],
        negative_prompt_1=[negative_prompt],
        negative_prompt_2=[negative_prompt],
        negative_prompt_3=[negative_prompt],
        num_inference_steps=num_inference_steps,
        seed=0,
    )

    # Performance measurement runs
    print("Running performance measurement iterations...")
    all_timings = []

    from tracy import Profiler

    profiler = Profiler()

    profiler.enable()
    for i in range(3):
        print(f"Performance run {i+1}/3...")

        # Create timing collector for this run
        timer = TimingCollector()
        pipeline.timing_collector = timer

        # Run pipeline
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
        )

        # Collect timing data
        timing_data = timer.get_timing_data()
        all_timings.append(timing_data)

    profiler.disable()
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

    for submesh_device in parallel_manager.submesh_devices:
        ttnn.synchronize_device(submesh_device)

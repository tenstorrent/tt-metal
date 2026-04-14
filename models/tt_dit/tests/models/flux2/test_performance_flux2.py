# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import statistics

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline


@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, num_inference_steps",
    [
        (1024, 1024, 4.0, 28),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp, tp, vae_tp, topology, num_links",
    [
        [(1, 2), (1, 0), (2, 1), (2, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 0), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 0), (4, 1), (4, 1), ttnn.Topology.Linear, 2],
        [(2, 2), (2, 0), (2, 1), (2, 1), ttnn.Topology.Linear, 2],
    ],
    ids=[
        "wh_1x2sp0tp1",
        "wh_2x4sp0tp1",
        "bh_2x4sp0tp1",
        "2x2sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
def test_flux2_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    sp,
    tp,
    vae_tp,
    topology,
    num_links,
    model_location_generator,
    is_ci_env,
    galaxy_type,
) -> None:
    benchmark_profiler = BenchmarkProfiler()

    if galaxy_type == "4U":
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

    pipeline = Flux2Pipeline.create_pipeline(
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        vae_tp=vae_tp,
        topology=topology,
        num_links=num_links,
    )

    prompts = [
        "A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms",
        "A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes",
        "A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind",
        "A tiny, fluffy dragon curled up in a teacup, warm cozy lighting, big expressive eyes",
        "An epic cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape",
    ]

    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        images = pipeline.run_single_prompt(
            width=image_w,
            height=image_h,
            prompt=prompts[0],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )
    images[0].save(f"flux2_dev_{image_w}_{image_h}_warmup.png")
    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    logger.info("Running performance measurement iterations...")
    num_perf_runs = 4

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
            prompt_idx = (i + 1) % len(prompts)
            with benchmark_profiler("run", iteration=i):
                images = pipeline.run_single_prompt(
                    width=image_w,
                    height=image_h,
                    prompt=prompts[prompt_idx],
                    num_inference_steps=num_inference_steps,
                    seed=0,
                    traced=True,
                    profiler=benchmark_profiler,
                    profiler_iteration=i,
                )
            images[0].save(f"flux2_dev_{image_w}_{image_h}_perf_run{i}.png")
            logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")
    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    total_encoding_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    all_denoising_steps = []
    for i in range(num_perf_runs):
        for j in range(num_inference_steps):
            assert benchmark_profiler.contains_step(
                f"denoising_step_{j}", i
            ), f"All runs should have {num_inference_steps} denoising steps"
            all_denoising_steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", i))

    sp_factor = sp[0]
    tp_factor = tp[0]
    vae_tp_factor = vae_tp[0]

    print("\n" + "=" * 80)
    print("FLUX.2 DEV PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: FLUX.2-dev")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: sp={sp_factor}, tp={tp_factor}, vae_tp={vae_tp_factor}")
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

    print_stats("Total Encoding", total_encoding_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)
    print("-" * 80)

    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * num_inference_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {num_inference_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} images/second")

        avg_encoding_time = statistics.mean(total_encoding_times)
        avg_vae_time = statistics.mean(vae_times)

        print(f"\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE: {avg_vae_time/avg_total_time*100:.1f}%")

    print("=" * 80)

    # Validate performance against expected thresholds.
    # Flux2 uses Mistral3 on CPU for text encoding instead of CLIP+T5, so encoding
    # metrics are a single "total_encoding_time" bucket.
    avg_step_time = statistics.mean(all_denoising_steps)
    total_denoising_time = avg_step_time * num_inference_steps

    measurements = {
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }

    if tuple(mesh_device.shape) == (1, 2) and not is_blackhole():
        expected_metrics = {
            "total_encoding_time": 1.0,
            "denoising_steps_time": 1.1 * num_inference_steps,
            "vae_decoding_time": 2.5,
            "total_time": 37.0,
        }
    elif tuple(mesh_device.shape) == (2, 4) and is_blackhole():
        expected_metrics = {
            "total_encoding_time": 0.5,
            "denoising_steps_time": 0.35 * num_inference_steps,
            "vae_decoding_time": 1.0,
            "total_time": 12.0,
        }
    elif tuple(mesh_device.shape) == (2, 4):
        expected_metrics = {
            "total_encoding_time": 0.5,
            "denoising_steps_time": 0.65 * num_inference_steps,
            "vae_decoding_time": 1.3,
            "total_time": 22.0,
        }
    elif tuple(mesh_device.shape) == (2, 2):
        assert is_blackhole(), "2x2 is only supported for blackhole"
        expected_metrics = {
            "total_encoding_time": 0.5,
            "denoising_steps_time": 0.50 * num_inference_steps,
            "vae_decoding_time": 1.3,
            "total_time": 17.0,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
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
            (1, 2): "WH_T3K",
            (2, 2): "BH_QB",
            (2, 4): "BH_LB" if is_blackhole() else "WH_T3K",
        }
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map[tuple(mesh_device.shape)],
            ml_model_name="Flux2Dev",
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

    pipeline.synchronize_devices()
    logger.info("Performance test completed successfully!")


@pytest.mark.parametrize(
    "image_w, image_h, guidance_scale, num_inference_steps",
    [
        (1024, 1024, 4.0, 4),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, sp, tp, vae_tp, topology, num_links",
    [
        [(2, 4), (2, 0), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
    ],
    ids=[
        "wh_2x4sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
def test_flux2_denoising_step_breakdown(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    sp,
    tp,
    vae_tp,
    topology,
    num_links,
    model_location_generator,
    is_ci_env,
) -> None:
    """Profile per-step timing breakdown to identify optimization targets."""
    benchmark_profiler = BenchmarkProfiler()

    pipeline = Flux2Pipeline.create_pipeline(
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        vae_tp=vae_tp,
        topology=topology,
        num_links=num_links,
    )

    prompt = "A neon-lit alley in a sprawling cyberpunk metropolis at night"

    logger.info("running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        pipeline.run_single_prompt(
            width=image_w,
            height=image_h,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )

    logger.info("running profiled iteration...")
    with benchmark_profiler("run", iteration=1):
        pipeline.run_single_prompt(
            width=image_w,
            height=image_h,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
            profiler=benchmark_profiler,
            profiler_iteration=1,
        )

    print("\n" + "=" * 80)
    print("FLUX.2 DENOISING STEP BREAKDOWN")
    print("=" * 80)
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print("-" * 80)

    for step_i in range(num_inference_steps):
        step_key = f"denoising_step_{step_i}"
        if benchmark_profiler.contains_step(step_key, 1):
            step_time = benchmark_profiler.get_duration(step_key, 1)
            print(f"  Step {step_i:3d}: {step_time:.4f}s")

    encoding_time = benchmark_profiler.get_duration("encoder", 1)
    denoising_time = benchmark_profiler.get_duration("denoising", 1)
    vae_time = benchmark_profiler.get_duration("vae", 1)
    total_time = benchmark_profiler.get_duration("run", 1)

    print("-" * 80)
    print(f"  Encoding:   {encoding_time:.4f}s  ({encoding_time/total_time*100:.1f}%)")
    print(f"  Denoising:  {denoising_time:.4f}s  ({denoising_time/total_time*100:.1f}%)")
    print(f"  VAE:        {vae_time:.4f}s  ({vae_time/total_time*100:.1f}%)")
    print(f"  Total:      {total_time:.4f}s")
    print("=" * 80)

    pipeline.synchronize_devices()

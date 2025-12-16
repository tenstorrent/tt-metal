# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import statistics
import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData

from ....pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "width, height, num_inference_steps",
    [
        (1024, 1024, 50),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        # 8-chip t3k/loudbox config, uncomment when running on 8-chip systems
        # note: this config is not valid on 6u (32-chip) systems with fabric enabled
        # [(1, 8), (1, 0), (1, 0), (8, 1), (8, 1), (8, 1), ttnn.Topology.Linear, 1],
        # t3k
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        # 6u
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        # "1x8tp1",
        "2x4cfg0sp0tp1",
        "4x8cfg1sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 40000000}],
    indirect=True,
)
@pytest.mark.parametrize("use_cache", [True, False], ids=["yes_use_cache", "no_use_cache"])
def test_qwenimage_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    use_cache: bool,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """performance test for qwenimage pipeline with detailed timing analysis."""

    benchmark_profiler = BenchmarkProfiler()

    # process skips
    if is_ci_env and use_cache:
        pytest.skip("use_cache not necessary for performance test in CI. See pipeline test.")

    # skip 4u
    if galaxy_type == "4U":
        # pipelines fail if a performance test is skipped without providing a benchmark output
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

    logger.info(f"  Image size: {width}x{height}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=False,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
    )

    # test prompts, diverse set for comprehensive performance testing
    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition',
        'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
        'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
        'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
        'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
    ]

    # warmup run, not timed
    logger.info("Running warmup iteration...")
    timer_warmup = TimingCollector()
    pipeline.timing_collector = timer_warmup

    images = pipeline(
        prompts=[prompts[0]],
        negative_prompts=[None],
        num_inference_steps=num_inference_steps,
        cfg_scale=4.0,
        seed=0,
        traced=True,
    )
    images[0].save(f"qwenimage_{width}_{height}_warmup.png")

    warmup_timing = timer_warmup.get_timing_data()
    logger.info(f"Warmup completed in {warmup_timing.total_time:.2f}s")

    # performance measurement runs
    logger.info("Running performance measurement iterations...")
    all_timings = []
    num_perf_runs = 4  # use 4 different prompts for performance testing

    # optional tracy profiling if available
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

            # create timing collector for this run
            timer = TimingCollector()
            pipeline.timing_collector = timer

            # run pipeline with different prompt
            prompt_idx = (i + 1) % len(prompts)
            with benchmark_profiler("run", iteration=i):
                images = pipeline(
                    prompts=[prompts[prompt_idx]],
                    negative_prompts=[None],
                    num_inference_steps=num_inference_steps,
                    cfg_scale=4.0,
                    seed=i,
                    traced=True,
                )
            images[0].save(f"qwenimage_{width}_{height}_perf_run{i}.png")

            # collect timing data
            timing_data = timer.get_timing_data()
            all_timings.append(timing_data)
            logger.info(f"  Run {i+1} completed in {timing_data.total_time:.2f}s")

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # calculate statistics
    clip_times = [t.clip_encoding_time for t in all_timings]
    t5_times = [t.t5_encoding_time for t in all_timings]
    total_encoding_times = [t.total_encoding_time for t in all_timings]
    vae_times = [t.vae_decoding_time for t in all_timings]
    total_times = [t.total_time for t in all_timings]

    # calculate per-step denoising times
    all_denoising_steps = []
    for timing in all_timings:
        all_denoising_steps.extend(timing.denoising_step_times)

    # report results
    cfg_factor = cfg[0] if cfg[1] == 0 else cfg[1]
    sp_factor = sp[0] if sp[1] == 0 else sp[1]
    tp_factor = tp[0] if tp[1] == 0 else tp[1]

    print("\n" + "=" * 80)
    print("QWEN IMAGE PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: cfg={cfg_factor}, sp={sp_factor}, tp={tp_factor}")
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

    # additional metrics
    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * num_inference_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {num_inference_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} images/second")

        # breakdown percentages
        avg_encoding_time = statistics.mean(total_encoding_times)
        avg_vae_time = statistics.mean(vae_times)

        print(f"\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE: {avg_vae_time/avg_total_time*100:.1f}%")

        # performance benchmarks
        print(f"\nPerformance Analysis:")
        if avg_total_time < 60:  # less than 1 minute for 1024x1024
            print(f"  Excellent performance: {avg_total_time:.1f}s per image")
        elif avg_total_time < 120:  # less than 2 minutes
            print(f"  Good performance: {avg_total_time:.1f}s per image")
        elif avg_total_time < 180:  # less than 3 minutes
            print(f"  Acceptable performance: {avg_total_time:.1f}s per image")
        else:
            print(f"  Performance may need optimization: {avg_total_time:.1f}s per image")

        # configuration notes
        print(f"\nConfiguration Notes:")
        print(f"  Parallel Efficiency: {cfg_factor}x CFG, {sp_factor}x SP, {tp_factor}x TP")
        print(f"  Expected Speedup: ~{cfg_factor * max(sp_factor, tp_factor):.1f}x theoretical")

    print("=" * 80)

    # validate that we got reasonable results
    assert len(all_timings) == num_perf_runs, f"Expected {num_perf_runs} timing results, got {len(all_timings)}"
    assert all(t.total_time > 0 for t in all_timings), "All runs should have positive total time"
    assert all(
        len(t.denoising_step_times) == num_inference_steps for t in all_timings
    ), f"All runs should have {num_inference_steps} denoising steps"

    # clean up
    pipeline.timing_collector = None

    # validate performance
    measurements = {
        "clip_encoding_time": statistics.mean(clip_times),
        "t5_encoding_time": statistics.mean(t5_times),
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }

    # expected metrics, these will need to be updated based on actual performance measurements
    if tuple(mesh_device.shape) == (2, 4):
        expected_metrics = {
            "clip_encoding_time": 0.2,
            "t5_encoding_time": 0.15,
            "total_encoding_time": 0.35,
            "denoising_steps_time": 15.0,
            "vae_decoding_time": 2.0,
            "total_time": 17.5,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        expected_metrics = {
            "clip_encoding_time": 0.25,
            "t5_encoding_time": 0.18,
            "total_encoding_time": 0.7,
            "denoising_steps_time": 6.0,
            "vae_decoding_time": 1.5,
            "total_time": 8.5,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # in ci, dump a performance report
        profiler_model_name = (
            f"qwenimage_{'t3k' if tuple(mesh_device.shape) == (2, 4) else 'tg'}_cfg{cfg_factor}_sp{sp_factor}_tp{tp_factor}"
        )
        benchmark_data = BenchmarkData()
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="qwenimage_traced",
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

    # synchronize all devices
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)

    logger.info("Performance test completed successfully!")


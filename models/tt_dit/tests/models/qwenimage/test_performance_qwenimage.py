# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import statistics

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

from .device_params import MESH_DEVICE_PARAMS


@pytest.mark.parametrize(
    ("image_w", "image_h", "num_inference_steps"),
    [
        (1024, 1024, 50),
    ],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w: int,
    image_h: int,
    num_inference_steps: int,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Performance test for QwenImage pipeline with detailed timing analysis."""
    benchmark_profiler = BenchmarkProfiler()

    # Set TT_DIT_CACHE in CI environment
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Inference steps: {num_inference_steps}")

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)
    # The configuration create_pipeline uses, for the report.
    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape)
    cfg = config.dit_parallel_config.cfg_parallel.factor
    sp = config.dit_parallel_config.sequence_parallel.factor
    tp = config.dit_parallel_config.tensor_parallel.factor
    encoder_tp = config.encoder_parallel_config.tensor_parallel.factor

    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition',
        'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
        'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
        'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
        'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
    ]

    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        images = pipeline(
            prompts=[prompts[0]],
            num_inference_steps=num_inference_steps,
            width=image_w,
            height=image_h,
            traced=True,
        )
    images[0].save(f"qwenimage_{image_w}_{image_h}_warmup.png")

    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    logger.info("Running performance measurement iterations...")
    num_perf_runs = 2  # len(prompts)

    for i in range(num_perf_runs):
        logger.info(f"Performance run {i+1}/{num_perf_runs}...")

        # run pipeline with different prompt
        prompt_idx = (i + 1) % len(prompts)

        with benchmark_profiler("run", iteration=i):
            images = pipeline(
                prompts=[prompts[prompt_idx]],
                num_inference_steps=num_inference_steps,
                width=image_w,
                height=image_h,
                traced=True,
                on_event=profiler_event_callback(benchmark_profiler, i),
            )

        logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

        images[0].save(f"qwenimage_{image_w}_{image_h}_perf_run{i}.png")

    total_encoding_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    # calculate per-step denoising times
    all_denoising_steps = []
    for i in range(num_perf_runs):
        for j in range(num_inference_steps):
            assert benchmark_profiler.contains_step(
                f"denoising_step_{j}", i
            ), f"All runs should have {num_inference_steps} denoising steps"
            all_denoising_steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", i))

    print("\n" + "=" * 100)
    print("QWEN IMAGE PIPELINE PERFORMANCE RESULTS")
    print("=" * 100)
    print("Model: QwenImage")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Configuration: cfg={cfg}, sp={sp}, tp={tp}, encoder_tp={encoder_tp}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {config.topology}")
    print("-" * 100)

    def print_stats(name: str, times: list[float]) -> None:
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

    print_stats("Total Encoding", total_encoding_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)

    print("-" * 100)

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

        print("\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE: {avg_vae_time/avg_total_time*100:.1f}%")

    # validate performance
    avg_step_time = statistics.mean(all_denoising_steps) if all_denoising_steps else 0
    total_denoising_time = avg_step_time * num_inference_steps
    measurements = {
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 2):
        # BH QuietBox 2, measured on a local QB2:
        # encoding 0.097s, denoising 27.2s, vae 0.111s, total 27.5s.
        expected_metrics = {
            "total_encoding_time": 0.12,
            "denoising_steps_time": 30.0,
            "vae_decoding_time": 0.15,
            "total_time": 31,
        }
    elif tuple(mesh_device.shape) == (2, 4):
        # WH T3K: encoding 0.552s, denoising 67.2s, vae 0.138s, total 68.0s.
        expected_metrics = {
            "total_encoding_time": 0.6,
            "denoising_steps_time": 74.0,
            "vae_decoding_time": 0.2,
            "total_time": 75,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        expected_metrics = {
            "total_encoding_time": 0.35,
            "denoising_steps_time": 25,
            "vae_decoding_time": 0.4,
            "total_time": 26,
        }
    else:
        pytest.fail(f"Unknown mesh device for performance comparison: {mesh_device}")

    if is_ci_env:
        # in ci, dump a performance report
        benchmark_data = BenchmarkData()
        for iteration in range(num_perf_runs):
            for step_name, target in zip(
                ["encoder", "denoising", "vae", "run"],
                [
                    expected_metrics["total_encoding_time"],
                    expected_metrics["denoising_steps_time"],
                    expected_metrics["vae_decoding_time"],
                    expected_metrics["total_time"],
                ],
                strict=True,
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
            (2, 2): "BH_QB",
            (2, 4): "WH_T3K",
            (4, 8): "BH_GLX" if is_blackhole() else "WH_GLX",
        }
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map[tuple(mesh_device.shape)],
            ml_model_name="QwenImage",
            batch_size=1,
            config_params={
                "width": image_w,
                "height": image_h,
                "num_steps": num_inference_steps,
                "cfg_factor": cfg,
                "sp_factor": sp,
                "tp_factor": tp,
                "num_frames": 1,
                "encoder_tp_factor": encoder_tp,
                "topology": str(config.topology),
                "num_links": config.num_links,
            },
        )

    pass_perf_check = True
    assert_msgs = []
    for k in expected_metrics:
        if measurements[k] > expected_metrics[k]:
            assert_msgs.append(
                f"Warning: {k} is outside of the tolerance range. "
                f"Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False

    assert pass_perf_check, "\n".join(assert_msgs)

    # synchronize all devices
    pipeline.synchronize_devices()
    logger.info("Performance test completed successfully!")

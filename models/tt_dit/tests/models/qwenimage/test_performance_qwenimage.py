# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.pipelines.events import log_section_durations, profiler_event_callback
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

from .device_params import MESH_DEVICE_PARAMS

NUM_INFERENCE_STEPS = 50
RUNS_PER_RESOLUTION = 2

# Upper limits at 1024², in seconds, per (is Blackhole, mesh shape). Only the listed metrics are
# checked.
EXPECTED_METRICS = {
    # BH QuietBox 2, measured on a local QB2: encoding 0.097s, denoising 27.2s, vae 0.111s, total 27.5s.
    (True, (2, 2)): {
        "total_encoding_time": 0.12,
        "denoising_steps_time": 30.0,
        "vae_decoding_time": 0.15,
        "total_time": 31,
    },
    # WH T3K: encoding 0.552s, denoising 67.2s, vae 0.138s, total 68.0s.
    (False, (2, 4)): {
        "total_encoding_time": 0.6,
        "denoising_steps_time": 74.0,
        "vae_decoding_time": 0.2,
        "total_time": 75,
    },
    # BH LoudBox: total 19.0s (CI run 38231039779).
    (True, (2, 4)): {"total_time": 21},
    # WH Galaxy: total 14.7–15.0s (CI runs 38337794894, 38364045034).
    (False, (4, 8)): {"total_time": 17},
    # BH Galaxy: total 7.7s (CI run 38231028525).
    (True, (4, 8)): {"total_time": 9},
}

RUN_TYPES = {
    (True, (2, 2)): "BH_QB",
    (False, (2, 4)): "WH_T3K",
    (True, (2, 4)): "BH_LB",
    (False, (4, 8)): "WH_GLX",
    (True, (4, 8)): "BH_GLX",
}

PROMPTS = [
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," with a neon light '
    'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
    'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition',
    'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
    'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
    'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
    'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
]


@pytest.mark.parametrize(
    "all_resolutions",
    [
        pytest.param(False, id="1024x1024"),
        # Off in CI, where it would lengthen the job; set QWENIMAGE_PERF_ALL_RESOLUTIONS=1 to run it.
        pytest.param(
            True,
            id="all_resolutions",
            marks=pytest.mark.skipif(
                os.environ.get("QWENIMAGE_PERF_ALL_RESOLUTIONS") != "1",
                reason="set QWENIMAGE_PERF_ALL_RESOLUTIONS=1",
            ),
        ),
    ],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    all_resolutions: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    key = (is_blackhole(), tuple(mesh_device.shape))
    if key not in EXPECTED_METRICS:
        pytest.fail(f"Unknown mesh device for performance comparison: {mesh_device}")

    # The pipeline warms up every resolution of its preset while it is constructed.
    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)
    # The configuration create_pipeline uses, for the report.
    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape)
    cfg = config.dit_parallel_config.cfg_parallel.factor
    sp = config.dit_parallel_config.sequence_parallel.factor
    tp = config.dit_parallel_config.tensor_parallel.factor
    encoder_tp = config.encoder_parallel_config.tensor_parallel.factor

    resolutions = list(config.resolutions) if all_resolutions else [(1024, 1024)]

    benchmark_profiler = BenchmarkProfiler()
    # Iterations of each resolution, in the profiler's numbering.
    iterations = {}
    for r, (width, height) in enumerate(resolutions):
        iterations[width, height] = []
        for i in range(RUNS_PER_RESOLUTION):
            iteration = r * RUNS_PER_RESOLUTION + i
            iterations[width, height].append(iteration)
            prompt = PROMPTS[iteration % len(PROMPTS)]

            with benchmark_profiler("run", iteration=iteration):
                images = pipeline(
                    prompts=[prompt],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    width=width,
                    height=height,
                    traced=True,
                    on_event=profiler_event_callback(benchmark_profiler, iteration),
                )

            logger.info(f"{width}x{height} run {i + 1}: {benchmark_profiler.get_duration('run', iteration):.2f}s")
            log_section_durations(benchmark_profiler, iteration, per_step={"denoising": NUM_INFERENCE_STEPS})
            images[0].save(f"qwenimage_{width}_{height}_perf_run{i}.png")

    def durations(name: str, its: list[int]) -> list[float]:
        return [benchmark_profiler.get_duration(name, it) for it in its]

    def step_durations(its: list[int]) -> list[float]:
        steps = []
        for it in its:
            for j in range(NUM_INFERENCE_STEPS):
                assert benchmark_profiler.contains_step(
                    f"denoising_step_{j}", it
                ), f"All runs should have {NUM_INFERENCE_STEPS} denoising steps"
                steps.append(benchmark_profiler.get_duration(f"denoising_step_{j}", it))
        return steps

    def print_stats(name: str, times: list[float]) -> None:
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min(times):8.4f}s | Max: {max(times):8.4f}s"
        )

    print("\n" + "=" * 100)
    print("QWEN IMAGE PIPELINE PERFORMANCE RESULTS")
    print("=" * 100)
    print("Model: QwenImage")
    print(f"Inference Steps: {NUM_INFERENCE_STEPS}, runs per resolution: {RUNS_PER_RESOLUTION}")
    print(f"Configuration: cfg={cfg}, sp={sp}, tp={tp}, encoder_tp={encoder_tp}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {config.topology}, links: {config.num_links}")

    for (width, height), its in iterations.items():
        encoding = durations("encoder", its)
        text_encoder = durations("qwen_encoding", its)
        steps = step_durations(its)
        denoising = durations("denoising", its)
        vae = durations("vae", its)
        total = durations("run", its)
        # Everything outside the three stages: latent preparation, image conversion, host overhead.
        other = [t - e - d - v for t, e, d, v in zip(total, encoding, denoising, vae, strict=True)]

        print("-" * 100)
        print(f"Image Size: {width}x{height}")
        print_stats("Total Encoding", encoding)
        print_stats("  Text Encoder", text_encoder)
        print_stats("Denoising (per step)", steps)
        print_stats("Denoising (total)", denoising)
        print_stats("VAE Decoding", vae)
        print_stats("Other", other)
        print_stats("Total Pipeline", total)
        print(f"Denoising throughput: {NUM_INFERENCE_STEPS / statistics.mean(denoising):.2f} steps/second")
        print(f"Overall throughput: {1 / statistics.mean(total):.4f} images/second")

    print("=" * 100)

    its_1024 = iterations[1024, 1024]
    measurements = {
        "total_encoding_time": statistics.mean(durations("encoder", its_1024)),
        "denoising_steps_time": statistics.mean(step_durations(its_1024)) * NUM_INFERENCE_STEPS,
        "vae_decoding_time": statistics.mean(durations("vae", its_1024)),
        "total_time": statistics.mean(durations("run", its_1024)),
    }
    expected_metrics = EXPECTED_METRICS[key]

    if is_ci_env:
        # in ci, dump a performance report
        benchmark_data = BenchmarkData()
        step_names = {
            "total_encoding_time": "encoder",
            "denoising_steps_time": "denoising",
            "vae_decoding_time": "vae",
            "total_time": "run",
        }
        for iteration in its_1024:
            for metric, step_name in step_names.items():
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=expected_metrics.get(metric),
                )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=RUN_TYPES[key],
            ml_model_name="QwenImage",
            batch_size=1,
            config_params={
                "width": 1024,
                "height": 1024,
                "num_steps": NUM_INFERENCE_STEPS,
                "cfg_factor": cfg,
                "sp_factor": sp,
                "tp_factor": tp,
                "num_frames": 1,
                "encoder_tp_factor": encoder_tp,
                "topology": str(config.topology),
                "num_links": config.num_links,
            },
        )

    assert_msgs = [
        f"Warning: {k} is outside of the tolerance range. Expected: {limit}, Actual: {measurements[k]}"
        for k, limit in expected_metrics.items()
        if measurements[k] > limit
    ]
    assert not assert_msgs, "\n".join(assert_msgs)

    pipeline.synchronize_devices()
    logger.info("Performance test completed successfully!")

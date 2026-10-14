# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics
from dataclasses import dataclass, field
from typing import Any

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.pipelines.events import log_section_durations, profiler_event_callback
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig

from .device_params import EDIT_MESH_DEVICE_PARAMS, MESH_DEVICE_PARAMS
from .test_pipeline_qwenimage import DOG_IMAGE_PATH

NUM_INFERENCE_STEPS = 50
# The Qwen-Image-Edit-2511 model card's step count.
EDIT_NUM_INFERENCE_STEPS = 40
# Runs per case in CI's 1024² test, and in the opt-in benchmarks.
CI_RUNS_PER_CASE = 2
BENCHMARK_RUNS_PER_CASE = 3

# The opt-in benchmark's sizes: the ones of our NVIDIA benchmark (1344² for its 1328²). A transposed
# size takes the same time, and the 4:3 and 3:2 sizes take about as long as 16:9.
BENCHMARK_RESOLUTIONS = [(1024, 1024), (1344, 1344), (1664, 928)]

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

# The prompts of the Qwen-Image-2512 and Qwen-Image-Edit-2511 model cards, as in our NVIDIA benchmark.
PROMPT = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and "
    "lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in "
    "twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute "
    "dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime "
    "convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged "
    "lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, "
    "fresh, youthful charm."
)
EDIT_PROMPT = (
    "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park "
    "square."
)

# Input aspect ratios for editing, (width, height); the edit output takes the last input's.
EDIT_ASPECT_RATIOS = {"1:1": (1, 1), "16:9": (16, 9)}


@dataclass
class Case:
    label: str
    num_inference_steps: int
    kwargs: dict[str, Any] = field(default_factory=dict)


def crop_to_aspect_ratio(image: Image.Image, aspect: tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width = min(width, height * aspect[0] // aspect[1])
    target_height = min(height, width * aspect[1] // aspect[0])
    left, top = (width - target_width) // 2, (height - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def inpaint_inputs() -> dict[str, Any]:
    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    # Repaints the dog, with a margin around it.
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (130, 20, 430, 425))
    return {"image": image, "mask_image": mask, "strength": 1.0}


def run_cases(
    pipeline: QwenImagePipeline, cases: list[Case], *, prompt: str, file_prefix: str, runs_per_case: int
) -> tuple[BenchmarkProfiler, dict[str, list[int]]]:
    """Runs each case ``runs_per_case`` times and returns the profiler and each case's iterations."""
    benchmark_profiler = BenchmarkProfiler()
    iterations = {}
    for c, case in enumerate(cases):
        iterations[case.label] = []
        for i in range(runs_per_case):
            iteration = c * runs_per_case + i
            iterations[case.label].append(iteration)

            with benchmark_profiler("run", iteration=iteration):
                images = pipeline(
                    prompts=[prompt],
                    num_inference_steps=case.num_inference_steps,
                    traced=True,
                    on_event=profiler_event_callback(benchmark_profiler, iteration),
                    **case.kwargs,
                )

            logger.info(f"{case.label} run {i + 1}: {benchmark_profiler.get_duration('run', iteration):.2f}s")
            log_section_durations(benchmark_profiler, iteration, per_step={"denoising": case.num_inference_steps})
            tag = case.label.replace(" ", "_").replace(":", "x")
            images[0].save(f"{file_prefix}_{tag}_perf_run{i}.png")
    return benchmark_profiler, iterations


def durations(profiler: BenchmarkProfiler, name: str, its: list[int]) -> list[float]:
    return [profiler.get_duration(name, it) for it in its]


def step_durations(profiler: BenchmarkProfiler, its: list[int], num_inference_steps: int) -> list[float]:
    steps = []
    for it in its:
        for j in range(num_inference_steps):
            assert profiler.contains_step(
                f"denoising_step_{j}", it
            ), f"All runs should have {num_inference_steps} denoising steps"
            steps.append(profiler.get_duration(f"denoising_step_{j}", it))
    return steps


def print_stats(name: str, times: list[float]) -> None:
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    print(
        f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
        f"Min: {min(times):8.4f}s | Max: {max(times):8.4f}s"
    )


def print_report(
    *,
    title: str,
    profiler: BenchmarkProfiler,
    iterations: dict[str, list[int]],
    cases: list[Case],
    config: QwenImagePipelineConfig,
    mesh_device: ttnn.MeshDevice,
) -> None:
    dit = config.dit_parallel_config
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Checkpoint: {config.checkpoint_name}, runs per case: {len(next(iter(iterations.values())))}")
    print(
        f"Configuration: cfg={dit.cfg_parallel.factor}, sp={dit.sequence_parallel.factor}, "
        f"tp={dit.tensor_parallel.factor}, encoder_tp={config.encoder_parallel_config.tensor_parallel.factor}"
    )
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {config.topology}, links: {config.num_links}")

    for case in cases:
        its = iterations[case.label]
        encoding = durations(profiler, "encoder", its)
        text_encoder = durations(profiler, "qwen_encoding", its)
        steps = step_durations(profiler, its, case.num_inference_steps)
        denoising = durations(profiler, "denoising", its)
        vae = durations(profiler, "vae", its)
        total = durations(profiler, "run", its)
        # Everything outside the three stages: latent preparation (including encoding input images
        # with the VAE), image conversion, host overhead.
        other = [t - e - d - v for t, e, d, v in zip(total, encoding, denoising, vae, strict=True)]

        print("-" * 100)
        print(f"Case: {case.label}, {case.num_inference_steps} steps")
        print_stats("Total Encoding", encoding)
        print_stats("  Text Encoder", text_encoder)
        print_stats("Denoising (per step)", steps)
        print_stats("Denoising (total)", denoising)
        print_stats("VAE Decoding", vae)
        print_stats("Other", other)
        print_stats("Total Pipeline", total)

    print("=" * 100)


@pytest.mark.parametrize(
    "all_resolutions",
    [
        pytest.param(False, id="1024x1024"),
        # Off in CI, where it would lengthen the job; set QWENIMAGE_PERF_ALL_RESOLUTIONS=1 to run it.
        # Also times inpainting.
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

    resolutions = [r for r in BENCHMARK_RESOLUTIONS if r in config.resolutions] if all_resolutions else [(1024, 1024)]
    cases = [
        Case(f"{width}x{height}", NUM_INFERENCE_STEPS, {"width": width, "height": height})
        for width, height in resolutions
    ]
    if all_resolutions:
        # The dog image is square, so this inpaints at 1024².
        cases.append(Case("inpaint 1024x1024", NUM_INFERENCE_STEPS, inpaint_inputs()))

    benchmark_profiler, iterations = run_cases(
        pipeline,
        cases,
        prompt=PROMPT,
        file_prefix="qwenimage",
        runs_per_case=BENCHMARK_RUNS_PER_CASE if all_resolutions else CI_RUNS_PER_CASE,
    )
    print_report(
        title="QWEN IMAGE PIPELINE PERFORMANCE RESULTS",
        profiler=benchmark_profiler,
        iterations=iterations,
        cases=cases,
        config=config,
        mesh_device=mesh_device,
    )

    its_1024 = iterations["1024x1024"]
    measurements = {
        "total_encoding_time": statistics.mean(durations(benchmark_profiler, "encoder", its_1024)),
        "denoising_steps_time": statistics.mean(step_durations(benchmark_profiler, its_1024, NUM_INFERENCE_STEPS))
        * NUM_INFERENCE_STEPS,
        "vae_decoding_time": statistics.mean(durations(benchmark_profiler, "vae", its_1024)),
        "total_time": statistics.mean(durations(benchmark_profiler, "run", its_1024)),
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
        dit = config.dit_parallel_config
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=RUN_TYPES[key],
            ml_model_name="QwenImage",
            batch_size=1,
            config_params={
                "width": 1024,
                "height": 1024,
                "num_steps": NUM_INFERENCE_STEPS,
                "cfg_factor": dit.cfg_parallel.factor,
                "sp_factor": dit.sequence_parallel.factor,
                "tp_factor": dit.tensor_parallel.factor,
                "num_frames": 1,
                "encoder_tp_factor": config.encoder_parallel_config.tensor_parallel.factor,
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


# Off in CI, where it would lengthen the job; set QWENIMAGE_PERF_EDIT=1 to run it.
@pytest.mark.skipif(os.environ.get("QWENIMAGE_PERF_EDIT") != "1", reason="set QWENIMAGE_PERF_EDIT=1")
@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_edit_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Times Qwen-Image-Edit-2511 with one square and one 16:9 input image, and edit inpainting.

    Edit and Edit-2509 run the same transformer and text encoder, so they take the same time.
    """
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    checkpoint_name = "Qwen/Qwen-Image-Edit-2511"
    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name=checkpoint_name)
    config = QwenImagePipelineConfig.default(mesh_shape=mesh_device.shape, checkpoint_name=checkpoint_name)

    dog = Image.open(DOG_IMAGE_PATH).convert("RGB")
    cases = [
        Case(f"edit {name}", EDIT_NUM_INFERENCE_STEPS, {"image": crop_to_aspect_ratio(dog, aspect)})
        for name, aspect in EDIT_ASPECT_RATIOS.items()
    ]
    cases.append(Case("edit inpaint 1:1", EDIT_NUM_INFERENCE_STEPS, inpaint_inputs()))

    benchmark_profiler, iterations = run_cases(
        pipeline,
        cases,
        prompt=EDIT_PROMPT,
        file_prefix="qwenimage_edit_2511",
        runs_per_case=BENCHMARK_RUNS_PER_CASE,
    )
    print_report(
        title="QWEN IMAGE EDIT PIPELINE PERFORMANCE RESULTS",
        profiler=benchmark_profiler,
        iterations=iterations,
        cases=cases,
        config=config,
        mesh_device=mesh_device,
    )

    pipeline.synchronize_devices()

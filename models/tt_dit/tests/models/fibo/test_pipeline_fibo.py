# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.events import log_section_durations, profiler_event_callback
from models.tt_dit.pipelines.fibo.pipeline_fibo import FiboPipeline

NUM_INFERENCE_STEPS = 20

TEXT_PROMPTS = ["A red bicycle leaning against a stone wall at sunset."]

STRUCTURED_PROMPTS = [
    """{"short_description": "A realistic image features a zebra standing on a concrete sidewalk next to a red fire hydrant. The zebra is positioned prominently in the center-right of the frame, facing towards the right with its head slightly lowered. The fire hydrant is in the bottom-left foreground. The background consists of a plain, light-colored wall, suggesting an urban or industrial setting. The lighting is even, highlighting the zebra's distinctive black and white stripes and the vibrant red of the hydrant.", "objects": [{"description": "A full-grown zebra with distinct black and white stripes covering its entire body. Its mane is short and upright, and its tail is long and bushy at the end. The zebra appears healthy and well-fed.", "location": "center-right", "relationship": "The zebra is standing next to the fire hydrant, appearing to be observing it or simply pausing in its vicinity.", "relative_size": "large within frame", "shape_and_color": "Elongated, equine shape with alternating black and white stripes.", "texture": "The zebra's coat appears smooth and short, typical of a mammal's fur. End of texture answer.", "appearance_details": "The stripes are sharply defined and vary in width and pattern across its body. Its muzzle is dark, and its eyes are dark and alert.", "number_of_objects": null, "pose": "Standing upright on all four legs, with its head slightly lowered and turned to its right.", "expression": "Calm and observant.", "clothing": null, "action": "Standing still.", "gender": "Unidentifiable.", "skin_tone_and_texture": null, "orientation": "Facing right."}, {"description": "A classic red fire hydrant, cylindrical in shape with various valves and caps. It has a chain connecting two of its components.", "location": "bottom-left foreground", "relationship": "The fire hydrant is situated on the sidewalk, directly in front of the zebra's left front leg.", "relative_size": "medium", "shape_and_color": "Cylindrical, bright red.", "texture": "The fire hydrant appears to have a smooth, painted metallic surface with some visible wear and tear. End of texture answer.", "appearance_details": "It has a slightly weathered appearance, with some dirt or grime near its base.", "number_of_objects": null, "pose": null, "expression": null, "clothing": null, "action": null, "gender": null, "skin_tone_and_texture": null, "orientation": "Upright."}], "background_setting": "The background is a plain, light gray concrete wall, suggesting an urban environment. Below the wall, there is a narrow strip of what appears to be dry grass or dirt, indicating a small patch of nature in an otherwise man-made setting. The ground is a concrete sidewalk with a curb separating it from a darker asphalt road.", "lighting": {"conditions": "Bright daylight", "direction": "Evenly lit, possibly from above or slightly front-lit.", "shadows": "Subtle, soft shadows are visible beneath the zebra and the fire hydrant, indicating a clear day with diffused light."}, "aesthetics": {"composition": "Centered, with the zebra occupying the majority of the frame and the fire hydrant providing a contrasting element in the foreground.", "color_scheme": "Monochromatic (black and white) for the zebra, contrasted with a vibrant red for the hydrant and neutral grays for the background.", "mood_atmosphere": "Surreal and intriguing, due to the unexpected presence of a zebra in an urban setting."}, "photographic_characteristics": {"depth_of_field": "Shallow, with the zebra and fire hydrant in sharp focus and the background slightly blurred.", "focus": "Sharp focus on subject.", "camera_angle": "Eye-level.", "lens_focal_length": "Standard."}, "style_medium": "photograph", "text_render": [], "context": "This is an art piece or conceptual photograph, likely created digitally, that plays on the juxtaposition of a wild animal in an unexpected urban environment. It could be used for advertising, editorial content, or as a standalone piece of art designed to provoke thought or amusement.", "artistic_style": "Surreal, realistic"}""",  # noqa: E501
]


DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32_768, "trace_region_size": 256_000_000}],
    indirect=True,
)
MESH_DEVICE = pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 2), id="2x2", marks=run_for_blackhole()),
        pytest.param((2, 4), id="2x4", marks=run_for_wormhole_b0()),
        pytest.param((4, 8), id="4x8", marks=run_for_blackhole()),
    ],
    indirect=True,
)


@DEVICE_PARAMS
@pytest.mark.parametrize(
    ("width", "height"),
    [
        # 1 MP tier, aspect ratios according to API docs
        # resolutions taken from images generated via https://huggingface.co/spaces/briaai/FIBO
        pytest.param(1024, 1024, id="1024x1024_1x1_1mp"),
        # pytest.param(1152, 768, id="1152x768_3x2_1mp"),
        # pytest.param(1024, 768, id="1024x768_4x3_1mp"),
        # pytest.param(960, 768, id="960x768_5x4_1mp"),
        # pytest.param(1024, 576, id="1024x576_16x9_1mp"),
    ],
)
@MESH_DEVICE
def test_fibo_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    model_location_generator,
) -> None:
    pipeline = FiboPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=height,
        width=width,
        checkpoint_name=model_location_generator("briaai/FIBO"),
        vlm_checkpoint_name=None,
    )

    benchmark_profiler = BenchmarkProfiler()

    for i, prompt in enumerate(STRUCTURED_PROMPTS):
        with benchmark_profiler("run", iteration=i):
            images = pipeline(
                prompts=[prompt],
                num_inference_steps=NUM_INFERENCE_STEPS,
                seed=0,
                use_vlm=False,
                on_event=profiler_event_callback(benchmark_profiler, i),
            )

        output_filename = f"fibo_{width}_{height}_{i}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved to {output_filename}")

        log_section_durations(benchmark_profiler, i, per_step={"denoising": NUM_INFERENCE_STEPS})


@DEVICE_PARAMS
@MESH_DEVICE
def test_fibo_pipeline_vlm(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    """Natural-language prompts, turned into FIBO's structured ones by the VLM."""
    pipeline = FiboPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("briaai/FIBO"),
        vlm_checkpoint_name=model_location_generator("briaai/FIBO-vlm"),
    )

    benchmark_profiler = BenchmarkProfiler()

    for i, prompt in enumerate(TEXT_PROMPTS):
        with benchmark_profiler("run", iteration=i):
            images = pipeline(
                prompts=[prompt],
                num_inference_steps=NUM_INFERENCE_STEPS,
                seed=0,
                on_event=profiler_event_callback(benchmark_profiler, i),
            )

        output_filename = f"fibo_vlm_{i}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved to {output_filename}")

        log_section_durations(benchmark_profiler, i, per_step={"denoising": NUM_INFERENCE_STEPS})

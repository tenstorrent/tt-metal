# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.fibo.pipeline_fibo import FiboPipeline, FiboPipelineConfig


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 34000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 20)])
@pytest.mark.parametrize(
    (
        "mesh_device",
        "cfg",
        "sp",
        "tp",
        "encoder_tp",
        "vae_tp",
        "topology",
        "num_links",
        "mesh_test_id",
    ),
    [
        pytest.param(
            (2, 2),  # mesh_device
            (1, 0),  # cfg
            (1, 0),  # sp
            (2, 1),  # tp
            (2, 1),  # encoder_tp
            (2, 1),  # vae_tp
            ttnn.Topology.Linear,
            1,
            "2x2cfg0sp0tp1",
            id="2x2cfg0sp0tp1",
            marks=run_for_blackhole(),
        ),
        pytest.param(
            (2, 4),  # mesh_device
            (2, 0),  # cfg
            (1, 0),  # sp
            (4, 1),  # tp
            (4, 1),  # encoder_tp
            (4, 1),  # vae_tp
            ttnn.Topology.Linear,
            1,
            "2x4cfg0sp0tp1",
            id="2x4cfg0sp0tp1",
            marks=run_for_wormhole_b0(),
        ),
        pytest.param(
            (2, 4),  # mesh_device
            (2, 1),  # cfg
            (2, 0),  # sp
            (2, 1),  # tp
            (4, 1),  # encoder_tp
            (4, 1),  # vae_tp
            ttnn.Topology.Linear,
            1,
            "2x4cfg1sp0tp1",
            id="2x4cfg1sp0tp1",
            marks=run_for_wormhole_b0(),
        ),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_torch_text_encoder",
    [
        pytest.param(False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        pytest.param(False, id="not_traced"),
    ],
)
def test_fibo_pipeline(
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
    no_prompt: bool,
    use_torch_text_encoder: bool,
    traced: bool,
    mesh_test_id: str,
    is_ci_env: bool,
    model_location_generator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FiboPipeline(
        device=mesh_device,
        config=FiboPipelineConfig.default(
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=cfg, sp=sp, tp=tp),
            encoder_parallel_config=EncoderParallelConfig.from_tuple(encoder_tp),
            vae_parallel_config=VAEParallelConfig.from_tuple(vae_tp),
            num_links=num_links,
            topology=topology,
            height=height,
            width=width,
            use_torch_text_encoder=use_torch_text_encoder,
            checkpoint_name=model_location_generator("briaai/FIBO"),
        ),
    )

    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
        if traced:
            pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

    prompts = [
        """{"short_description": "A realistic image features a zebra standing on a concrete sidewalk next to a red fire hydrant. The zebra is positioned prominently in the center-right of the frame, facing towards the right with its head slightly lowered. The fire hydrant is in the bottom-left foreground. The background consists of a plain, light-colored wall, suggesting an urban or industrial setting. The lighting is even, highlighting the zebra's distinctive black and white stripes and the vibrant red of the hydrant.", "objects": [{"description": "A full-grown zebra with distinct black and white stripes covering its entire body. Its mane is short and upright, and its tail is long and bushy at the end. The zebra appears healthy and well-fed.", "location": "center-right", "relationship": "The zebra is standing next to the fire hydrant, appearing to be observing it or simply pausing in its vicinity.", "relative_size": "large within frame", "shape_and_color": "Elongated, equine shape with alternating black and white stripes.", "texture": "The zebra's coat appears smooth and short, typical of a mammal's fur. End of texture answer.", "appearance_details": "The stripes are sharply defined and vary in width and pattern across its body. Its muzzle is dark, and its eyes are dark and alert.", "number_of_objects": null, "pose": "Standing upright on all four legs, with its head slightly lowered and turned to its right.", "expression": "Calm and observant.", "clothing": null, "action": "Standing still.", "gender": "Unidentifiable.", "skin_tone_and_texture": null, "orientation": "Facing right."}, {"description": "A classic red fire hydrant, cylindrical in shape with various valves and caps. It has a chain connecting two of its components.", "location": "bottom-left foreground", "relationship": "The fire hydrant is situated on the sidewalk, directly in front of the zebra's left front leg.", "relative_size": "medium", "shape_and_color": "Cylindrical, bright red.", "texture": "The fire hydrant appears to have a smooth, painted metallic surface with some visible wear and tear. End of texture answer.", "appearance_details": "It has a slightly weathered appearance, with some dirt or grime near its base.", "number_of_objects": null, "pose": null, "expression": null, "clothing": null, "action": null, "gender": null, "skin_tone_and_texture": null, "orientation": "Upright."}], "background_setting": "The background is a plain, light gray concrete wall, suggesting an urban environment. Below the wall, there is a narrow strip of what appears to be dry grass or dirt, indicating a small patch of nature in an otherwise man-made setting. The ground is a concrete sidewalk with a curb separating it from a darker asphalt road.", "lighting": {"conditions": "Bright daylight", "direction": "Evenly lit, possibly from above or slightly front-lit.", "shadows": "Subtle, soft shadows are visible beneath the zebra and the fire hydrant, indicating a clear day with diffused light."}, "aesthetics": {"composition": "Centered, with the zebra occupying the majority of the frame and the fire hydrant providing a contrasting element in the foreground.", "color_scheme": "Monochromatic (black and white) for the zebra, contrasted with a vibrant red for the hydrant and neutral grays for the background.", "mood_atmosphere": "Surreal and intriguing, due to the unexpected presence of a zebra in an urban setting."}, "photographic_characteristics": {"depth_of_field": "Shallow, with the zebra and fire hydrant in sharp focus and the background slightly blurred.", "focus": "Sharp focus on subject.", "camera_angle": "Eye-level.", "lens_focal_length": "Standard."}, "style_medium": "photograph", "text_render": [], "context": "This is an art piece or conceptual photograph, likely created digitally, that plays on the juxtaposition of a wild animal in an unexpected urban environment. It could be used for advertising, editorial content, or as a standalone piece of art designed to provoke thought or amusement.", "artistic_style": "Surreal, realistic"}""",  # noqa: E501
    ]

    filename_prefix = f"fibo_{width}_{height}_{mesh_test_id}"
    if use_torch_text_encoder:
        filename_prefix += "_enccpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        benchmark_profiler = BenchmarkProfiler()
        with benchmark_profiler("run", iteration=0):
            images = pipeline(
                prompts=[prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                traced=traced,
                vae_traced=False,
                encoder_traced=False,
                on_event=profiler_event_callback(benchmark_profiler, 0),
            )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        logger.info(f"SmolLM3 encoding time: {benchmark_profiler.get_duration('smollm3_encoding', 0):.2f}s")
        logger.info(f"Total encoding time: {benchmark_profiler.get_duration('encoder', 0):.2f}s")
        logger.info(f"VAE decoding time: {benchmark_profiler.get_duration('vae', 0):.2f}s")
        logger.info(f"Total pipeline time: {benchmark_profiler.get_duration('total', 0):.2f}s")
        avg_step_time = benchmark_profiler.get_duration("denoising", 0) / num_inference_steps
        logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

    if no_prompt:
        for i, prompt in enumerate(prompts):
            run(prompt=prompt, number=i, seed=0)
    else:
        prompt = prompts[0]
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)

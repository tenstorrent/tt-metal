# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

# if TYPE_CHECKING:
import ttnn

from .tt.fun_pipeline import TtStableDiffusion3Pipeline
from .tt.parallel_config import create_dit_parallel_config, ParallelConfig


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",  # "prompt_sequence_length", "spatial_sequence_length",
    [
        #        ("medium", 512, 512, 4.5, 40, 333, 1024),
        #        ("medium", 1024, 1024, 4.5, 40, 333, 4096),
        #        ("large", 512, 512, 3.5, 28, 333, 1024),
        ("large", 1024, 1024, 3.5, 28),  # , 333, 4096),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15210496}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_sd3(
    *, mesh_device: ttnn.MeshDevice, model_name, image_w, image_h, guidance_scale, num_inference_steps
) -> None:  # , prompt_sequence_length, spatial_sequence_length,) -> None:
    mesh_shape = tuple(mesh_device.shape)
    cfg_parallel = ParallelConfig(mesh_shape=mesh_shape, factor=1, mesh_axis=0)
    tensor_parallel = ParallelConfig(mesh_shape=(mesh_shape[0], 1), factor=mesh_shape[1], mesh_axis=1)
    dit_parallel_config = create_dit_parallel_config(
        mesh_shape=mesh_shape, cfg_parallel=cfg_parallel, tensor_parallel=tensor_parallel
    )

    if guidance_scale > 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    pipeline = TtStableDiffusion3Pipeline(
        checkpoint=f"stabilityai/stable-diffusion-3.5-{model_name}",
        device=mesh_device,
        enable_t5_text_encoder=mesh_device.get_num_devices() >= 4,
        guidance_cond=guidance_cond,
        parallel_config=dit_parallel_config,
    )

    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )

    while True:
        new_prompt = input("Enter the input prompt, or q to exit:")
        if new_prompt:
            prompt = new_prompt
        if prompt[0] == "q":
            break

        negative_prompt = ""

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

        images[0].save(f"sd35_{image_w}_{image_h}.png")

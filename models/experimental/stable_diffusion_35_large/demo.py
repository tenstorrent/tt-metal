# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import os

import pytest

# if TYPE_CHECKING:
import ttnn

from .tt import TtStableDiffusion3Pipeline


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
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
    "model_version , image_w, image_h, guidance_scale, num_inference_steps",
    [
        # Commented out options, to be considered later for different variants of the model size and resolution
        #        ("medium", 512, 512, 4.5, 40, 333, 1024),
        #        ("medium", 1024, 1024, 4.5, 40, 333, 4096),
        #        ("large", 512, 512, 3.5, 28, 333, 1024),
        ("large", 1024, 1024, 3.5, 28),  # , 333, 4096),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024, "trace_region_size": 15210496}], indirect=True)
def test_sd3(
    *,
    mesh_device: ttnn.MeshDevice,
    model_version,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    no_prompt,
    model_location_generator,
) -> None:  # , prompt_sequence_length, spatial_sequence_length,) -> None:
    if guidance_scale > 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_version}",
        device=mesh_device,
        enable_t5_text_encoder=mesh_device.get_num_devices() >= 4,
        vae_cpu_fallback=True,
        guidance_cond=guidance_cond,
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
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

    if no_prompt:
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

    else:
        ## interactive demo
        for i in itertools.count():
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

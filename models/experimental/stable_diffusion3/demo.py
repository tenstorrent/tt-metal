# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .tt import TtStableDiffusion3Pipeline

if TYPE_CHECKING:
    import ttnn


@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        #        ("medium", 512, 512, 4.5, 40),
        #        ("medium", 1024, 1024, 4.5, 40),
        #        ("large", 512, 512, 3.5, 28),
        ("large", 1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15210496}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_sd3(*, device: ttnn.Device, model_name, image_w, image_h, guidance_scale, num_inference_steps) -> None:
    pipeline = TtStableDiffusion3Pipeline(
        checkpoint=f"stabilityai/stable-diffusion-3.5-{model_name}",
        device=device,
        enable_t5_text_encoder=True,
    )

    pipeline.prepare(batch_size=1, width=image_w, height=image_h, guidance_scale=guidance_scale)

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

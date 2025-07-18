# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import os

import ttnn
from models.experimental.flux import FluxPipeline

import pytest


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_inference_steps,image_w,image_h,batch_size,mesh_width,t5_on_device",
    [
        (4, 1024, 1024, 1, 8, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024, "trace_region_size": 15210496}], indirect=True)
def test_flux_1_schnell(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps,
    image_w,
    image_h,
    batch_size,
    mesh_width,
    t5_on_device,
    no_prompt,
    model_location_generator,
) -> None:
    device_count = ttnn.get_num_devices()
    mesh_height = batch_size
    mesh_width = mesh_width if mesh_width is not None else device_count // mesh_height

    pipeline = FluxPipeline(
        checkpoint="black-forest-labs/FLUX.1-schnell",
        device=mesh_device,
        use_torch_encoder=(not t5_on_device),
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
        width=image_w,
        height=image_h,
        prompt_count=1,
        num_images_per_prompt=mesh_height,
    )

    prompt = "A luxury sports car."

    if no_prompt:
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
        )
        for i, image in enumerate(images, start=1):
            image.save(f"flux_{image_w}_{i}.png")
    else:
        for iteration in itertools.count(start=1):
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt == "q":
                break

            images = pipeline(
                prompt_1=[prompt],
                prompt_2=[prompt],
                num_inference_steps=num_inference_steps,
                seed=iteration,
            )
            for i, image in enumerate(images, start=1):
                image.save(f"flux_{image_w}_{i}.png")

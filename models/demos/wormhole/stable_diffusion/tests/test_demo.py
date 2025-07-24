# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline

from models.demos.wormhole.stable_diffusion.demo.demo import load_inputs
from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference as demo
from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference_diffusiondb as demo_db
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 11 * 8192, "trace_region_size": 789321728}], indirect=True)
@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((5),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo_sd_db(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    if device.core_grid.y != 8:
        pytest.skip("Needs 8x8 Grid")
    demo_db(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 11 * 8192, "trace_region_size": 789835776}], indirect=True)
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/wormhole/stable_diffusion/demo/input_data.json"),),
    ids=["default_input"],
)
def test_demo_sd(device, reset_seeds, input_path):
    num_prompts = 1
    num_inference_steps = 50
    image_size = (512, 512)

    inputs = load_inputs(input_path)
    input_prompts = inputs[:1]
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cpu")

    import torch

    torch_image = pipe(
        input_prompts[0], generator=torch.Generator(device="cpu").manual_seed(174), num_inference_steps=50
    ).images[0]
    torch_image.save("output_torch_image.png")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL to tensor and scales [0,255] -> [0,1]
        ]
    )
    torch_image = transform(torch_image)

    ttnn_image = demo(
        device,
        reset_seeds,
        input_path,
        num_prompts,
        num_inference_steps,
        image_size,
    )[0]
    assert_with_pcc(torch_image, ttnn_image.reshape(ttnn_image.shape[1:]).permute([2, 0, 1]), 0.935)

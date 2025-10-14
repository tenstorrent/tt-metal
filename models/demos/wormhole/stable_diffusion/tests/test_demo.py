# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torchvision.transforms as transforms

from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE, SD_TRACE_REGION_SIZE
from models.demos.wormhole.stable_diffusion.demo.demo import load_inputs
from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference as demo
from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference_diffusiondb as demo_db
from models.demos.wormhole.stable_diffusion.sd_helper_funcs import get_refference_stable_diffusion_pipeline
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SD_L1_SMALL_SIZE, "trace_region_size": SD_TRACE_REGION_SIZE}], indirect=True
)
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
def test_demo_sd_db(
    device,
    reset_seeds,
    input_path,
    num_prompts,
    num_inference_steps,
    image_size,
    is_ci_env,
    is_ci_v2_env,
    model_location_generator,
):
    if device.core_grid.y != 8:
        pytest.skip("Needs 8x8 Grid")
    demo_db(
        device,
        is_ci_env,
        is_ci_v2_env,
        model_location_generator,
        reset_seeds,
        input_path,
        num_prompts,
        num_inference_steps,
        image_size,
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SD_L1_SMALL_SIZE, "trace_region_size": SD_TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/wormhole/stable_diffusion/demo/input_data.json"),),
    ids=["default_input"],
)
def test_demo_sd(device, reset_seeds, input_path, is_ci_env, is_ci_v2_env, model_location_generator):
    num_prompts = 1
    num_inference_steps = 50
    image_size = (512, 512)

    inputs = load_inputs(input_path)
    input_prompts = inputs[:1]
    pipe = get_refference_stable_diffusion_pipeline(is_ci_env, is_ci_v2_env, model_location_generator)

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
        is_ci_env,
        is_ci_v2_env,
        model_location_generator,
        reset_seeds,
        input_path,
        num_prompts,
        num_inference_steps,
        image_size,
    )[0]
    assert_with_pcc(torch_image, ttnn_image.reshape(ttnn_image.shape[1:]).permute([2, 0, 1]), 0.935)

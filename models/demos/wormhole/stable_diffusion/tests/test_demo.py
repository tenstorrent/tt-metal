# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference as demo
from models.demos.wormhole.stable_diffusion.demo.demo import run_demo_inference_diffusiondb as demo_db


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/wormhole/stable_diffusion/demo/input_data.json"),),
    ids=["default_input"],
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
def test_demo_sd(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    if device.core_grid.y != 8:
        pytest.skip("Needs 8x8 Grid")
    demo(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
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

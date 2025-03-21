# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.utility_functions import (
    run_for_wormhole_b0,
)

from models.demos.ttnn_resnet.demo.demo import run_resnet_imagenet_inference, run_resnet_inference


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((16, 100),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/ttnn_resnet/demo/images/"),),
)
def test_demo_sample(
    mesh_device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator
):
    run_resnet_inference(batch_size, input_loc, imagenet_label_dict, mesh_device, model_location_generator)

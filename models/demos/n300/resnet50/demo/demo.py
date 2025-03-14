# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import (
    run_for_wormhole_b0,
)

from models.demos.ttnn_resnet.tests.resnet50_performant import (
    run_resnet50_inference,
)

from models.demos.ttnn_resnet.demo.demo import run_resnet_imagenet_inference


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((16, 1),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_demo_sample(mesh_device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator):
    run_resnet50_inference(mesh_device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)

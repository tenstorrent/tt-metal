# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import argparse

import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_x2_harvested

from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.unet_utils import create_custom_preprocessor

import time
import os

import ttnn

from models.experimental.functional_unet.unet_utils import create_unet_models, create_unet_input_tensors


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
def test_unet_demo(device, perf_mode, batch, groups):
    with torch.no_grad():
        torch.manual_seed(0)

        # Create initial parameters
        torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(device, batch, groups)
        torch_model, ttnn_model = create_unet_models(device, groups, torch_input_tensor)

        # Run torch golden result
        torch_output_tensor = torch_model(torch_input_tensor)

        # Run ttnn output result
        output_tensor = ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)

        # Tensor postprocessing
        input_shape = torch_input_tensor.shape
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = output_tensor[:, :, :, :1]
        output_tensor = output_tensor.reshape(input_shape[0], input_shape[2], input_shape[3], -1)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)


"""
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=0, type=int)
    ap.add_argument("--perf-mode", action="store_true")
    ap.add_argument("--groups", default=1, type=int)
    args = ap.parse_args()

    device_id = 0
    device = ttnn.open_device(device_id=device_id, l1_small_size=32768)
    test_unet_demo(device, args.perf_mode, args.batch, args.groups)
    ttnn.close_device(device)
"""

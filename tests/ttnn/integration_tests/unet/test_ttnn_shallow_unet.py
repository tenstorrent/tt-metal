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
import tt_lib as ttl
import os
from tt_lib import profiler

import ttnn


@pytest.mark.skip(reason="#9417: Various failures preventing model from running")
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("loop", [0])
@pytest.mark.parametrize("perf_mode, groups", [(False, 1), (True, 1), (True, 2)])  # , (True, 4)])
def test_unet(device, loop, perf_mode, groups):
    if perf_mode and device.arch() == ttl.device.Arch.GRAYSKULL:
        pytest.skip("Perf mode is not supported on Grayskull")

    if is_x2_harvested(device):
        pytest.skip("x2 harvested chip is not supported")

    with torch.no_grad():
        torch.manual_seed(0)
        torch_model = unet_shallow_torch.UNet(groups=groups)
        torch_model.eval()

        new_state_dict = {}
        for name, parameter in torch_model.state_dict().items():
            if isinstance(parameter, torch.FloatTensor):
                if "b1" or "b2" or "b3" or "b4" or "bnb" in name:
                    new_state_dict[name] = parameter
                else:
                    new_state_dict[name] = parameter + 1000

        torch_model.load_state_dict(new_state_dict)

        batch = 1 if groups > 1 else 2
        torch_input_tensor = torch.randn(
            batch, 4 * groups, 1056, 160
        )  # Batch size of 2, 3 channels (RGB), 1056x160 input
        original_shape = list(torch_input_tensor.shape)
        torch_output_tensor = torch_model(torch_input_tensor)

        reader_patterns_cache = {}
        parameters = preprocess_model(
            initialize_model=lambda: torch_model,
            run_model=lambda model: model(torch_input_tensor),
            custom_preprocessor=create_custom_preprocessor(device, groups=groups),
            reader_patterns_cache=reader_patterns_cache,
            device=device,
        )

        ttnn_model = unet_shallow_ttnn.UNet(parameters)

        #
        # Tensor Preprocessing
        #
        input_shape = torch_input_tensor.shape
        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

        # Pad to 16 if grayskull run and 32 for wormhole
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
        )
        pad = 32 if device.arch() == ttl.device.Arch.WORMHOLE_B0 else 16
        hpad = 0  # 96*32*64
        if input_tensor.shape[-1] < pad or input_tensor.shape[-2] < hpad:
            input_tensor = torch.nn.functional.pad(
                input_tensor, (0, max(0, pad - input_tensor.shape[-1]), 0, max(0, hpad - input_tensor.shape[-2]))
            )
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

        warmup = 1
        start = None
        for i in range(loop + warmup):
            if i == warmup:
                start = time.perf_counter()
            profiler.tracy_frame()
            output_tensor = ttnn_model(device, input_tensor, original_shape, perf_mode=perf_mode)
        if start is not None:
            stop = time.perf_counter()
            total_time = stop - start
            batch = input_shape[0]
            total_frame_count = batch * loop
            print(f"Elapsed host time (sec): {total_time}")
            print(f"Frames processed: {total_frame_count}")
            print(f"Host perf (fps): {total_frame_count / total_time}")

        #
        # Tensor Postprocessing
        #
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = output_tensor[:, :, :, :groups]
        output_tensor = output_tensor.reshape(input_shape[0], input_shape[2], input_shape[3], -1)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.to(torch_input_tensor.dtype)

        if perf_mode:
            pass  # skip PCC checks if perf_mode is set
        elif device.arch() == ttl.device.Arch.WORMHOLE_B0:
            assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
        else:
            assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.97)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default=0, type=int)
    ap.add_argument("--perf-mode", action="store_true")
    ap.add_argument("--groups", default=1, type=int)
    args = ap.parse_args()

    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    test_unet(device, args.loop, args.perf_mode, args.groups)
    ttnn.close_device(device)

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.models import Generator as TorchGenerator
from models.demos.rvc.tt_impl.synthesizer.models import Generator as TTGenerator
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_generator(device):
    torch.manual_seed(0)

    batch_size = 1
    initial_channel = 8
    input_length = 16
    resblock = "2"
    resblock_kernel_sizes = [3]
    resblock_dilation_sizes = [(1, 3)]
    upsample_rates = [2]
    upsample_initial_channel = 16
    upsample_kernel_sizes = [4]
    gin_channels = 0

    torch_generator = TorchGenerator(
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
    ).eval()

    torch_input = torch.randn(batch_size, initial_channel, input_length, dtype=torch.float32)
    torch_output = torch_generator(torch_input, g=None)

    tt_generator = TTGenerator(
        device=device,
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=list(resblock_kernel_sizes),
        resblock_dilation_sizes=[list(x) for x in resblock_dilation_sizes],
        upsample_rates=list(upsample_rates),
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=list(upsample_kernel_sizes),
        gin_channels=gin_channels,
    )
    parameters = {f"dec.{k}": v for k, v in torch_generator.state_dict().items()}
    tt_generator.load_parameters(parameters, prefix="dec.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_generator(tt_input, g=None)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, 1).permute(0, 2, 1)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.98)

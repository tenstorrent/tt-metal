# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.models import GeneratorNSF as TorchGeneratorNSF
from models.demos.rvc.tt_impl.synthesizer.models import GeneratorNSF as TTGeneratorNSF
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_generator_nsf(device):
    torch.manual_seed(0)

    batch_size = 1
    initial_channel = 64
    input_length = 128
    resblock = "2"
    resblock_kernel_sizes = [3]
    resblock_dilation_sizes = [(1, 3)]
    upsample_rates = [2]
    upsample_initial_channel = 16
    upsample_kernel_sizes = [4]
    gin_channels = 32
    sr = 32000

    torch_generator = TorchGeneratorNSF(
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
        sr=sr,
    ).eval()

    tt_generator = TTGeneratorNSF(
        device=device,
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=list(resblock_kernel_sizes),
        resblock_dilation_sizes=[list(x) for x in resblock_dilation_sizes],
        upsample_rates=list(upsample_rates),
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=list(upsample_kernel_sizes),
        gin_channels=gin_channels,
        sr=sr,
    )
    parameters = {f"dec.{k}": v for k, v in torch_generator.state_dict().items()}
    tt_generator.load_state_dict(parameters, module_prefix="dec.")

    torch_x = torch.randn(batch_size, initial_channel, input_length, dtype=torch.float32)
    torch_f0 = torch.rand(batch_size, input_length, dtype=torch.float32) * 300.0
    torch_g = torch.randn(batch_size, gin_channels, 128, dtype=torch.float32)

    torch.manual_seed(1234)
    torch_output = torch_generator(torch_x, torch_f0, g=torch_g)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_f0 = ttnn.from_torch(
        torch_f0.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_g = ttnn.from_torch(
        torch_g.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    torch.manual_seed(1234)
    tt_output = tt_generator(tt_x, tt_f0, g=tt_g)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).permute(0, 2, 1)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.95)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.torch_impl.bottleneck import BottleneckTch
from models.demos.unet_3d.ttnn_impl.bottleneck import Bottleneck
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C_in, C_out, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C_in, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)

    # initlize torch bottleneck layer
    bottleneck_torch = BottleneckTch(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=k,
        padding=1,
    ).to(dtype=torch.bfloat16)
    bottleneck_torch.eval()
    torch_output = bottleneck_torch(torch_tensor)
    torch_output = torch_output.permute(0, 2, 3, 4, 1).reshape(N, -1)
    # Initialize Bottleneck layer
    bottleneck_layer = Bottleneck(device=device, in_channels=C_in, out_channels=C_out)
    params_dict = bottleneck_torch.state_dict()
    bottleneck_layer.init_params(device, params_dict)
    # Perform encoding
    tt_output = bottleneck_layer(ttnn_tensor)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)
    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_bottleneck():
    N, D, H, W = 1, 32, 32, 32
    C_in, C_out = 32, 32
    k = 3
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4,
    )
    run_single_case(device, N, D, H, W, C_in, C_out, k)


if __name__ == "__main__":
    test_bottleneck()

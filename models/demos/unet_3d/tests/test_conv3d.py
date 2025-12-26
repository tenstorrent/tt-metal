# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.conv3d import Conv3D
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C_in, C_out, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C_in, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(
        torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # initlize torch conv3d layer
    conv3d_torch = torch.nn.Conv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=k,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    ).to(dtype=torch.bfloat16)
    conv3d_torch.eval()
    torch_output = conv3d_torch(torch_tensor).permute(0, 2, 3, 4, 1)
    torch_output = torch_output.reshape(N, -1)

    # Initialize Conv3D layer
    conv3d_layer = Conv3D(device=device, in_channels=C_in, out_channels=C_out)
    params_dict = conv3d_torch.state_dict()
    conv3d_layer.init_params(device, params_dict)
    # Perform convolution
    tt_output = conv3d_layer(ttnn_tensor)

    import time

    start = time.time()
    for _ in range(10):
        tt_output = conv3d_layer(ttnn_tensor)
    end = time.time()
    print(f"TTNN Conv3D average forward time: {(end - start)/10} seconds")

    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)

    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_conv3d():
    N, D, H, W = 1, 64, 64, 64
    C_in, C_out = 64, 64
    k = 3
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4,
    )
    run_single_case(device, N, D, H, W, C_in, C_out, k)


if __name__ == "__main__":
    test_conv3d()

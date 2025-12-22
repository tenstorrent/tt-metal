# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.torch_impl.model import UNet3DTch
from models.demos.unet_3d.ttnn_impl.model import UNet3D
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C_in, C_out, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C_in, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)

    # initlize torch UNet3D model
    unet3d_torch = UNet3DTch(
        in_channels=C_in,
        out_channels=C_out,
    ).to(dtype=torch.bfloat16)
    unet3d_torch.eval()
    import time

    start = time.time()
    torch_output = unet3d_torch(torch_tensor)
    end = time.time()
    print(f"Torch UNet3D forward time: {end - start} seconds")

    torch_output = torch_output.permute(0, 2, 3, 4, 1).reshape(N, -1)
    # Initialize UNet3D model
    unet3d_model = UNet3D(device=device, in_channels=C_in, out_channels=C_out)
    params_dict = unet3d_torch.state_dict()
    unet3d_model.init_params(device, params_dict)
    # Perform prediction
    import time

    start = time.time()
    tt_output = unet3d_model(ttnn_tensor)
    end = time.time()
    print(f"TTNN UNet3D forward time: {end - start} seconds")

    start = time.time()
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)
    tt_2 = unet3d_model(ttnn_tensor)
    end = time.time()
    print(f"TTNN UNet3D second forward time: {end - start} seconds")
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)
    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_model():
    N, D, H, W = 1, 64, 64, 64
    C_in, C_out = 32, 32
    k = 3
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4,
    )
    run_single_case(device, N, D, H, W, C_in, C_out, k)


if __name__ == "__main__":
    test_model()

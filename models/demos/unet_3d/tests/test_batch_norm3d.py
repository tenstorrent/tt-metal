# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.batch_norm_3d import BatchNorm3D
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C):
    # Create random input tensor
    torch_tensor = torch.randn((N, C, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=device)

    # initlize torch batchnorm3d layer
    bn3d_torch = torch.nn.BatchNorm3d(
        num_features=C,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ).to(dtype=torch.bfloat16)
    bn3d_torch.eval()
    torch_output = bn3d_torch(torch_tensor)
    torch_output = torch_output.reshape(N, -1)

    # Initialize BatchNorm3D layer
    bn3d_layer = BatchNorm3D(device=device, channels=C)
    params_dict = bn3d_torch.state_dict()
    bn3d_layer.init_params(device, params_dict)

    # Perform batch normalization
    tt_output = bn3d_layer(ttnn_tensor)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)

    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_batchnorm3d():
    N, D, H, W = 1, 32, 32, 32
    C = 32
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4,
    )
    run_single_case(device, N, D, H, W, C)


if __name__ == "__main__":
    test_batchnorm3d()

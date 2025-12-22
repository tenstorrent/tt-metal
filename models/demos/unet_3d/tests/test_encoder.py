# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.torch_impl.encoder import EncoderTch
from models.demos.unet_3d.ttnn_impl.encoder import Encoder
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C_in, C_out, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C_in, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)

    # initlize torch encoder layer
    encoder_torch = EncoderTch(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=k,
        padding=1,
    ).to(dtype=torch.bfloat16)
    encoder_torch.eval()
    torch_output_pooled, torch_output = encoder_torch(torch_tensor)
    torch_output_pooled = torch_output_pooled.permute(0, 2, 3, 4, 1).reshape(N, -1)
    torch_output = torch_output.permute(0, 2, 3, 4, 1).reshape(N, -1)
    # Initialize Encoder layer
    encoder_layer = Encoder(device=device, in_channels=C_in, out_channels=C_out)
    params_dict = encoder_torch.state_dict()
    encoder_layer.init_params(device, params_dict)
    # Perform encoding
    tt_output_pooled, tt_output = encoder_layer(ttnn_tensor)
    tt_output_pooled = ttnn.to_layout(tt_output_pooled, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch_pooled = ttnn.to_torch(tt_output_pooled.cpu()).reshape(N, -1)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)
    # Verify outputs are similar
    assert_with_pcc(torch_output_pooled, ttnn_torch_pooled, pcc=0.99)
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_encoder():
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
    test_encoder()

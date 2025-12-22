# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.unet_3d.torch_impl.decoder import DecoderTch
from models.demos.unet_3d.ttnn_impl.decoder import Decoder
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C_in, C_out, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C_in, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)

    torch_skip_connection = torch.randn((N, C_out, D * 2, H * 2, W * 2), dtype=torch.bfloat16)
    ttnn_skip_connection = ttnn.from_torch(
        torch_skip_connection.permute(0, 2, 3, 4, 1),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # initlize torch decoder layer
    decoder_torch = DecoderTch(
        in_channels=C_in + C_out,
        out_channels=C_out,
        kernel_size=k,
        padding=1,
    ).to(dtype=torch.bfloat16)
    decoder_torch.eval()
    torch_output = decoder_torch(torch_tensor, torch_skip_connection).permute(0, 2, 3, 4, 1)
    torch_output = torch_output.reshape(N, -1)
    # Initialize Decoder layer
    decoder_layer = Decoder(device=device, in_channels=C_in + C_out, out_channels=C_out)
    params_dict = decoder_torch.state_dict()
    decoder_layer.init_params(device, params_dict)
    # Perform decoding
    tt_output = decoder_layer(ttnn_tensor, ttnn_skip_connection)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)
    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_decoder():
    N, D, H, W = 1, 16, 16, 16
    C_in, C_out = 32, 32
    k = 3
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4,
    )
    run_single_case(device, N, D, H, W, C_in, C_out, k)


if __name__ == "__main__":
    test_decoder()

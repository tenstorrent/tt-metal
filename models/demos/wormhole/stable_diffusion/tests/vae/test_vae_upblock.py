import torch
from diffusers import (
    AutoencoderKL,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upblock import UpDecoderBlock
from models.utility_functions import is_wormhole_b0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, resnet_norm_blocks, resnet_conv1_channel_split_factors, resnet_conv2_channel_split_factors, upsample_conv_channel_split_factors, block_id",
    [
        (
            512,
            64,
            64,
            512,
            128,
            128,
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            (1, 1),
            0,
        ),
        (
            512,
            128,
            128,
            512,
            256,
            256,
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            (8 if is_wormhole_b0() else 2, 1 if is_wormhole_b0() else 2),
            1,
        ),
        (
            512,
            256,
            256,
            256,
            512,
            512,
            [(4, 4), (4, 4), (4, 16)],
            [(2, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1)],
            (8 if is_wormhole_b0() else 4, 2),
            2,
        ),
        (
            256,
            512,
            512,
            128,
            512,
            512,
            [(16, 32), (32, 32), (32, 32)],
            [(8, 1), (4, 1), (4, 1)],
            [(4, 1), (4, 1), (4, 1)],
            (1, 1),
            3,
        ),
    ],
)
def test_upblock(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    resnet_norm_blocks,
    resnet_conv1_channel_split_factors,
    resnet_conv2_channel_split_factors,
    upsample_conv_channel_split_factors,
    block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    torch_upblock = vae.decoder.up_blocks[block_id]

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_upblock(torch_input)

    ttnn_model = UpDecoderBlock(
        torch_upblock,
        device,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        resnet_norm_blocks,
        resnet_conv1_channel_split_factors,
        resnet_conv2_channel_split_factors,
        upsample_conv_channel_split_factors,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    ttnn_output = ttnn_model(ttnn_input)

    if ttnn_output.shape[1] == 1:
        ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])

    result = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, result, 0.99)

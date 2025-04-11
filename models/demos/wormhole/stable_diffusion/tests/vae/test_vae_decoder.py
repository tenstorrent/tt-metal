import torch
from diffusers import (
    AutoencoderKL,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_decoder import VaeDecoder


@pytest.mark.parametrize("device_params", [{"l1_small_size": 524288}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width",
    [
        (
            512,
            64,
            64,
            512,
            512,
            512,
        ),
    ],
)
def test_decoder(
    device, input_channels, input_height, input_width, out_channels, output_height, output_width, use_program_cache
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_decoder = vae.decoder

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_decoder(torch_input)

    ttnn_model = VaeDecoder(
        torch_decoder,
        device,
        input_channels,
        input_height,
        input_width,
        512,
        out_channels,
        output_height,
        output_width,
    )
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

    print(result.shape)

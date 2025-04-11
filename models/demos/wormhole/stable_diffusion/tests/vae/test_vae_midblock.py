import torch
from diffusers import (
    AutoencoderKL,
)

from tracy import signpost

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import ttnn


from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_midblock import MidBlock


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width",
    [
        (512, 64, 64),
    ],
)
def test_upblock(device, input_channels, input_height, input_width, use_program_cache):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    torch_midblock = vae.decoder.mid_block

    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_midblock(torch_input)

    ttnn_model = MidBlock(torch_midblock, device, input_channels, input_height, input_width)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    signpost(header="start")
    ttnn_output = ttnn_model(ttnn_input)
    signpost(header="end")
    ttnn_output = ttnn.reshape(ttnn_output, [1, input_height, input_width, input_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])

    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, result, 0.99)

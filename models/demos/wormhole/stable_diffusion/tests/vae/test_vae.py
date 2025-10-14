# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE
from models.demos.wormhole.stable_diffusion.sd_helper_funcs import get_refference_vae
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae import Vae
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": SD_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    """input_channels, input_height, input_width, out_channels, output_height, output_width""",
    [
        (
            4,  # input_channels
            64,  # input_height
            64,  # input_width
            3,  # out_channels
            512,  # output_height
            512,  # output_width
        ),
    ],
)
def test_decoder(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    is_ci_env,
    is_ci_v2_env,
    model_location_generator,
):
    torch.manual_seed(0)

    vae = get_refference_vae(is_ci_env, is_ci_v2_env, model_location_generator)

    # Run pytorch model
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = vae.decode(torch_input).sample

    # Initialize ttnn model
    ttnn_model = Vae(torch_vae=vae, device=device)

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model.decode(ttnn_input)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model.decode(ttnn_input)

    ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    # TODO: Improve PCC (issue #21131)
    assert_with_pcc(torch_output, ttnn_output, 0.985)

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr
from models.demos.stable_diffusion_xl_base.quasar.tests.sdxl_quasar_test_utils import load_torch_unet, to_device_nhwc
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.quasar.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, temb_shape, residuals, encoder_shape, query_dim, num_attn_heads, out_dim, block_id, pcc",
    [
        # 1024x1024 image resolution
        (
            (1024, 1024),
            (1, 1280, 32, 32),
            (1, 1280),
            ((1, 640, 32, 32), (1, 1280, 32, 32), (1, 1280, 32, 32)),
            (1, 77, 2048),
            1280,
            20,
            1280,
            0,
            0.970,
        ),
        (
            (1024, 1024),
            (1, 1280, 64, 64),
            (1, 1280),
            ((1, 320, 64, 64), (1, 640, 64, 64), (1, 640, 64, 64)),
            (1, 77, 2048),
            640,
            10,
            640,
            1,
            0.993,
        ),
    ],
)
def test_crossattnup(
    device,
    image_resolution,
    input_shape,
    temb_shape,
    residuals,
    encoder_shape,
    query_dim,
    num_attn_heads,
    out_dim,
    block_id,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    unet = load_torch_unet(sdxl_base_unet_location, is_ci_env, is_ci_v2_env)
    state_dict = unet.state_dict()

    torch_crosattn = unet.up_blocks[block_id]

    model_config = load_model_optimisations(image_resolution)
    tt_crosattn = TtCrossAttnUpBlock2D(
        device,
        state_dict,
        f"up_blocks.{block_id}",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
        True,
        debug_mode=debug_mode,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_residual_tensors = ()
    for r in residuals:
        residual = torch_random(r, -0.1, 0.1, dtype=torch.float32)
        torch_residual_tensors = torch_residual_tensors + (residual,)

    torch_output_tensor = torch_crosattn(
        torch_input_tensor, torch_residual_tensors, temb=torch_temb_tensor, encoder_hidden_states=torch_encoder_tensor
    )

    ttnn_input_tensor, [B, C, H, W] = to_device_nhwc(torch_input_tensor, device)

    ttnn_residual_tensors = ()
    for torch_residual in torch_residual_tensors:
        ttnn_residual, [Br, Cr, Hr, Wr] = to_device_nhwc(torch_residual, device)
        ttnn_residual_tensors = ttnn_residual_tensors + (ttnn_residual,)

    ttnn_temb_tensor = qsr.from_torch(
        torch.nn.functional.silu(torch_temb_tensor), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    ttnn_encoder_tensor = qsr.from_torch(
        torch_encoder_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    ttnn_output_tensor, output_shape = tt_crosattn.forward(
        ttnn_input_tensor,
        ttnn_residual_tensors,
        [B, C, H, W],
        temb=ttnn_temb_tensor,
        encoder_hidden_states=ttnn_encoder_tensor,
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet, tt_crosattn
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is: {pcc_message}")

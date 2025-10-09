# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from diffusers import UNet2DConditionModel
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.stable_diffusion_xl_refiner.tt.tt_upblock2d import TtUpBlock2D
from models.experimental.stable_diffusion_xl_refiner.tests.test_common import SDXL_REFINER_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, temb_shape, residuals, block_id, pcc",
    [
        (
            (1, 1536, 16, 16),
            (1, 1536),
            ((1, 1536, 16, 16), (1, 1536, 16, 16), (1, 1536, 16, 16)),
            0,
            0.996,
        ),
        (
            (1, 768, 128, 128),
            (1, 1536),
            ((1, 384, 128, 128), (1, 384, 128, 128), (1, 384, 128, 128)),
            3,
            0.999,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_REFINER_L1_SMALL_SIZE}], indirect=True)
def test_upblock2d_refiner(
    device,
    input_shape,
    temb_shape,
    residuals,
    block_id,
    pcc,
    is_ci_env,
):
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_upblock = unet.up_blocks[block_id]
    module_path = f"up_blocks.{block_id}"

    tt_upblock = TtUpBlock2D(
        device=device,
        state_dict=state_dict,
        module_path=module_path,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)

    torch_residual_tensors = ()
    for r in residuals:
        residual = torch_random(r, -0.1, 0.1, dtype=torch.float32)
        torch_residual_tensors = torch_residual_tensors + (residual,)

    torch_output_tensor = torch_upblock(torch_input_tensor, torch_residual_tensors, torch_temb_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_temb_tensor = ttnn.from_torch(
        torch_temb_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_residual_tensors = ()
    for torch_residual in torch_residual_tensors:
        ttnn_residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        Br, Cr, Hr, Wr = list(ttnn_residual.shape)

        ttnn_residual = ttnn.permute(ttnn_residual, (0, 2, 3, 1))
        ttnn_residual = ttnn.reshape(ttnn_residual, (Br, 1, Hr * Wr, Cr))

        ttnn_residual_tensors = ttnn_residual_tensors + (ttnn_residual,)

    ttnn_output_tensor, output_shape = tt_upblock.forward(
        ttnn_input_tensor, [B, C, H, W], ttnn_residual_tensors, ttnn_temb_tensor
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")

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
from models.experimental.stable_diffusion_xl_refiner.tt.tt_upsample2d import TtUpsample2D
from models.experimental.stable_diffusion_xl_refiner.tests.test_common import SDXL_REFINER_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, block_id, block, pcc",
    [
        ((1, 1536, 16, 16), 0, "up_blocks", 0.999),
        ((1, 1536, 32, 32), 1, "up_blocks", 0.999),
        ((1, 768, 64, 64), 2, "up_blocks", 0.999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_REFINER_L1_SMALL_SIZE}], indirect=True)
def test_upsample2d_refiner(
    device,
    input_shape,
    block_id,
    block,
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

    if block == "up_blocks":
        torch_upsample = unet.up_blocks[block_id].upsamplers[0]
    else:
        assert "Incorrect block name"

    module_path = f"{block}.{block_id}.upsamplers.0"

    tt_upsample = TtUpsample2D(
        device=device,
        state_dict=state_dict,
        module_path=module_path,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_upsample(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))

    ttnn_output_tensor, output_shape = tt_upsample.forward(ttnn_input_tensor)

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")

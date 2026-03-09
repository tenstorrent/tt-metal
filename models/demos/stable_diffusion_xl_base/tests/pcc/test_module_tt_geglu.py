# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from functools import reduce

import pytest
import torch
from diffusers import UNet2DConditionModel
from loguru import logger

import ttnn
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, module_path, pcc",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1024, 1280), "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 0.948),
        ((1024, 1024), (4096, 640), "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 0.950),
        # 512x512 image resolution
        ((512, 512), (256, 1280), "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 0.948),
        ((512, 512), (1024, 640), "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 0.950),
    ],
)
def test_geglu(
    device,
    image_resolution,
    input_shape,
    module_path,
    pcc,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_base_unet_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    try:
        torch_geglu = reduce(
            lambda obj, key: obj[int(key)] if key.isdigit() else getattr(obj, key), module_path.split("."), unet
        )
    except (AttributeError, IndexError, TypeError) as e:
        torch_geglu = None

    assert torch_geglu is not None, f"{module_path} is not a valid UNet module"

    model_config = load_model_optimisations(image_resolution)
    tt_geglu = TtGEGLU(device, state_dict, module_path, model_config)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_geglu(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_geglu.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor).squeeze()

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")

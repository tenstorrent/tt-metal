# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from functools import reduce

import pytest
import torch
import ttnn
from diffusers import DiffusionPipeline
from loguru import logger

from models.experimental.stable_diffusion_xl_base.lora.lora_weights_manager import TtLoRAWeightsManager
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc

LORA_PATH = "lora_weights/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"


def _get_base_state_dict_and_pipeline_with_lora(is_ci_env):
    """Load pipeline, get base UNet state_dict, load LoRA (do not fuse yet). Returns (base_state_dict, pipeline)."""
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    pipeline.unet.eval()
    base_state_dict = {k: v.clone() for k, v in pipeline.unet.state_dict().items()}
    pipeline.load_lora_weights(LORA_PATH)
    return base_state_dict, pipeline


@pytest.mark.parametrize(
    "input_shape, module_path, pcc",
    [
        ((1024, 1280), "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 0.948),
        ((4096, 640), "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 0.950),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_lora_fusion_pcc_geglu(device, input_shape, module_path, pcc, is_ci_env, reset_seeds):
    """Compare TT GEGLU (with fused LoRA) vs Diffusers GEGLU (with fused LoRA) via output PCC."""
    base_state_dict, pipeline = _get_base_state_dict_and_pipeline_with_lora(is_ci_env)

    lora_mgr = TtLoRAWeightsManager(device, pipeline)
    tt_geglu = TtGEGLU(device, base_state_dict, module_path, ModelOptimisations(), lora_weights_manager=lora_mgr)
    lora_mgr.fuse_lora_weights(lora_scale=1.0)
    pipeline.fuse_lora()

    try:
        torch_geglu = reduce(
            lambda obj, key: obj[int(key)] if key.isdigit() else getattr(obj, key),
            module_path.split("."),
            pipeline.unet,
        )
    except (AttributeError, IndexError, TypeError):
        torch_geglu = None
    assert torch_geglu is not None, f"{module_path} is not a valid UNet module"

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

    del pipeline, tt_geglu, lora_mgr
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"LoRA GEGLU PCC is {pcc_message}")

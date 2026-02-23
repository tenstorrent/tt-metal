# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
import pytest
import torch
import ttnn
from diffusers import DiffusionPipeline
from loguru import logger

from models.experimental.stable_diffusion_xl_base.lora.lora_weights_manager import TtLoRAWeightsManager
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_feedforward import TtFeedForward
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
    "input_shape, block_id, transformer_block_id, pcc",
    [
        ((1024, 1280), 2, 0, 0.997),
        ((4096, 640), 1, 0, 0.999),
        ((4096, 640), 1, 1, 0.998),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_lora_fusion_pcc_feedforward(device, input_shape, block_id, transformer_block_id, pcc, is_ci_env, reset_seeds):
    """Compare TT FeedForward (with fused LoRA) vs Diffusers FeedForward (with fused LoRA) via output PCC."""
    base_state_dict, pipeline = _get_base_state_dict_and_pipeline_with_lora(is_ci_env)

    lora_mgr = TtLoRAWeightsManager(device, pipeline)
    module_path = f"down_blocks.{block_id}.attentions.0.transformer_blocks.{transformer_block_id}.ff"
    tt_ff = TtFeedForward(device, base_state_dict, module_path, ModelOptimisations(), lora_weights_manager=lora_mgr)
    lora_mgr.fuse_lora_weights(lora_scale=1.0)
    pipeline.fuse_lora()

    torch_ff = pipeline.unet.down_blocks[block_id].attentions[0].transformer_blocks[transformer_block_id].ff

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_ff(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_ff.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor).squeeze()

    del pipeline, tt_ff, lora_mgr
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"LoRA FeedForward PCC is {pcc_message}")

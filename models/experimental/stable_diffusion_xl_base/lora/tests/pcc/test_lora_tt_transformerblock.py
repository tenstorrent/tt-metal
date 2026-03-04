# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
import pytest
import torch
import ttnn
from diffusers import DiffusionPipeline
from loguru import logger

from models.experimental.stable_diffusion_xl_base.lora.tt_lora_weights_manager import TtLoRAWeightsManager
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations1024x1024
from models.experimental.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


def _get_diffusers_pipeline(is_ci_env):
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    return pipeline


@pytest.mark.parametrize(
    "input_shape, encoder_shape, down_block_id, block_id, query_dim, num_attn_heads, out_dim, pcc",
    [
        ((1, 4096, 640), (1, 77, 2048), 1, 0, 640, 10, 640, 0.999),
        ((1, 4096, 640), (1, 77, 2048), 1, 1, 640, 10, 640, 0.998),
        ((1, 1024, 1280), (1, 77, 2048), 2, 0, 1280, 20, 1280, 0.999),
        ((1, 1024, 1280), (1, 77, 2048), 2, 1, 1280, 20, 1280, 0.998),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_lora_fusion_pcc_transformerblock(
    device,
    input_shape,
    encoder_shape,
    down_block_id,
    block_id,
    query_dim,
    num_attn_heads,
    out_dim,
    pcc,
    is_ci_env,
    reset_seeds,
    lora_path,
):
    pipeline = _get_diffusers_pipeline(is_ci_env)
    pipeline.unet.eval()

    pipeline_for_tt = _get_diffusers_pipeline(is_ci_env)
    state_dict = pipeline_for_tt.unet.state_dict()

    lora_mgr = TtLoRAWeightsManager(device, pipeline_for_tt)
    tt_transformerblock = TtBasicTransformerBlock(
        device,
        state_dict,
        f"down_blocks.{down_block_id}.attentions.0.transformer_blocks.{block_id}",
        ModelOptimisations1024x1024(),
        query_dim,
        num_attn_heads,
        out_dim,
        lora_weights_manager=lora_mgr,
    )

    lora_mgr.load_lora_weights(lora_path)
    lora_mgr.fuse_lora(lora_scale=1.0)
    pipeline.load_lora_weights(lora_path)
    pipeline.fuse_lora(lora_scale=1.0)

    torch_transformerblock = pipeline.unet.down_blocks[down_block_id].attentions[0].transformer_blocks[block_id]

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output_tensor = torch_transformerblock(torch_input_tensor, None, torch_encoder_tensor).unsqueeze(0)

    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_transformerblock.forward(ttnn_input_tensor, None, ttnn_encoder_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del pipeline, pipeline_for_tt, tt_transformerblock, lora_mgr
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")

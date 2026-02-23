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
from models.experimental.stable_diffusion_xl_base.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
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
    "input_shape, temb_shape, residuals, encoder_shape, query_dim, num_attn_heads, out_dim, block_id, pcc",
    [
        (
            (1, 1280, 32, 32),
            (1, 1280),
            ((1, 640, 32, 32), (1, 1280, 32, 32), (1, 1280, 32, 32)),
            (1, 77, 2048),
            1280,
            20,
            1280,
            0,
            0.972,
        ),
        (
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_lora_fusion_pcc_crossattnup(
    device,
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
    reset_seeds,
):
    """Compare TT CrossAttnUpBlock2D (with fused LoRA) vs Diffusers (with fused LoRA) via output PCC."""
    base_state_dict, pipeline = _get_base_state_dict_and_pipeline_with_lora(is_ci_env)

    lora_mgr = TtLoRAWeightsManager(device, pipeline)
    tt_crosattn = TtCrossAttnUpBlock2D(
        device,
        base_state_dict,
        f"up_blocks.{block_id}",
        ModelOptimisations(),
        query_dim,
        num_attn_heads,
        out_dim,
        True,
        debug_mode=debug_mode,
        lora_weights_manager=lora_mgr,
    )
    lora_mgr.fuse_lora_weights(lora_scale=1.0)
    pipeline.fuse_lora()

    torch_crosattn = pipeline.unet.up_blocks[block_id]

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

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_residual_tensors = ()
    for torch_residual in torch_residual_tensors:
        ttnn_residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        Br, Cr, Hr, Wr = list(ttnn_residual.shape)
        ttnn_residual = ttnn.permute(ttnn_residual, (0, 2, 3, 1))
        ttnn_residual = ttnn.reshape(ttnn_residual, (Br, 1, Hr * Wr, Cr))
        ttnn_residual_tensors = ttnn_residual_tensors + (ttnn_residual,)

    ttnn_temb_tensor = ttnn.from_torch(torch_temb_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_temb_tensor = ttnn.silu(ttnn_temb_tensor)
    ttnn_encoder_tensor = ttnn.from_torch(
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

    del pipeline, tt_crosattn, lora_mgr
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"LoRA CrossAttnUpBlock2D PCC is {pcc_message}")

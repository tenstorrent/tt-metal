# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from functools import reduce

import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.lora.tt_lora_weights_manager import TtLoRAWeightsManager
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
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
    "image_resolution, input_shape, module_path, pcc",
    [
        ((1024, 1024), (1024, 1280), "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 0.948),
        ((1024, 1024), (4096, 640), "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 0.950),
        ((512, 512), (256, 1280), "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 0.948),
        ((512, 512), (1024, 640), "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 0.950),
    ],
)
def test_geglu(device, image_resolution, input_shape, module_path, pcc, is_ci_env, reset_seeds, lora_path):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")

    pipeline = _get_diffusers_pipeline(is_ci_env)
    pipeline.unet.eval()

    pipeline_for_tt = _get_diffusers_pipeline(is_ci_env)
    state_dict = pipeline_for_tt.unet.state_dict()

    lora_mgr = TtLoRAWeightsManager(device, pipeline_for_tt)
    tt_geglu = TtGEGLU(
        device, state_dict, module_path, load_model_optimisations(image_resolution), lora_weights_manager=lora_mgr
    )

    lora_mgr.load_lora_weights(lora_path)
    lora_mgr.fuse_lora(lora_scale=1.0)
    pipeline.load_lora_weights(lora_path)
    pipeline.fuse_lora(lora_scale=1.0)

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

    del pipeline, pipeline_for_tt, tt_geglu, lora_mgr
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")

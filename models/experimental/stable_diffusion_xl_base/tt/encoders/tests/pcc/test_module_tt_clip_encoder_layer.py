# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_encoder_layer import TtClipEncoderLayer
from diffusers import DiffusionPipeline
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, encoder_id, num_attention_heads",
    [
        ((1, 77, 768), 1, 12),
        # ((1, 77, 1280), 2, 20),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_layer(
    device,
    input_shape,
    encoder_id,
    num_attention_heads,
):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    text_encoder = pipe.text_encoder if encoder_id == 1 else pipe.text_encoder_2
    text_encoder.eval()
    state_dict = text_encoder.state_dict()

    torch_layer = text_encoder.text_model.encoder.layers[0]

    model_config = ModelOptimisations()
    tt_layer = TtClipEncoderLayer(
        device,
        state_dict,
        "text_model.encoder.layers.0",
        model_config,
        num_attention_heads=num_attention_heads,
        hidden_size=input_shape[-1],
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    causal_mask = _create_4d_causal_attention_mask(
        input_shape[:1], torch_input_tensor.dtype, device=torch_input_tensor.device
    )

    torch_output_tensor = torch_layer(torch_input_tensor, attention_mask=None, causal_attention_mask=causal_mask)[
        0
    ].unsqueeze(0)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor = tt_layer.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del pipe, tt_layer
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is: {pcc_message}")

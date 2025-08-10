# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_embeddings import TtClipEmbeddings
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape, encoder_id",
    [
        ((1, 77), 1),
        ((1, 77), 2),
    ],
)
def test_embeddings(device, input_shape, encoder_id, is_ci_env, reset_seeds):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    text_encoder = pipe.text_encoder if encoder_id == 1 else pipe.text_encoder_2
    text_encoder.eval()
    state_dict = text_encoder.state_dict()

    torch_embeddings = text_encoder.text_model.embeddings

    model_config = ModelOptimisations()
    tt_embeddings = TtClipEmbeddings(
        device,
        state_dict,
        f"text_model.embeddings",
        model_config,
    )

    torch_input_ids = torch.randint(low=0, high=tt_embeddings.vocab_size, size=input_shape, dtype=torch.int64)
    torch_output_tensor = torch_embeddings(input_ids=torch_input_ids)

    tt_text_input_ids = ttnn.from_torch(
        torch_input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn_output_tensor = tt_embeddings.forward(input_ids=tt_text_input_ids)
    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    del pipe
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is {pcc_message}")

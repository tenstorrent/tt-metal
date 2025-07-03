# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn
from loguru import logger
from transformers.models.t5.modeling_t5 import T5EncoderModel

from models.experimental.flux.reference.t5_encoder import T5Config
from models.experimental.flux.reference.t5_encoder import T5Encoder as T5EncoderReference
from models.experimental.flux.tt.t5_encoder import T5Encoder, T5EncoderParameters
from models.experimental.flux.tt.utils import assert_quality


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_t5_encoder(*, mesh_device: ttnn.Device, model_location_generator) -> None:
    # if use_program_cache:
    #    ttnn.enable_program_cache(device)

    checkpoint = "black-forest-labs/FLUX.1-schnell"

    model_name_checkpoint = model_location_generator(checkpoint, model_subdir="Flux1_Schnell")

    hf_model = T5EncoderModel.from_pretrained(
        model_name_checkpoint,
        subfolder="text_encoder_2",
    )
    with torch.device("meta"):
        torch_model = T5EncoderReference(
            T5Config(
                vocab_size=hf_model.config.vocab_size,
                d_model=hf_model.config.d_model,
                d_ff=hf_model.config.d_ff,
                d_kv=hf_model.config.d_kv,
                num_layers=hf_model.config.num_layers,
                num_heads=hf_model.config.num_heads,
                relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
                relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
                layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
            )
        )
    torch_model.load_state_dict(hf_model.state_dict(), assign=True)
    torch_model.eval()

    start_time = time.time()
    parameters = T5EncoderParameters.from_torch(torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b)
    tt_model = T5Encoder(
        parameters,
        num_heads=hf_model.config.num_heads,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
        layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
    )
    logger.info(f"model creation time: {time.time() - start_time}")

    torch.manual_seed(0)
    tokens = torch.randint(hf_model.config.vocab_size, [1, 256])

    tt_tokens_host = ttnn.from_torch(
        tokens,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None)),
    )

    start_time = time.time()
    with torch.no_grad():
        output = torch_model(tokens)
    logger.info(f"CPU runtime: {time.time() - start_time}")

    tt_tokens = tt_tokens_host.to(mesh_device)

    logger.info("compiling...")
    tt_model.forward(tt_tokens)

    logger.info("executing...")
    start_time = time.time()
    tt_output = tt_model.forward(tt_tokens)
    logger.info(f"TT-NN runtime: {time.time() - start_time}")
    logger.info("done...")

    # tt_output_torch = ttnn.to_torch(tt_output)

    # assert output.shape == tt_output_torch.shape

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))
    assert_quality(output, tt_output, pcc=0.945, mesh_composer=composer)

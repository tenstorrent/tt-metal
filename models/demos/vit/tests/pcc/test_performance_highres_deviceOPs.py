# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import time

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0, torch_random
from models.demos.vit.common import load_torch_model
from models.demos.vit.tt import ttnn_functional_vit_highres
from models.demos.vit.tt import ttnn_optimized_vit_highres_gs as ttnn_optimized_vit_highres


def get_expected_times(functional_vit):
    return {
        ttnn_functional_vit_highres: (12, 17),
        ttnn_optimized_vit_highres: (12, 0.08),
    }[functional_vit]


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, patch_size, num_patches, height: int, width: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.

    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """
    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, 0]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size

    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [448])
@pytest.mark.parametrize("functional_vit", [ttnn_functional_vit_highres, ttnn_optimized_vit_highres])
def test_performance_vit_encoder(
    device, model_name, batch_size, sequence_size, functional_vit, model_location_generator
):
    model = load_torch_model(model_location_generator)
    config = model.config
    model = model.vit.encoder

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)

    if functional_vit == ttnn_functional_vit_highres:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_vit == ttnn_optimized_vit_highres:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    if functional_vit == ttnn_optimized_vit_highres:
        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if torch_attention_mask is not None:
            head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(config.num_hidden_layers)
            ]
        else:
            head_masks = [None for _ in range(config.num_hidden_layers)]
    else:
        hidden_states = ttnn.from_torch(
            torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        if torch_attention_mask is not None:
            head_masks = ttnn.from_torch(
                torch_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            head_masks = None
        head_masks = None

    durations = []
    for _ in range(1):
        start = time.time()
        tt_output = functional_vit.vit_encoder(
            config,
            hidden_states,
            head_masks,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)

    inference_time = durations

    logger.info(f"Inference time: {inference_time}")


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [960])
@pytest.mark.parametrize("sequence_size", [448])
@pytest.mark.parametrize("functional_vit", [ttnn_functional_vit_highres, ttnn_optimized_vit_highres])
def test_performance_vit_e2e(
    device, model_name, batch_size, image_size, sequence_size, functional_vit, model_location_generator
):
    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config
    config = model.config

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0].resize((image_size, image_size))

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(
        image, return_tensors="pt", do_resize=False, do_center_crop=False
    ).pixel_values.to(torch.bfloat16)

    torch_attention_mask = (
        None  # torch.zeros(1, sequence_size) if functional_vit == ttnn_optimized_functional_vit else None
    )

    if functional_vit == ttnn_functional_vit_highres:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_vit == ttnn_optimized_vit_highres:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    # High resolution patch_parameters interpolation
    model_state_dict = model.state_dict()
    init_position_embeddings = torch.nn.Parameter(model_state_dict["vit.embeddings.position_embeddings"])
    patch_size = 16
    tot_patch_count = (image_size // patch_size) * (image_size // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size, image_size)
    )
    position_embeddings = ttnn.from_torch(torch_position_embeddings, layout=ttnn.TILE_LAYOUT, device=device)
    torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    durations = []
    for _ in range(1):
        start = time.time()
        tt_output = functional_vit.vit(
            config,
            pixel_values,
            head_masks,
            position_embeddings,
            cls_token,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)

    inference_time, *_ = durations

    logger.info(f"Inference time: {inference_time}")

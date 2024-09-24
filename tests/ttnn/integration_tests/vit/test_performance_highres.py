# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import math
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor

import ttnn

from models.experimental.functional_vit.tt import ttnn_functional_vit_highres
from models.experimental.functional_vit.tt import ttnn_optimized_vit_highres

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
)
from models.perf.perf_utils import prep_perf_report


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

    # num_patches = embeddings.shape[1] - 1
    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, 0]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
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
def test_performance_vit_encoder(device, use_program_cache, model_name, batch_size, sequence_size, functional_vit):
    # disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    ).vit.encoder

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
    for _ in range(2):
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
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(functional_vit)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


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
    device, use_program_cache, model_name, batch_size, image_size, sequence_size, functional_vit
):
    disable_persistent_kernel_cache()

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0].resize((image_size, image_size))

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(
        image, return_tensors="pt", do_resize=False, do_center_crop=False
    ).pixel_values.to(torch.bfloat16)

    # torch_pixel_values = torch.rand((1, 3, 960, 960))
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
        # memory_config=ttnn.L1_MEMORY_CONFIG,
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
    for _ in range(2):
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
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(functional_vit)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")

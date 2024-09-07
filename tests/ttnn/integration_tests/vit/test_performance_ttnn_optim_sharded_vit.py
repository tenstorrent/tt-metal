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
from models.experimental.functional_vit.tt import ttnn_optimized_sharded_vit
from models.utility_functions import torch_random, is_wormhole_b0, is_blackhole

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    is_wormhole_b0,
    torch_random,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_vit):
    return {
        ttnn_optimized_sharded_vit: (12, 0.08),
    }[functional_vit]


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [196])  ## padded from 197 to 224
@pytest.mark.parametrize("functional_vit", [ttnn_optimized_sharded_vit])
def test_performance_vit_encoder(device, use_program_cache, model_name, batch_size, sequence_size, functional_vit):
    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    ).vit.encoder

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)

    if functional_vit == ttnn_optimized_sharded_vit:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

    durations = []
    for _ in range(2):
        start = time.time()
        device_hidden_states = ttnn.to_device(hidden_states, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        device_head_masks = [
            ttnn.to_device(head_mask, device=device, memory_config=ttnn.L1_MEMORY_CONFIG) for head_mask in head_masks
        ]

        tt_output = functional_vit.vit_encoder(
            config,
            device_hidden_states,
            device_head_masks,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)

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


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("sequence_size", [224])
@pytest.mark.parametrize("functional_vit", [ttnn_optimized_sharded_vit])
def test_performance_vit_e2e(
    device, use_program_cache, model_name, batch_size, image_size, sequence_size, functional_vit
):
    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"]
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

    # cls_token expand to batch_size
    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["vit.embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    # torch_cls_token_padded = torch.nn.functional.pad(torch_cls_token, (0, 0, 0, 196, 0, 0))
    # torch_cls_position_embeddings = torch.add(torch_cls_token_padded, torch_position_embeddings)
    # cls_position_embeddings = ttnn.from_torch(
    #     torch_cls_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    # )

    if functional_vit == ttnn_optimized_sharded_vit:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_vit: {functional_vit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        custom_preprocessor=functional_vit.custom_preprocessor,
        device=device,
    )

    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
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

    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size)

    durations = []
    import tracy

    patch_size = 16
    batch_size, _, img_h, img_w = torch_pixel_values.shape
    N, H, W, C = batch_size, img_h, img_w // patch_size, 4 * patch_size
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 0),
            ),
        }
    )
    n_cores = 8
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, False)

    # tracyProfiler = tracy.Profiler()
    # tracyProfiler.enable()
    for _ in range(10):
        start = time.time()
        pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
        pixel_values = pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        pixel_values = ttnn.from_torch(
            pixel_values,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec),
            dtype=ttnn.bfloat16,
        )

        tt_output = functional_vit.vit(
            config,
            pixel_values,
            head_masks,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output, blocking=False)
        end = time.time()
        durations.append(end - start)

    # tracyProfiler.disable()
    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)

    expected_compile_time, expected_inference_time = get_expected_times(functional_vit)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference time: {average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Samples per second: {1 / average_inference_time * batch_size}")

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import math
import transformers
from datasets import load_dataset

# from transformers import DeiTImageProcessor

import ttnn
import tt_lib
from models.experimental.functional_deit.tt import ttnn_optimized_sharded_deit
from models.utility_functions import torch_random, skip_for_wormhole_b0

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    torch_random,
)
from models.perf.perf_utils import prep_perf_report
import tracy


def get_expected_times(functional_deit):
    return {
        ttnn_optimized_sharded_deit: (12.4, 0.04),
    }[functional_deit]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["facebook/deit-base-distilled-patch16-224"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("image_size", [224])
@pytest.mark.parametrize("sequence_size", [224])
@pytest.mark.parametrize("functional_deit", [ttnn_optimized_sharded_deit])
def test_performance_deit_e2e(
    device, use_program_cache, model_name, batch_size, image_size, sequence_size, functional_deit
):
    config = transformers.DeiTConfig.from_pretrained(model_name)
    model = transformers.DeiTForImageClassificationWithTeacher.from_pretrained(model_name, config=config)

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"]
    image_processor = transformers.DeiTImageProcessor.from_pretrained(model_name)
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(torch.bfloat16)
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

    # cls_token expand to batch_size
    model_state_dict = model.state_dict()
    torch_cls_token = model_state_dict["deit.embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["deit.embeddings.position_embeddings"]
    torch_distillation_token = model_state_dict["deit.embeddings.distillation_token"]

    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
        torch_distillation_token = torch.nn.Parameter(torch_distillation_token)

    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    distillation_tokens = ttnn.from_torch(
        torch_distillation_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # torch_cls_token_padded = torch.nn.functional.pad(torch_cls_token, (0, 0, 0, 196, 0, 0))
    # torch_cls_position_embeddings = torch.add(torch_cls_token_padded, torch_position_embeddings)
    # cls_position_embeddings = ttnn.from_torch(
    #     torch_cls_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    # )

    if functional_deit == ttnn_optimized_sharded_deit:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_deit: {functional_deit}")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        custom_preprocessor=functional_deit.custom_preprocessor,
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

    config = ttnn_optimized_sharded_deit.update_model_config(config, batch_size)

    durations = []

    patch_size = 16
    batch_size, _, img_h, img_w = torch_pixel_values.shape
    N, H, W, C = batch_size, img_h, img_w // patch_size, 4 * patch_size
    shard_grid = tt_lib.tensor.CoreRangeSet(
        {
            tt_lib.tensor.CoreRange(
                tt_lib.tensor.CoreCoord(0, 0),
                tt_lib.tensor.CoreCoord(7, 0),
            ),
        }
    )
    n_cores = 8
    shard_spec = tt_lib.tensor.ShardSpec(
        shard_grid, [N * H * W // n_cores, C], tt_lib.tensor.ShardOrientation.ROW_MAJOR, False
    )

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
            memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1, shard_spec
            ),
            dtype=ttnn.bfloat16,
        )

        tt_output = functional_deit.deit_for_image_classification_teacher(
            config,
            pixel_values,
            head_masks,
            cls_token,
            position_embeddings,
            distillation_tokens,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output, blocking=False)
        end = time.time()
        durations.append(end - start)

    # tracyProfiler.disable()
    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)

    expected_compile_time, expected_inference_time = get_expected_times(functional_deit)
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

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import torch
from loguru import logger
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.classification.vit.blackhole.tt import ttnn_optimized_sharded_vit_bh
from models.demos.vision.classification.vit.common.common import load_torch_model
from models.demos.vision.classification.vit.common.tests.vit_helper_funcs import get_batch, get_data_loader
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_vit):
    return {
        ttnn_optimized_sharded_vit_bh: (11, 0.02),
    }[functional_vit]


import os

os.environ["TTNN_CONFIG_OVERRIDES"] = '{"enable_fast_runtime_mode": true}'


def test_vit(device, model_location_generator):
    torch.manual_seed(0)

    model_name = "google/vit-base-patch16-224"
    batch_size = 10

    model = load_torch_model(model_location_generator, embedding=True)
    config = model.config
    config = ttnn_optimized_sharded_vit_bh.update_model_config(config, batch_size, device)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_sharded_vit_bh.custom_preprocessor,
    )

    # cls_token & position embeddings expand to batch_size
    # TODO: pass batch_size to preprocess_model_parameters
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

    # IMAGENET INFERENCE
    #####################
    data_loader = get_data_loader(batch_size, 2)

    durations = []
    for iter in range(2):
        torch_pixel_values, labels = get_batch(data_loader, image_processor)
        start = time.time()
        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
        patch_size = config.patch_size
        torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
        pixel_values = ttnn.from_torch(
            torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn_optimized_sharded_vit_bh.pixel_values_memory_config(
                config, device, torch_pixel_values.shape
            ),
        )

        output = ttnn_optimized_sharded_vit_bh.vit(
            config,
            pixel_values,
            cls_token,
            position_embeddings,
            parameters=parameters,
        )
        output = ttnn.from_device(output)

        end = time.time()
        durations.append(end - start)

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(ttnn_optimized_sharded_vit_bh)
    prep_perf_report(
        model_name="vit_ttnn_optim_sharded",
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

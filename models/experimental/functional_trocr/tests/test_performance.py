# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
import torch
import ttnn

from loguru import logger
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.ttnn_trocr_generation_utils import GenerationMixin

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
)

from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_trocr.tt.ttnn_encoder_vit import custom_preprocessor


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["microsoft/trocr-base-handwritten"])
@pytest.mark.parametrize("image_size", [384])
def test_demo(device, model_name, image_size):
    disable_persistent_kernel_cache()
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        config = model.decoder.config

        processor = TrOCRProcessor.from_pretrained(model_name)

        model.eval()

        pixel_values = torch_random((1, 3, image_size, image_size), -1, 1, dtype=torch.float32)

        torch_generated_ids = model.generate(pixel_values)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, device=device, custom_preprocessor=custom_preprocessor
        )

        generationmixin = GenerationMixin(model=model, device=device, config=config, parameters=parameters)

        inputs_tensor = torch.permute(pixel_values, (0, 2, 3, 1))
        inputs_tensor = torch.nn.functional.pad(inputs_tensor, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = inputs_tensor.shape  # permuted input NHWC
        patch_size = 16
        sequence_size = 384
        inputs_tensor = inputs_tensor.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        cls_token = model.encoder.state_dict()["embeddings.cls_token"]
        torch_position_embeddings = model.encoder.state_dict()["embeddings.position_embeddings"]
        vit_config = model.encoder.config
        torch_attention_mask = None

        if torch_attention_mask is not None:
            head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(vit_config.num_hidden_layers)
            ]
        else:
            head_masks = [None for _ in range(vit_config.num_hidden_layers)]

        if batch_size > 1:
            cls_token = torch.nn.Parameter(cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            cls_token = torch.nn.Parameter(cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)

        cls_token = ttnn.from_torch(cls_token, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        inputs_tensor = ttnn.from_torch(inputs_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        durations = []
        for _ in range(2):
            start = time.time()
            ttnn_output = generationmixin.generate(
                inputs_tensor, cls_token=cls_token, position_embeddings=position_embeddings, head_masks=head_masks
            )
            end = time.time()
            durations.append(end - start)
            enable_persistent_kernel_cache()

        inference_and_compile_time, inference_time, *_ = durations

        expected_compile_time, expected_inference_time = inference_and_compile_time, inference_time
        prep_perf_report(
            model_name=f"tt_{model_name}",
            batch_size=1,
            inference_and_compile_time=inference_and_compile_time,
            inference_time=inference_time,
            expected_compile_time=expected_compile_time,
            expected_inference_time=expected_inference_time,
            comments="",
            inference_time_cpu=0.0,
        )

        logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
        logger.info(f"Inference time: {inference_time}")
        logger.info(f"Samples per second: {1 / inference_time}")

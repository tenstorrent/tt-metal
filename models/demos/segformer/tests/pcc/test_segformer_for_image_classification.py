# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_for_image_classification import SegformerForImageClassificationReference
from models.demos.segformer.tests.pcc.test_segformer_model import (
    create_custom_preprocessor as custom_preprocessor_main_model,
)
from models.demos.segformer.tests.pcc.test_segformer_model import move_to_device
from models.demos.segformer.tt.ttnn_segformer_for_image_classification import TtSegformerForImageClassification
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerForImageClassificationReference):
            parameters["segformer"] = {}
            custom_preprocessor_main_model_obj = custom_preprocessor_main_model(device)
            parameters["segformer"] = custom_preprocessor_main_model_obj(
                model.segformer, name=name, ttnn_module_args=ttnn_module_args
            )
            parameters["classifier"] = {}
            parameters["classifier"]["weight"] = ttnn.from_torch(
                model.classifier.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            parameters["classifier"]["bias"] = ttnn.from_torch(
                model.classifier.bias.reshape(1, 1, 1, model.classifier.bias.shape[-1]),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_image_classificaton(device, model_location_generator):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["train"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    inputs = image_processor(image, return_tensors="pt")
    torch_input_tensor = inputs.pixel_values
    torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor_permuted,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    CONV2D_MIN_CHANNEL_SIZE = 8
    # adjust padding if necessary
    if ttnn_input_tensor.shape[3] < CONV2D_MIN_CHANNEL_SIZE:
        padded_shape = [
            ttnn_input_tensor.shape[0],
            ttnn_input_tensor.shape[1],
            ttnn_input_tensor.shape[2],
            CONV2D_MIN_CHANNEL_SIZE,
        ]
        ttnn_input_tensor = ttnn.pad(ttnn_input_tensor, padded_shape, [0, 0, 0, 0], 0)
    elif ttnn_input_tensor.shape[3] > CONV2D_MIN_CHANNEL_SIZE and ttnn_input_tensor.shape[3] % 32 != 0:
        padded_shape = [
            ttnn_input_tensor.shape[0],
            ttnn_input_tensor.shape[1],
            ttnn_input_tensor.shape[2],
            (ttnn_input_tensor.shape[3] + 31) // 32 * 32,
        ]
        ttnn_input_tensor = ttnn.pad(ttnn_input_tensor, padded_shape, [0, 0, 0, 0], 0)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    config = load_config("configs/segformer_img_classification_config.json")
    reference_model = SegformerForImageClassificationReference(config)
    target_prefix = f""
    reference_model = load_torch_model(
        reference_model, f"", module="image_classification", model_location_generator=model_location_generator
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters = move_to_device(parameters, device)
    torch_output = reference_model(torch_input_tensor)

    ttnn_model = TtSegformerForImageClassification(config, parameters)
    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
        model=reference_model,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output.logits)
    assert_with_pcc(torch_output.logits, ttnn_final_output, pcc=0.968)

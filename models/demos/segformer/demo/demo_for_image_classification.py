# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import AutoImageProcessor, SegformerForImageClassification
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.demo.classification_demo_utils import get_batch, get_data_loader
from models.demos.segformer.reference.segformer_for_image_classification import SegformerForImageClassificationReference
from models.demos.segformer.tt.ttnn_segformer_for_image_classification import TtSegformerForImageClassification
from tests.ttnn.integration_tests.segformer.test_segformer_for_image_classification import create_custom_preprocessor
from tests.ttnn.integration_tests.segformer.test_segformer_model import move_to_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("batch_size", [1])
def test_segformer_classification_demo(device, imagenet_label_dict, iterations, batch_size, model_location_generator):
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0").to(torch.bfloat16)
    reference_model = SegformerForImageClassificationReference(config=torch_model.config)
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    correct, torch_correct, ttnn_correct = 0, 0, 0
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        inputs = image_processor(inputs, return_tensors="pt")
        torch_input_tensor = inputs.pixel_values
        torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        reference_model.load_state_dict(torch_model.state_dict())
        reference_model.eval()
        torch_output = reference_model(torch_input_tensor)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=None,
        )
        parameters = move_to_device(parameters, device)
        ttnn_model = TtSegformerForImageClassification(torch_model.config, parameters)
        ttnn_output = ttnn_model(
            ttnn_input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=parameters,
            model=reference_model,
        )
        torch_final_output = torch_output.logits
        torch_predicted_id = torch_final_output.argmax(-1).item()
        labels_org = reference_model.config.id2label[labels[0]]
        torch_predicted_label = reference_model.config.id2label[torch_predicted_id]
        ttnn_final_output = ttnn.to_torch(ttnn_output.logits)
        ttnn_predicted_id = ttnn_final_output.argmax(-1).item()
        ttnn_predicted_label = reference_model.config.id2label[ttnn_predicted_id]

        if ttnn_predicted_label == labels_org:
            ttnn_correct += 1
        if torch_predicted_label == labels_org:
            torch_correct += 1
        if torch_predicted_label == ttnn_predicted_label:
            correct += 1
    ttnn_accuracy = ttnn_correct / iterations
    logger.info(f"ttnn_accuracy: {ttnn_accuracy:.2f}%")

    torch_accuracy = torch_correct / iterations
    logger.info(f"torch_accuracy: {torch_accuracy:.2f}%")

    accuracy = correct / iterations
    logger.info(f"accuracy: {accuracy:.2f}%")

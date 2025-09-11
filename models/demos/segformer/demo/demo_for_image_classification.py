# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from transformers import AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_for_image_classification import SegformerForImageClassificationReference
from models.demos.segformer.tests.pcc.test_segformer_for_image_classification import create_custom_mesh_preprocessor
from models.demos.segformer.tests.pcc.test_segformer_model import move_to_device
from models.demos.segformer.tt.ttnn_segformer_for_image_classification import TtSegformerForImageClassification
from models.demos.utils.common_demo_utils import get_batch, get_data_loader, get_mesh_mappers, load_imagenet_dataset


def run_segformer_classification_demo(
    device, imagenet_label_dict, iterations, device_batch_size, model_location_generator, resolution=512
):
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    logger.info("ImageNet-1k validation Dataset")
    input_loc = load_imagenet_dataset(model_location_generator)
    correct = 0
    torch_correct = 0
    ttnn_correct = 0
    batch_size = device_batch_size * device.get_num_devices()
    iterations = iterations // batch_size
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    inputs_mapper, wts_mapper, output_composer = get_mesh_mappers(device)

    for iter in range(iterations):
        torch_input_tensor, labels = get_batch(data_loader, resolution)
        torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mapper,
        )
        config = load_config("configs/segformer_img_classification_config.json")
        reference_model = SegformerForImageClassificationReference(config)
        reference_model = load_torch_model(
            reference_model, f"", module="image_classification", model_location_generator=model_location_generator
        )
        torch_output = reference_model(torch_input_tensor)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_mesh_preprocessor(wts_mapper),
            device=None,
        )
        parameters = move_to_device(parameters, device)
        ttnn_model = TtSegformerForImageClassification(config, parameters)
        ttnn_output = ttnn_model(
            ttnn_input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=parameters,
            model=reference_model,
        )
        torch_final_output = torch_output.logits
        ttnn_final_output = ttnn.to_torch(ttnn_output.logits, mesh_composer=output_composer)
        torch_predicted_ids = torch_final_output.argmax(dim=-1)
        ttnn_predicted_ids = ttnn_final_output.argmax(dim=-1)
        for i in range(len(labels)):
            label_id = labels[i]
            label_str = reference_model.config.id2label[label_id]
            torch_predicted_label = reference_model.config.id2label[torch_predicted_ids[i].item()]
            ttnn_predicted_label = reference_model.config.id2label[ttnn_predicted_ids[i].item()]
            if torch_predicted_label == label_str:
                torch_correct += 1
            if ttnn_predicted_label == label_str:
                ttnn_correct += 1

    total_samples = iterations * batch_size
    ttnn_accuracy = ttnn_correct / total_samples * 100
    torch_accuracy = torch_correct / total_samples * 100
    logger.info(f"TTNN Top-1 Accuracy: {ttnn_accuracy:.2f}%")
    logger.info(f"Torch Top-1 Accuracy: {torch_accuracy:.2f}%")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("batch_size", [1])
def test_segformer_classification_demo(device, imagenet_label_dict, iterations, batch_size, model_location_generator):
    return run_segformer_classification_demo(
        device, imagenet_label_dict, iterations, batch_size, model_location_generator
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("iterations", [100])
@pytest.mark.parametrize("device_batch_size", [1])
def test_segformer_classification_demo_dp(
    mesh_device, imagenet_label_dict, iterations, device_batch_size, model_location_generator
):
    return run_segformer_classification_demo(
        mesh_device, imagenet_label_dict, iterations, device_batch_size, model_location_generator
    )

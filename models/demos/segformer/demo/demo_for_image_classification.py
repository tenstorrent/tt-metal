# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from datasets import load_dataset
from transformers import SegformerForImageClassification, AutoImageProcessor
from tests.ttnn.integration_tests.segformer.test_segformer_image_classification import create_custom_preprocessor
from tests.ttnn.integration_tests.segformer.test_segformer_model import move_to_device
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.segformer.reference.segformer_image_classification import (
    SegformerForImageClassificationReference,
)
from models.demos.segformer.tt.ttnn_segformer_image_classification import (
    TtSegformerForImageClassification,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger
import os
import cv2
import json


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_demo_image_classification(device, reset_seeds):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    torch_model = SegformerForImageClassification.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config
    state_dict = torch_model.state_dict()

    reference_model = SegformerForImageClassificationReference(config=config)

    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    if not os.path.exists("models/demos/segformer/demo/validation.zip"):
        print("yessss")
        os.system("bash models/demos/segformer/demo/validation_data_download.sh")

    data_path = "models/demos/segformer/demo/validation_data/"

    images = []
    gt = []
    image_dict = {}
    for root, dirs, files in os.walk(data_path):
        image_files = set()
        json_files = set()

        for file in files:
            if file.endswith(".jpg"):
                base_name = os.path.splitext(file)[0]
                image_files.add(base_name)
            elif file.endswith(".json"):
                base_name = os.path.splitext(file)[0]
                json_files.add(base_name)
        common_bases = image_files & json_files
        for base_name in common_bases:
            image_path = os.path.join(root, base_name + ".jpg")
            json_path = os.path.join(root, base_name + ".json")
            images.append(image_path)
            gt.append(json_path)
            # print("image: ", image_path)
            # print("json: ", json_path)

    print(len(images), len(gt))
    samples = 100
    correct = 0
    torch_correct = 0
    torch_org_correct = 0
    ttnn_correct = 0
    for image_file, gt_file in zip(images[:samples], gt[:samples]):
        print(f"Image: {image_file}")
        print(f"Ground Truth: {gt_file}")

        image = cv2.imread(image_file)
        with open(gt_file, "r") as f:
            data = json.load(f)
        objects = data.get("annotation", {}).get("object", [])
        ground_truth_labels = [obj.get("name") for obj in objects]
        print("Ground Truth Labels:", ground_truth_labels)

        inputs = image_processor(image, return_tensors="pt")
        torch_input_tensor = inputs.pixel_values
        torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        new_state_dict = {}
        keys = [name for name, parameter in reference_model.state_dict().items()]
        values = [parameter for name, parameter in state_dict.items()]
        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]
        reference_model.load_state_dict(new_state_dict)
        reference_model.eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=None,
        )
        parameters = move_to_device(parameters, device)

        torch_output = reference_model(torch_input_tensor)
        org_torch = torch_model(torch_input_tensor)
        ttnn_model = TtSegformerForImageClassification(config, parameters)
        ttnn_output = ttnn_model(
            ttnn_input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=parameters,
            model=reference_model,
        )

        torch_final_output_org = org_torch.logits
        torch_predicted_id_org = torch_final_output_org.argmax(-1).item()

        torch_predicted_label_org = torch_model.config.id2label[torch_predicted_id_org]
        print("torch_predicted_label_org: ", torch_predicted_label_org, torch_predicted_id_org)

        torch_final_output = torch_output.logits
        torch_predicted_id = torch_final_output.argmax(-1).item()

        torch_predicted_label = reference_model.config.id2label[torch_predicted_id]
        print("torch_predicted_label: ", torch_predicted_label, torch_predicted_id)

        ttnn_final_output = ttnn.to_torch(ttnn_output.logits)
        ttnn_predicted_id = ttnn_final_output.argmax(-1).item()

        ttnn_predicted_label = reference_model.config.id2label[ttnn_predicted_id]
        print("ttnn_predicted_label: ", ttnn_predicted_label, ttnn_predicted_id)

        if torch_predicted_label_org in ground_truth_labels:
            torch_org_correct += 1

        if ttnn_predicted_label in ground_truth_labels:
            ttnn_correct += 1
        else:
            print("TTNN Prediction is incorrect.")

        if torch_predicted_label in ground_truth_labels:
            torch_correct += 1
        else:
            print("Torch Prediction is incorrect.")

        if torch_predicted_label == ttnn_predicted_label:
            correct += 1
        else:
            print("Torch and TTNN Predictions are different.")

        # logger.info("Output")
        # logger.info(reference_model.config.id2label[ttnn_predicted_label])
    ttnn_accuracy = ttnn_correct / samples
    print("org, torch and ttnn corrects", torch_org_correct, torch_correct, ttnn_correct)
    print(f"ttnn_accuracy: {ttnn_accuracy:.2f}%")

    torch_accuracy = torch_correct / samples
    print(f"torch_accuracy: {torch_accuracy:.2f}%")

    accuracy = correct / samples
    print(f"accuracy: {accuracy:.2f}%")
    org_accuracy = torch_org_correct / samples
    print(f"org_accuracy: {org_accuracy:.2f}%")

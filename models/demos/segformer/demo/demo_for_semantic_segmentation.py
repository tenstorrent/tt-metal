# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os

import evaluate
import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.demos.segformer.tt.ttnn_segformer_for_semantic_segmentation import TtSegformerForSemanticSegmentation
from tests.ttnn.integration_tests.segformer.test_segformer_for_semantic_segmentation import create_custom_preprocessor
from tests.ttnn.integration_tests.segformer.test_segformer_model import move_to_device


class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, image_processor):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        return {"input": inputs, "gt_mask": mask_np}


def shift_gt_indices(gt_mask):
    return gt_mask - 1


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_demo_semantic_segmentation(device):
    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    reference_model = SegformerForSemanticSegmentationReference(config=torch_model.config)
    reference_model.load_state_dict(torch_model.state_dict())
    reference_model.eval()
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    if not os.path.exists("models/demos/segformer/demo/validation_data_ade20k"):
        logger.info("downloading data")
        os.system("bash models/demos/segformer/demo/data_download.sh")

    image_folder = "models/demos/segformer/demo/validation_data_ade20k/images"
    mask_folder = "models/demos/segformer/demo/validation_data_ade20k/annotations"

    dataset = SemanticSegmentationDataset(image_folder, mask_folder, image_processor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    ref_metric = evaluate.load("mean_iou")
    ttnn_metric = evaluate.load("mean_iou")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    ttnn_model = TtSegformerForSemanticSegmentation(torch_model.config, parameters)
    parameters = move_to_device(parameters, device)
    for i in range(4):
        parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device=device
        )
    for batch in data_loader:
        image = batch["input"]
        mask = batch["gt_mask"].squeeze()
        torch_input_tensor_permuted = torch.permute(image["pixel_values"].squeeze(dim=0), (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn_output = ttnn_model(
            device,
            ttnn_input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=parameters,
        )
        ttnn_output = ttnn.to_torch(ttnn_output.logits)
        ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
        h = w = int(math.sqrt(ttnn_output.shape[-1]))
        ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))
        ref_logits = reference_model(image["pixel_values"].squeeze(dim=0)).logits
        ref_upsampled_logits = torch.nn.functional.interpolate(
            ref_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        ttnn_upsampled_logits2 = torch.nn.functional.interpolate(
            ttnn_final_output, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        ref_pred_mask = ref_upsampled_logits.argmax(dim=1).squeeze().numpy()
        ttnn_pred_mask = ttnn_upsampled_logits2.argmax(dim=1).squeeze().numpy()
        mask = shift_gt_indices(mask)
        mask = np.array(mask)
        ref_metric.add(predictions=ref_pred_mask, references=mask)
        ttnn_metric.add(predictions=ttnn_pred_mask, references=mask)

    ref_results = ref_metric.compute(
        num_labels=reference_model.config.num_labels, ignore_index=255, reduce_labels=False
    )
    ttnn_results = ttnn_metric.compute(
        num_labels=reference_model.config.num_labels, ignore_index=255, reduce_labels=False
    )

    logger.info(
        f"mean IoU values for Reference and ttnn model are {ref_results['mean_iou']}, {ttnn_results['mean_iou']} respectively"
    )

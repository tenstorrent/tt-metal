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
from transformers import AutoImageProcessor

import ttnn
from models.demos.segformer.common import download_and_unzip_dataset, load_config, load_torch_model
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.demos.segformer.runner.performant_runner import SegformerTrace2CQ


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


def custom_collate_fn(batch):
    inputs = [item["input"] for item in batch]
    gt_masks = [item["gt_mask"] for item in batch]
    input_pixel_values = torch.cat([inp["pixel_values"] for inp in inputs], dim=0)
    return {"pixel_values": input_pixel_values, "gt_mask": gt_masks}


def run_demo_semantic_segmentation(device, model_location_generator, device_batch_size):
    batch_size = device_batch_size * device.get_num_devices()
    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerForSemanticSegmentationReference(config)
    target_prefix = f""
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )
    reference_model.eval()
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    dataset_path = "segformer-segmentation"
    dataset_name = "validation_data_ade20k"
    weights_path = download_and_unzip_dataset(model_location_generator, dataset_path, dataset_name)

    image_folder = f"{weights_path}/images"
    mask_folder = f"{weights_path}/annotations"

    dataset = SemanticSegmentationDataset(image_folder, mask_folder, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    ref_metric = evaluate.load("mean_iou")
    ttnn_metric = evaluate.load("mean_iou")
    segformer_trace_2cq = SegformerTrace2CQ()
    segformer_trace_2cq.initialize_segformer_trace_2cqs_inference(
        device,
        model_location_generator=model_location_generator,
        device_batch_size=device_batch_size,
    )

    for batch in data_loader:
        masks = batch["gt_mask"]
        input = batch["pixel_values"]
        ttnn_output = segformer_trace_2cq.run(input)
        ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=segformer_trace_2cq.test_infra.output_mesh_composer)
        ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
        h = w = int(math.sqrt(ttnn_output.shape[-1]))
        ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))
        ref_logits = reference_model(input).logits
        for i in range(len(masks)):
            mask = masks[i]
            ref_up = torch.nn.functional.interpolate(
                ref_logits[i : i + 1], size=mask.shape, mode="bilinear", align_corners=False
            )
            ttnn_up = torch.nn.functional.interpolate(
                ttnn_final_output[i : i + 1], size=mask.shape, mode="bilinear", align_corners=False
            )
            ref_pred_mask = ref_up.argmax(dim=1).squeeze().cpu().numpy()
            ttnn_pred_mask = ttnn_up.argmax(dim=1).squeeze().cpu().numpy()
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


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_demo_semantic_segmentation(device, model_location_generator, batch_size):
    return run_demo_semantic_segmentation(device, model_location_generator, batch_size)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((1),),
)
def test_demo_semantic_segmentation_dp(mesh_device, model_location_generator, device_batch_size):
    return run_demo_semantic_segmentation(mesh_device, model_location_generator, device_batch_size)

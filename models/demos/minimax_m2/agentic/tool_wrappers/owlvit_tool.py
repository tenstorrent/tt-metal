# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
OWLViTTool: Wraps OWL-ViT TTNN end-to-end detection pipeline.

Zero-shot open-vocabulary object detection using text query strings.
No Metal trace — single-pass encoder.  First call JIT-compiles TTNN ops;
subsequent calls are fast.
"""

from typing import List

import torch
from loguru import logger
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

import ttnn
from models.demos.wormhole.owl_vit.tests.test_end_to_end import preprocess_all_weights_for_ttnn, run_owl_vit_end_to_end

MODEL_NAME = "google/owlvit-base-patch32"
DETECTION_THRESHOLD = 0.1


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class OWLViTTool:
    """
    TTNN-accelerated OWL-ViT zero-shot object detection wrapper.

    Accepts an image file path and text query; returns detections as a list
    of dicts with keys: label, score, bbox ([x1, y1, x2, y2] normalised 0–1).
    """

    def __init__(self, mesh_device):
        self.device = mesh_device

        logger.info("Loading OWL-ViT model and preprocessing weights...")
        self.processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        self.hf_model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
        self.hf_model.eval()
        self.parameters = preprocess_all_weights_for_ttnn(self.hf_model, self.device)
        logger.info("OWL-ViT ready.")

    def detect(self, image_path: str, query: str, threshold: float = DETECTION_THRESHOLD) -> List[dict]:
        """
        Detect objects matching *query* in the image at *image_path*.

        Args:
            image_path: Path to image file.
            query: Free-text description of what to detect (e.g. "a cat").
            threshold: Confidence score threshold (0–1).

        Returns:
            List of detection dicts: [{label, score, bbox}, ...]
            bbox is [x1, y1, x2, y2] in normalised image coordinates (0–1).
        """
        text_queries = [q.strip() for q in query.split(",")]
        image = _load_image(image_path)

        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        pred_boxes_tt, logits_tt = run_owl_vit_end_to_end(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            parameters=self.parameters,
            device=self.device,
            pytorch_model=self.hf_model,
        )

        # Handle multi-device mesh: use mesh_composer to gather shards
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if num_devices > 1 else None

        pred_boxes = ttnn.to_torch(pred_boxes_tt, mesh_composer=mesh_composer)
        logits = ttnn.to_torch(logits_tt, mesh_composer=mesh_composer)

        # For multi-device, take first replica (data is replicated, not sharded)
        if num_devices > 1:
            pred_boxes = pred_boxes[: pred_boxes.shape[0] // num_devices]
            logits = logits[: logits.shape[0] // num_devices]

        pred_boxes = pred_boxes.squeeze(0)  # [num_patches, 4]
        logits = logits.squeeze(0)  # [num_patches, num_queries]
        scores = torch.sigmoid(logits)

        detections = []
        for patch_idx in range(scores.shape[0]):
            for query_idx, label in enumerate(text_queries):
                score = scores[patch_idx, query_idx].item()
                if score >= threshold:
                    box = pred_boxes[patch_idx].tolist()
                    detections.append(
                        {
                            "label": label,
                            "score": round(score, 4),
                            "bbox": [round(v, 4) for v in box],
                        }
                    )

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

try:
    from transformers import AutoImageProcessor
except ModuleNotFoundError:  # pragma: no cover
    AutoImageProcessor = None

from models.experimental.maskformer_swin.tt.fallback import MaskFormerFallbackPipeline
from models.experimental.maskformer_swin.tt.weights import (
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
)


@pytest.mark.skipif(AutoImageProcessor is None, reason="transformers package required.")
def test_maskformer_fallback_end_to_end(tmp_path):
    """Validate the HuggingFace fallback pipeline produces reasonable outputs."""

    weight_cfg = WeightConversionConfig()
    reference = download_reference_weights(weight_cfg)
    tt_state = convert_state_dict_to_tt(reference.state_dict, weight_cfg)
    pipeline = MaskFormerFallbackPipeline.from_reference(reference, tt_state)

    processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
    image_path = Path("models/sample_data/demo.jpeg")
    image = Image.open(image_path).convert("RGB")
    pixel_values: torch.Tensor = processor(images=image, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        outputs = pipeline.forward(pixel_values)

    num_queries = reference.config.get("num_queries", 100)
    num_classes = len(reference.config.get("id2label", {}))

    h, w = pixel_values.shape[-2:]
    assert outputs.class_logits.shape == (1, num_queries, num_classes + 1)
    assert outputs.mask_logits.shape == (1, num_queries, h // 4, w // 4)

    segmentation = pipeline.post_process_semantic(outputs, image_processor=processor, target_sizes=[image.size[::-1]])[
        0
    ]
    assert segmentation.shape == image.size[::-1]
    assert segmentation.dtype in (np.int64, torch.int64)
    assert segmentation.max() <= num_classes

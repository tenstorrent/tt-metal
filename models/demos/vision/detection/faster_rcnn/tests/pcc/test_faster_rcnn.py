# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC validation tests for Faster-RCNN TTNN implementation.
Validates backbone, FPN, and full model outputs against PyTorch reference.
"""

import math
from collections import OrderedDict

import pytest
import torch

import ttnn
from models.demos.vision.detection.faster_rcnn.common import (
    FASTER_RCNN_BATCH_SIZE,
    FASTER_RCNN_INPUT_HEIGHT,
    FASTER_RCNN_INPUT_WIDTH,
    FASTER_RCNN_L1_SMALL_SIZE,
    load_torch_faster_rcnn,
)
from models.demos.vision.detection.faster_rcnn.tt.model_preprocessing import (
    create_faster_rcnn_model_parameters,
)
from models.demos.vision.detection.faster_rcnn.tt.ttnn_faster_rcnn import (
    TtFasterRCNN,
    TtFasterRCNNBackboneOnly,
    ttnn_to_torch_feature,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_input_tensor(batch_size, input_height, input_width):
    """Create a random input tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 3, input_height, input_width)


def prepare_ttnn_input(torch_input, device, transform):
    """Prepare TTNN input from a PyTorch tensor using the model's transform."""
    with torch.no_grad():
        image_list, _ = transform(torch_input)
        transformed = image_list.tensors

    nhwc = transformed.permute(0, 2, 3, 1).contiguous()
    nhwc = torch.nn.functional.pad(nhwc, (0, 16 - nhwc.shape[-1]), value=0)
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn.reshape(
        ttnn_input,
        (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
    )
    return ttnn_input, transformed, image_list.image_sizes


@pytest.mark.parametrize("device_params", [{"l1_small_size": FASTER_RCNN_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [FASTER_RCNN_BATCH_SIZE])
@pytest.mark.parametrize(
    "input_height, input_width",
    [(FASTER_RCNN_INPUT_HEIGHT, FASTER_RCNN_INPUT_WIDTH)],
)
def test_faster_rcnn_backbone_pcc(device, batch_size, input_height, input_width, reset_seeds):
    """Validate ResNet-50 backbone + FPN output against PyTorch reference."""
    torch_model = load_torch_faster_rcnn(pretrained=True)
    torch_input = create_input_tensor(batch_size, input_height, input_width)

    with torch.no_grad():
        image_list, _ = torch_model.transform(torch_input)
        transformed = image_list.tensors
        torch_features = torch_model.backbone(transformed)

    parameters = create_faster_rcnn_model_parameters(torch_model, device=device)
    ttnn_backbone = TtFasterRCNNBackboneOnly(parameters, device, batch_size)

    nhwc = transformed.permute(0, 2, 3, 1).contiguous()
    nhwc = torch.nn.functional.pad(nhwc, (0, 16 - nhwc.shape[-1]), value=0)
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn.reshape(
        ttnn_input,
        (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
    )

    fpn_features = ttnn_backbone(ttnn_input)

    for key in ["0", "1", "2", "3"]:
        ttnn_feat = fpn_features[key]
        torch_feat = torch_features[key]

        ttnn_torch = ttnn_to_torch_feature(ttnn_feat, batch_size, 256)
        assert_with_pcc(torch_feat, ttnn_torch, pcc=0.90)


@pytest.mark.parametrize("device_params", [{"l1_small_size": FASTER_RCNN_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [FASTER_RCNN_BATCH_SIZE])
@pytest.mark.parametrize(
    "input_height, input_width",
    [(FASTER_RCNN_INPUT_HEIGHT, FASTER_RCNN_INPUT_WIDTH)],
)
def test_faster_rcnn_full_model(device, batch_size, input_height, input_width, reset_seeds):
    """Validate full Faster-RCNN detections produce valid output."""
    torch_model = load_torch_faster_rcnn(pretrained=True)
    torch_input = create_input_tensor(batch_size, input_height, input_width)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    parameters = create_faster_rcnn_model_parameters(torch_model, device=device)
    ttnn_model = TtFasterRCNN(
        parameters,
        device,
        torch_model,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    ttnn_output = ttnn_model(torch_input)

    for i in range(batch_size):
        assert "boxes" in ttnn_output[i], "Output must contain 'boxes'"
        assert "labels" in ttnn_output[i], "Output must contain 'labels'"
        assert "scores" in ttnn_output[i], "Output must contain 'scores'"

        if len(ttnn_output[i]["boxes"]) > 0:
            assert ttnn_output[i]["boxes"].shape[1] == 4, "Boxes must have 4 coordinates"
            assert len(ttnn_output[i]["labels"]) == len(
                ttnn_output[i]["scores"]
            ), "Labels and scores must have same length"
            assert (ttnn_output[i]["scores"] >= 0).all(), "Scores must be non-negative"
            assert (ttnn_output[i]["scores"] <= 1).all(), "Scores must be <= 1"

    if len(torch_output[0]["boxes"]) > 0 and len(ttnn_output[0]["boxes"]) > 0:
        torch_top_label = torch_output[0]["labels"][0].item()
        ttnn_top_label = ttnn_output[0]["labels"][0].item()

        torch_top_score = torch_output[0]["scores"][0].item()
        ttnn_top_score = ttnn_output[0]["scores"][0].item()

        score_diff = abs(torch_top_score - ttnn_top_score)
        assert score_diff < 0.3, f"Top detection score difference too large: {score_diff}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": FASTER_RCNN_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [FASTER_RCNN_BATCH_SIZE])
def test_faster_rcnn_output_format(device, batch_size, reset_seeds):
    """Verify that the TTNN model output format matches expected structure."""
    torch_model = load_torch_faster_rcnn(pretrained=True)
    torch_input = create_input_tensor(batch_size, FASTER_RCNN_INPUT_HEIGHT, FASTER_RCNN_INPUT_WIDTH)

    parameters = create_faster_rcnn_model_parameters(torch_model, device=device)
    ttnn_model = TtFasterRCNN(
        parameters,
        device,
        torch_model,
        batch_size=batch_size,
        input_height=FASTER_RCNN_INPUT_HEIGHT,
        input_width=FASTER_RCNN_INPUT_WIDTH,
    )

    output = ttnn_model(torch_input)

    assert isinstance(output, list), "Output must be a list"
    assert len(output) == batch_size, f"Output length must match batch size ({batch_size})"

    for det in output:
        assert isinstance(det, dict), "Each detection must be a dict"
        assert set(det.keys()) == {"boxes", "labels", "scores"}, "Detection dict must have keys: boxes, labels, scores"
        assert det["boxes"].dtype == torch.float32, "Boxes must be float32"
        assert det["labels"].dtype == torch.int64, "Labels must be int64"
        assert det["scores"].dtype == torch.float32, "Scores must be float32"

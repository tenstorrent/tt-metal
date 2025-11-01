"""Tests for TTNN detection layer implementation."""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import torch_to_tt_tensor_rm
from models.common.utility_functions import comp_pcc
from models.experimental.SSD512.tt.layers.detect import TtDetect
from models.experimental.SSD512.reference.layers.functions.detection import Detect as TorchDetect


@pytest.mark.parametrize("pcc", [0.97])
def test_detect_layer(device, pcc):
    """Test TTNN detection layer against PyTorch implementation."""

    # Test parameters
    num_classes = 21
    top_k = 200
    conf_thresh = 0.01
    nms_thresh = 0.45
    batch_size = 1
    num_priors = 8732  # Total number of prior boxes in SSD300

    # Create models
    torch_detect = TorchDetect(num_classes, 0, top_k, conf_thresh, nms_thresh)
    tt_detect = TtDetect(num_classes, top_k, conf_thresh, nms_thresh, device)

    # Create test inputs
    # Location predictions [batch, num_priors, 4]
    loc_data = torch.randn(batch_size, num_priors, 4)

    # Class confidence scores [batch, num_priors, num_classes]
    conf_data = torch.randn(batch_size, num_priors, num_classes)
    conf_data = torch.softmax(conf_data, dim=-1)

    # Prior boxes [num_priors, 4]
    prior_data = torch.rand(num_priors, 4)

    # Convert to TTNN tensors
    tt_loc = torch_to_tt_tensor_rm(loc_data, device)
    tt_conf = torch_to_tt_tensor_rm(conf_data, device)
    tt_prior = torch_to_tt_tensor_rm(prior_data, device)

    # Run forward passes
    torch_output = torch_detect.forward(loc_data, conf_data, prior_data)

    tt_output = tt_detect(tt_loc, tt_conf, tt_prior)

    # Compare outputs
    # Note: Order of detections may differ slightly due to floating point differences
    # affecting NMS, so we'll compare sorted scores and corresponding boxes/labels
    for i in range(batch_size):
        torch_det = torch_output[i]
        tt_det = tt_output[i]

        # Sort by scores
        torch_scores, torch_idx = torch_det["scores"].sort(descending=True)
        tt_scores, tt_idx = tt_det["scores"].sort(descending=True)

        # Reorder boxes and labels
        torch_boxes = torch_det["boxes"][torch_idx]
        torch_labels = torch_det["labels"][torch_idx]
        tt_boxes = tt_det["boxes"][tt_idx]
        tt_labels = tt_det["labels"][tt_idx]

        # Compare scores
        scores_pass, scores_pcc = comp_pcc(torch_scores, tt_scores, pcc)
        logger.info(f"Detection scores PCC: {scores_pcc}")

        # Compare boxes
        boxes_pass, boxes_pcc = comp_pcc(torch_boxes, tt_boxes, pcc)
        logger.info(f"Detection boxes PCC: {boxes_pcc}")

        # Compare labels (these should match exactly)
        labels_match = torch.all(torch_labels == tt_labels)
        logger.info(f"Detection labels match: {labels_match}")

        assert scores_pass, f"Detection scores do not meet PCC requirement {pcc}"
        assert boxes_pass, f"Detection boxes do not meet PCC requirement {pcc}"
        assert labels_match, "Detection labels do not match"

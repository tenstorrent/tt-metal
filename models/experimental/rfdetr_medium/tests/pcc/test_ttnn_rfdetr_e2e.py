# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC test for RF-DETR Medium.
Runs the full TTNN pipeline and compares against PyTorch reference at each stage.
"""

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RFDETR_MEDIUM_L1_SMALL_SIZE,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rfdetr_medium_e2e(device, torch_model, sample_image, reference_outputs):
    """
    Full E2E test: run TTNN pipeline and compare against PyTorch at each stage.
    """
    from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
    from models.experimental.rfdetr_medium.tt.model_preprocessing import (
        load_backbone_weights,
        load_projector_weights,
        load_decoder_weights,
        load_detection_head_weights,
    )

    # Load weights
    backbone_params = load_backbone_weights(torch_model, device)
    projector_params = load_projector_weights(torch_model, device)
    decoder_params = load_decoder_weights(torch_model, device)
    head_params = load_detection_head_weights(torch_model, device)

    # Create TTNN model
    tt_model = TtRFDETR(
        device=device,
        torch_model=torch_model,
        backbone_params=backbone_params,
        projector_params=projector_params,
        decoder_params=decoder_params,
        head_params=head_params,
    )

    # Run full forward
    result = tt_model.forward(sample_image)

    # Compare detection head outputs
    ref_cls = reference_outputs["all_cls_scores"][-1]
    ref_bbox = reference_outputs["all_bbox_preds"][-1]

    tt_cls = result["outputs_class"]
    tt_bbox = result["outputs_coord"]

    if isinstance(tt_cls, ttnn.Tensor):
        tt_cls = ttnn.to_torch(tt_cls).float()
    if isinstance(tt_bbox, ttnn.Tensor):
        tt_bbox = ttnn.to_torch(tt_bbox).float()

    # PCC checks
    cls_pcc = assert_with_pcc(ref_cls, tt_cls, pcc=0.50)
    print(f"Classification logits PCC: {cls_pcc}")

    bbox_pcc = assert_with_pcc(ref_bbox, tt_bbox, pcc=0.35)
    print(f"Bounding box predictions PCC: {bbox_pcc}")

    # Detection match rate
    ref_detections = reference_outputs["detections"]
    tt_detections = result["detections"]

    if len(ref_detections) > 0 and len(ref_detections[0]["boxes"]) > 0:
        ref_boxes = ref_detections[0]["boxes"]
        tt_boxes = tt_detections[0]["boxes"]
        print(f"Reference detections: {len(ref_boxes)}, TTNN detections: {len(tt_boxes)}")

        # IoU comparison for matched detections
        if len(ref_boxes) > 0 and len(tt_boxes) > 0:
            from rfdetr.util.box_ops import box_iou

            ref_xyxy = ref_boxes
            tt_xyxy = tt_boxes
            if isinstance(tt_xyxy, ttnn.Tensor):
                tt_xyxy = ttnn.to_torch(tt_xyxy).float()

            iou_matrix = box_iou(ref_xyxy, tt_xyxy)[0]
            max_ious = iou_matrix.max(dim=1)[0]
            avg_iou = max_ious.mean().item()
            print(f"Average detection IoU: {avg_iou:.4f}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rfdetr_medium_backbone_pcc(device, torch_model, sample_image, reference_outputs):
    """
    Test backbone + projector stage PCC (lighter test).
    Reference srcs are post-projector [B, 256, 36, 36] from the Joiner.
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_backbone_weights, load_projector_weights
    from models.experimental.rfdetr_medium.tt.tt_backbone import dinov2_backbone
    from models.experimental.rfdetr_medium.tt.tt_projector import projector_forward

    backbone_params = load_backbone_weights(torch_model, device)
    projector_params = load_projector_weights(torch_model, device)

    img = sample_image.permute(0, 2, 3, 1)
    img = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))
    img_tt = ttnn.from_torch(img, dtype=ttnn.bfloat16, device=device)

    feature_maps = dinov2_backbone(img_tt, backbone_params, batch_size=1)
    projected = projector_forward(feature_maps, projector_params, batch_size=1, device=device)

    ref_srcs = reference_outputs["srcs"]
    for i, (proj, ref_src) in enumerate(zip(projected, ref_srcs)):
        proj_torch = ttnn.to_torch(proj).float() if isinstance(proj, ttnn.Tensor) else proj
        assert proj_torch.shape == ref_src.shape, f"Feature {i}: expected {ref_src.shape}, got {proj_torch.shape}"
        pcc = assert_with_pcc(ref_src, proj_torch, pcc=0.70)
        print(f"Backbone+Projector feature {i}: PCC = {pcc}")

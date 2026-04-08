# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple
from models.experimental.ssd.tt.ssd import TtSSD
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)


def _ssd(
    config,
    size: Tuple[int, int],
    num_classes: int,
    image_mean: Optional[List[float]],
    image_std: Optional[List[float]],
    score_thresh: float,
    nms_thresh: float,
    detections_per_image: int,
    topk_candidates: int,
    state_dict=None,
    base_address="",
    device=None,
):
    return TtSSD(
        config=config,
        size=size,
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
        score_thresh=score_thresh,
        detections_per_image=detections_per_image,
        topk_candidates=topk_candidates,
        nms_thresh=nms_thresh,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )


def ssd_for_object_detection(device) -> TtSSD:
    model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights)
    model.eval()
    state_dict = model.state_dict()
    config = {}
    base_address = f""
    model = _ssd(
        config,
        size=(320, 320),
        num_classes=91,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        score_thresh=0.001,
        detections_per_image=300,
        topk_candidates=300,
        nms_thresh=0.55,
        state_dict=state_dict,
        base_address=f"",
        device=device,
    )
    return model

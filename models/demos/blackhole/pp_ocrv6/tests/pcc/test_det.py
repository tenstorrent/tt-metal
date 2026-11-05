# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import cv2
import pytest
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import (
    DET_L1_SMALL_SIZE,
    det_normalize_bgr,
    load_torch_det,
    synthetic_gate_image,
)
from models.demos.blackhole.pp_ocrv6.tt.ttnn_det import TtPPOCRv6SmallDet, unshuffle_probability
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("size", [640])
@pytest.mark.parametrize("source", ["gate_image", "random"])
def test_det(device, size, source):
    torch.manual_seed(0)
    torch_model = load_torch_det()
    if source == "random":
        x = torch.randn(1, 3, size, size)
    else:
        x = det_normalize_bgr(cv2.resize(synthetic_gate_image(), (size, size)))
    with torch.no_grad():
        ref = torch_model(x)

    tt_model = TtPPOCRv6SmallDet(device, torch_model)
    tt_x = ttnn.from_torch(x.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    out, h4, w4 = tt_model(tt_x, 1, size, size)
    prob = unshuffle_probability(ttnn.to_torch(out).float(), 1, h4, w4)

    passed, pcc = check_with_pcc(ref, prob, pcc=0.99)
    text_px = (ref > 0.2).float().mean().item()
    flips = ((prob > 0.2) != (ref > 0.2)).float().mean().item()
    print(f"[{source}] PCC {pcc}  text pixels {text_px:.3%}  mask disagreement at 0.2 {flips:.4%}")
    assert passed, pcc

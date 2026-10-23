# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the TtYOLO26 class (the model used by the demo, runner and perf tests).

Unlike test_pcc.py, activations stay on device between layers.

Usage:
    pytest models/experimental/yolo26/tests/pcc/test_tt_yolo26.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE
from models.common.utility_functions import comp_pcc


def to_nhwc(t, batch_size, h, w, ch):
    """Convert tensor to NHWC format."""
    if t.memory_config().is_sharded():
        t = ttnn.sharded_to_interleaved(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [batch_size, h, w, ch])
    return t


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["yolo26n", "yolo26s"])
def test_tt_yolo26_pcc(device, variant):
    """Compare TtYOLO26 one2one box/cls outputs against Ultralytics."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26

    torch_model = YOLO(f"{variant}.pt")
    tt_model = TtYOLO26(device, variant)
    tt_model.load_weights_from_state_dict(torch_model.model.state_dict())

    batch_size = 1
    input_size = 640

    torch.manual_seed(42)
    x_torch = torch.randn(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # ========== PyTorch reference ==========
    with torch.no_grad():
        pt_intermediates = {}
        x_pt = x_torch.float()
        for i in range(23):
            layer = torch_model.model.model[i]
            if layer.__class__.__name__ == "Concat":
                x_pt = torch.cat([pt_intermediates[idx] if idx != -1 else x_pt for idx in layer.f], dim=1)
            else:
                x_pt = layer(x_pt)
            pt_intermediates[i] = x_pt

        detect = torch_model.model.model[23]
        pt_feats = [pt_intermediates[16], pt_intermediates[19], pt_intermediates[22]]
        pt_box = [detect.one2one_cv2[i](pt_feats[i]) for i in range(3)]
        pt_cls = [detect.one2one_cv3[i](pt_feats[i]) for i in range(3)]

    # ========== TTNN forward ==========
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_out = tt_model(tt_x)

    # ========== Compare per scale ==========
    tt_box_list, tt_cls_list = [], []
    for i, name in enumerate(["N3", "N4", "N5"]):
        tt_box_t, hh, ww = tt_out["boxes"][i]
        tt_cls_t, _, _ = tt_out["scores"][i]
        tt_box = ttnn.to_torch(to_nhwc(tt_box_t, batch_size, hh, ww, 4)).float()
        tt_cls = ttnn.to_torch(to_nhwc(tt_cls_t, batch_size, hh, ww, 80)).float()
        _, box_pcc = comp_pcc(pt_box[i].permute(0, 2, 3, 1), tt_box)
        _, cls_pcc = comp_pcc(pt_cls[i].permute(0, 2, 3, 1), tt_cls)
        logger.info(f"{name} ({hh}x{ww}): box PCC={box_pcc:.6f}, cls PCC={cls_pcc:.6f}")
        tt_box_list.append(tt_box.flatten())
        tt_cls_list.append(tt_cls.flatten())

    pt_box_all = torch.cat([b.permute(0, 2, 3, 1).flatten() for b in pt_box])
    pt_cls_all = torch.cat([c.permute(0, 2, 3, 1).flatten() for c in pt_cls])

    pcc_threshold = 0.99
    box_pass, pcc_box = comp_pcc(pt_box_all, torch.cat(tt_box_list), pcc_threshold)
    cls_pass, pcc_cls = comp_pcc(pt_cls_all, torch.cat(tt_cls_list), pcc_threshold)
    logger.info(f"TtYOLO26 {variant} PCC ({input_size}x{input_size}): box={pcc_box:.6f}, cls={pcc_cls:.6f}")

    assert box_pass, f"box PCC {pcc_box:.4f} < {pcc_threshold}"
    assert cls_pass, f"cls PCC {pcc_cls:.4f} < {pcc_threshold}"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["yolo26n", "yolo26s"])
def test_tt_yolo26_uint8_input_pcc(device, variant):
    """uint8 input path (on-device sharding/typecast/view, 1/255 folded into the stem) on a real image."""
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26, uint8_input_from_torch

    torch_model = YOLO(f"{variant}.pt")
    tt_model = TtYOLO26(device, variant, uint8_input=True)
    tt_model.load_weights_from_state_dict(torch_model.model.state_dict())

    img = torch.from_numpy(np.asarray(Image.open(ASSETS / "bus.jpg").convert("RGB").resize((640, 640))).copy())

    with torch.no_grad():
        preds = torch_model.model.eval().float()(img.permute(2, 0, 1)[None].float() / 255)
    one2one = preds[1]["one2one"]
    pt_box = one2one["boxes"].flatten()
    pt_cls = one2one["scores"].flatten()

    tt_x = ttnn.from_torch(
        uint8_input_from_torch(img[None]), dtype=ttnn.uint8, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    tt_out = tt_model(tt_x)

    tt_box, tt_cls = [], []
    for i in range(3):
        bt, hh, ww = tt_out["boxes"][i]
        ct, _, _ = tt_out["scores"][i]
        tt_box.append(ttnn.to_torch(to_nhwc(bt, 1, hh, ww, 4)).float().reshape(1, hh * ww, 4).permute(0, 2, 1))
        tt_cls.append(ttnn.to_torch(to_nhwc(ct, 1, hh, ww, 80)).float().reshape(1, hh * ww, 80).permute(0, 2, 1))

    pcc_threshold = 0.99
    box_pass, pcc_box = comp_pcc(pt_box, torch.cat(tt_box, dim=-1).flatten(), pcc_threshold)
    cls_pass, pcc_cls = comp_pcc(pt_cls, torch.cat(tt_cls, dim=-1).flatten(), pcc_threshold)
    logger.info(f"TtYOLO26 {variant} uint8 input PCC (bus.jpg): box={pcc_box:.6f}, cls={pcc_cls:.6f}")

    assert box_pass, f"box PCC {pcc_box:.4f} < {pcc_threshold}"
    assert cls_pass, f"cls PCC {pcc_cls:.4f} < {pcc_threshold}"

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full YOLO26 model end-to-end PCC test.
Tests complete TTNN model against PyTorch reference.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


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
def test_full_model_e2e(device):
    """Test full YOLO26 model end-to-end with TTNN detection head."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import (
        TtConvBNSiLU,
        TtC2f,
        TtC3k2,
        TtSPPF,
        TtC2PSA,
        TtC3k2PSA,
        TtUpsample,
        TtYOLO26Head,
    )
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # ========== Create TTNN model ==========
    # Backbone
    backbone_layers = [
        TtConvBNSiLU(device, 3, 16, kernel_size=3, stride=2, padding=1, name="model.0"),
        TtConvBNSiLU(device, 16, 32, kernel_size=3, stride=2, padding=1, name="model.1"),
        TtC2f(device, 32, 64, hidden_channels=16, n=1, name="model.2"),
        TtConvBNSiLU(device, 64, 64, kernel_size=3, stride=2, padding=1, name="model.3"),
        TtC2f(device, 64, 128, hidden_channels=32, n=1, name="model.4"),
        TtConvBNSiLU(device, 128, 128, kernel_size=3, stride=2, padding=1, name="model.5"),
        TtC3k2(device, 128, 128, hidden_channels=64, n=1, name="model.6"),
        TtConvBNSiLU(device, 128, 256, kernel_size=3, stride=2, padding=1, name="model.7"),
        TtC3k2(device, 256, 256, hidden_channels=128, n=1, name="model.8"),
        TtSPPF(device, 256, 256, kernel_size=5, name="model.9"),
    ]

    for i, layer in enumerate(backbone_layers):
        if isinstance(layer, TtConvBNSiLU):
            w, b = weight_loader.get_conv_bn(f"model.{i}")
            layer.load_weights(w, b)
        else:
            layer.load_weights(weight_loader, f"model.{i}")

    # Neck
    c2psa_10 = TtC2PSA(device, 256, 256, n=1, name="model.10")
    c2psa_10.load_weights(weight_loader, "model.10")

    upsample = TtUpsample(scale_factor=2)

    c3k2_13 = TtC3k2(device, 384, 128, hidden_channels=64, n=1, name="model.13")
    c3k2_13.load_weights(weight_loader, "model.13")

    c3k2_16 = TtC3k2(device, 256, 64, hidden_channels=32, n=1, name="model.16")
    c3k2_16.load_weights(weight_loader, "model.16")

    conv_17 = TtConvBNSiLU(device, 64, 64, kernel_size=3, stride=2, padding=1, name="model.17")
    w, b = weight_loader.get_conv_bn("model.17")
    conv_17.load_weights(w, b)

    c3k2_19 = TtC3k2(device, 192, 128, hidden_channels=64, n=1, name="model.19")
    c3k2_19.load_weights(weight_loader, "model.19")

    conv_20 = TtConvBNSiLU(device, 128, 128, kernel_size=3, stride=2, padding=1, name="model.20")
    w, b = weight_loader.get_conv_bn("model.20")
    conv_20.load_weights(w, b)

    c3k2_22 = TtC3k2PSA(device, 384, 256, hidden_channels=128, n=1, name="model.22")
    c3k2_22.load_weights(weight_loader, "model.22")

    # Detection head
    detect_head = TtYOLO26Head(device, "yolo26n", num_classes=80)
    detect_head.load_weights(weight_loader)

    # ========== Input ==========
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # ========== PyTorch reference ==========
    logger.info("Running PyTorch reference...")
    with torch.no_grad():
        pt_out = torch_model.model(x_torch.float())

    # Get intermediate outputs for comparison
    pt_intermediates = {}
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(23):
            layer = torch_model.model.model[i]
            if layer.__class__.__name__ == "Concat":
                tensors = [pt_intermediates[idx] if idx != -1 else x_pt for idx in layer.f]
                x_pt = torch.cat(tensors, dim=1)
            else:
                x_pt = layer(x_pt)
            pt_intermediates[i] = x_pt.clone()

    pt_n3 = pt_intermediates[16]
    pt_n4 = pt_intermediates[19]
    pt_n5 = pt_intermediates[22]

    # ========== TTNN forward ==========
    logger.info("Running TTNN model...")
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Backbone
    tt_intermediates = {}
    h, w = input_size, input_size
    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    for i, layer in enumerate(backbone_layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)
        tt_x_conv = to_nhwc(tt_x, batch_size, h, w, out_channels[i])
        tt_intermediates[i] = (ttnn.to_torch(tt_x_conv), h, w, out_channels[i])
        tt_x = ttnn.from_torch(tt_intermediates[i][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Neck
    tt_x, h, w = c2psa_10(tt_x, batch_size, h, w)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[10] = (ttnn.to_torch(tt_x_conv), h, w, 256)

    tt_x = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 20, 20, 256)
    tt_x6 = ttnn.from_torch(tt_intermediates[6][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x6], dim=3)

    tt_x, h, w = c3k2_13(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_intermediates[13] = (ttnn.to_torch(tt_x_conv), h, w, 128)

    tt_x = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 40, 40, 128)
    tt_x4 = ttnn.from_torch(tt_intermediates[4][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x4], dim=3)

    tt_x, h, w = c3k2_16(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_n3 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[16] = (tt_n3, 80, 80, 64)

    tt_x = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_17(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x13 = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x13], dim=3)

    tt_x, h, w = c3k2_19(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_n4 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[19] = (tt_n4, 40, 40, 128)

    tt_x = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_20(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x10 = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x10], dim=3)

    tt_x, h, w = c3k2_22(tt_x, batch_size, 20, 20)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_n5 = ttnn.to_torch(tt_x_conv)

    # ========== Compare neck outputs ==========
    pt_n3_nhwc = pt_n3.permute(0, 2, 3, 1).contiguous()
    pt_n4_nhwc = pt_n4.permute(0, 2, 3, 1).contiguous()
    pt_n5_nhwc = pt_n5.permute(0, 2, 3, 1).contiguous()

    _, pcc_n3 = comp_pcc(pt_n3_nhwc, tt_n3.float(), 0.99)
    _, pcc_n4 = comp_pcc(pt_n4_nhwc, tt_n4.float(), 0.99)
    _, pcc_n5 = comp_pcc(pt_n5_nhwc, tt_n5.float(), 0.99)

    logger.info(f"\n=== Neck Output PCC ===")
    logger.info(f"N3 PCC: {pcc_n3:.6f}")
    logger.info(f"N4 PCC: {pcc_n4:.6f}")
    logger.info(f"N5 PCC: {pcc_n5:.6f}")
    avg_neck_pcc = (pcc_n3 + pcc_n4 + pcc_n5) / 3
    logger.info(f"Average Neck PCC: {avg_neck_pcc:.6f}")

    # ========== Run TTNN detection head ==========
    n3_tensor = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n4_tensor = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n5_tensor = ttnn.from_torch(tt_n5, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_detect_out = detect_head((n3_tensor, 80, 80), (n4_tensor, 40, 40), (n5_tensor, 20, 20), batch_size)

    # Get PyTorch detection head outputs
    detect = torch_model.model.model[23]
    detect.eval()
    with torch.no_grad():
        pt_detect_out = detect.forward_head(
            [pt_n3, pt_n4, pt_n5], box_head=detect.one2one_cv2, cls_head=detect.one2one_cv3
        )

    # Compare per-scale outputs
    logger.info(f"\n=== Detection Head Per-Scale PCC ===")

    scales = ["N3 (80x80)", "N4 (40x40)", "N5 (20x20)"]
    box_pccs = []
    cls_pccs = []

    for i, scale_name in enumerate(scales):
        # Get TTNN outputs
        tt_bbox, h, w = tt_detect_out["boxes"][i]
        tt_cls, _, _ = tt_detect_out["scores"][i]

        tt_bbox_nhwc = to_nhwc(tt_bbox, batch_size, h, w, 4)
        tt_cls_nhwc = to_nhwc(tt_cls, batch_size, h, w, 80)

        tt_bbox_torch = ttnn.to_torch(tt_bbox_nhwc).float()
        tt_cls_torch = ttnn.to_torch(tt_cls_nhwc).float()

        # Get PyTorch outputs (need to run per-scale)
        pt_feats = [pt_n3, pt_n4, pt_n5]
        with torch.no_grad():
            pt_bbox = detect.one2one_cv2[i](pt_feats[i])
            pt_cls = detect.one2one_cv3[i](pt_feats[i])

        pt_bbox_nhwc = pt_bbox.permute(0, 2, 3, 1).contiguous()
        pt_cls_nhwc = pt_cls.permute(0, 2, 3, 1).contiguous()

        _, bbox_pcc = comp_pcc(pt_bbox_nhwc, tt_bbox_torch, 0.90)
        _, cls_pcc = comp_pcc(pt_cls_nhwc, tt_cls_torch, 0.90)

        box_pccs.append(bbox_pcc)
        cls_pccs.append(cls_pcc)

        logger.info(f"{scale_name}: bbox PCC={bbox_pcc:.6f}, cls PCC={cls_pcc:.6f}")

    avg_box_pcc = sum(box_pccs) / len(box_pccs)
    avg_cls_pcc = sum(cls_pccs) / len(cls_pccs)

    logger.info(f"\n=== FINAL SUMMARY ===")
    logger.info(f"Backbone+Neck Average PCC: {avg_neck_pcc:.6f}")
    logger.info(f"Detection Box Average PCC: {avg_box_pcc:.6f}")
    logger.info(f"Detection Cls Average PCC: {avg_cls_pcc:.6f}")

    overall_pcc = (avg_neck_pcc + avg_box_pcc + avg_cls_pcc) / 3
    logger.info(f"Overall Average PCC: {overall_pcc:.6f}")

    # Verify neck PCC meets target
    assert pcc_n3 > 0.99, f"N3 PCC {pcc_n3:.4f} below 0.99"
    assert pcc_n4 > 0.99, f"N4 PCC {pcc_n4:.4f} below 0.99"
    assert pcc_n5 > 0.99, f"N5 PCC {pcc_n5:.4f} below 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

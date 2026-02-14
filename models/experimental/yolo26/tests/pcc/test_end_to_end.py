# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end YOLO26 test - backbone+neck in TTNN, detection head in PyTorch.
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
def test_backbone_neck_pcc(device):
    """Test backbone+neck PCC against full PyTorch model."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import (
        TtConvBNSiLU,
        TtC2f,
        TtC3k2,
        TtSPPF,
        TtC2PSA,
        TtC3k2PSA,
        TtUpsample,
    )
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # ========== Create TTNN layers ==========
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

    # Load backbone weights
    for i, layer in enumerate(backbone_layers):
        if isinstance(layer, TtConvBNSiLU):
            w, b = weight_loader.get_conv_bn(f"model.{i}")
            layer.load_weights(w, b)
        else:
            layer.load_weights(weight_loader, f"model.{i}")

    # Neck layers
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

    # ========== Input ==========
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # ========== PyTorch reference ==========
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

    # PyTorch neck outputs
    pt_n3 = pt_intermediates[16]
    pt_n4 = pt_intermediates[19]
    pt_n5 = pt_intermediates[22]
    logger.info(f"PyTorch N3: {pt_n3.shape}, mean={pt_n3.mean():.4f}")
    logger.info(f"PyTorch N4: {pt_n4.shape}, mean={pt_n4.mean():.4f}")
    logger.info(f"PyTorch N5: {pt_n5.shape}, mean={pt_n5.mean():.4f}")

    # ========== TTNN forward ==========
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run backbone
    tt_intermediates = {}
    h, w = input_size, input_size
    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    for i, layer in enumerate(backbone_layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)
        tt_x_conv = to_nhwc(tt_x, batch_size, h, w, out_channels[i])
        tt_intermediates[i] = (ttnn.to_torch(tt_x_conv), h, w, out_channels[i])
        tt_x = ttnn.from_torch(tt_intermediates[i][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # model.10: C2PSA
    tt_x, h, w = c2psa_10(tt_x, batch_size, h, w)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[10] = (ttnn.to_torch(tt_x_conv), h, w, 256)

    # model.11: Upsample
    tt_x = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 20, 20, 256)

    # model.12: Concat with model.6
    tt_x6 = ttnn.from_torch(tt_intermediates[6][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x6], dim=3)

    # model.13: C3k2
    tt_x, h, w = c3k2_13(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_intermediates[13] = (ttnn.to_torch(tt_x_conv), h, w, 128)

    # model.14: Upsample
    tt_x = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 40, 40, 128)

    # model.15: Concat with model.4
    tt_x4 = ttnn.from_torch(tt_intermediates[4][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x4], dim=3)

    # model.16: C3k2 -> N3
    tt_x, h, w = c3k2_16(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_n3 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[16] = (tt_n3, h, w, 64)

    # model.17: Downsample
    tt_x = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_17(tt_x, batch_size, 80, 80)

    # model.18: Concat with model.13
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x13 = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x13], dim=3)

    # model.19: C3k2 -> N4
    tt_x, h, w = c3k2_19(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_n4 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[19] = (tt_n4, h, w, 128)

    # model.20: Downsample
    tt_x = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_20(tt_x, batch_size, 40, 40)

    # model.21: Concat with model.10
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x10 = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x10], dim=3)

    # model.22: C3k2PSA -> N5
    tt_x, h, w = c3k2_22(tt_x, batch_size, 20, 20)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_n5 = ttnn.to_torch(tt_x_conv)

    # ========== Compare neck outputs ==========
    logger.info(f"\nTTNN N3: {tt_n3.shape}, mean={tt_n3.float().mean():.4f}")
    logger.info(f"TTNN N4: {tt_n4.shape}, mean={tt_n4.float().mean():.4f}")
    logger.info(f"TTNN N5: {tt_n5.shape}, mean={tt_n5.float().mean():.4f}")

    # PCC comparison
    pt_n3_nhwc = pt_n3.permute(0, 2, 3, 1).contiguous()
    pt_n4_nhwc = pt_n4.permute(0, 2, 3, 1).contiguous()
    pt_n5_nhwc = pt_n5.permute(0, 2, 3, 1).contiguous()

    passed_n3, pcc_n3 = comp_pcc(pt_n3_nhwc, tt_n3.float(), 0.99)
    passed_n4, pcc_n4 = comp_pcc(pt_n4_nhwc, tt_n4.float(), 0.99)
    passed_n5, pcc_n5 = comp_pcc(pt_n5_nhwc, tt_n5.float(), 0.99)

    logger.info(f"\n=== PCC Results (target: 0.99) ===")
    logger.info(f"N3 PCC: {pcc_n3:.6f} - {'PASS' if passed_n3 else 'FAIL'}")
    logger.info(f"N4 PCC: {pcc_n4:.6f} - {'PASS' if passed_n4 else 'FAIL'}")
    logger.info(f"N5 PCC: {pcc_n5:.6f} - {'PASS' if passed_n5 else 'FAIL'}")

    avg_pcc = (pcc_n3 + pcc_n4 + pcc_n5) / 3
    logger.info(f"\nAverage PCC: {avg_pcc:.6f}")

    # ========== Run detection head on TTNN outputs using PyTorch ==========
    # Convert TTNN neck outputs to NCHW for PyTorch detection head
    tt_n3_nchw = tt_n3.float().permute(0, 3, 1, 2).contiguous()
    tt_n4_nchw = tt_n4.float().permute(0, 3, 1, 2).contiguous()
    tt_n5_nchw = tt_n5.float().permute(0, 3, 1, 2).contiguous()

    # Run PyTorch detection head on TTNN outputs
    detect = torch_model.model.model[23]
    detect.eval()
    with torch.no_grad():
        # Forward head returns dict with boxes and scores
        tt_feats = [tt_n3_nchw, tt_n4_nchw, tt_n5_nchw]
        pt_feats = [pt_n3, pt_n4, pt_n5]

        # Use forward_head directly
        tt_head_out = detect.forward_head(tt_feats, box_head=detect.one2one_cv2, cls_head=detect.one2one_cv3)
        pt_head_out = detect.forward_head(pt_feats, box_head=detect.one2one_cv2, cls_head=detect.one2one_cv3)

    logger.info(f"\n=== Detection Output Comparison ===")
    if "boxes" in tt_head_out and "boxes" in pt_head_out:
        _, boxes_pcc = comp_pcc(pt_head_out["boxes"], tt_head_out["boxes"], 0.99)
        logger.info(f"Detection boxes PCC: {boxes_pcc:.6f}")
    if "scores" in tt_head_out and "scores" in pt_head_out:
        _, scores_pcc = comp_pcc(pt_head_out["scores"], tt_head_out["scores"], 0.99)
        logger.info(f"Detection scores PCC: {scores_pcc:.6f}")

    assert passed_n3 and passed_n4 and passed_n5, f"PCC check failed: N3={pcc_n3:.4f}, N4={pcc_n4:.4f}, N5={pcc_n5:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

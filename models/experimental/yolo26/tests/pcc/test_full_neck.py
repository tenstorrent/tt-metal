# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test full neck (model.10-22) for YOLO26.
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
def test_full_neck(device):
    """Test all neck layers (model.10-22)."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtConvBNSiLU, TtC2f, TtC3k2, TtSPPF, TtC2PSA, TtUpsample
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # Create backbone layers (model.0-9)
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

    # Create neck layers
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

    from models.experimental.yolo26.tt.ttnn_yolo26 import TtC3k2PSA

    c3k2_22 = TtC3k2PSA(device, 384, 256, hidden_channels=128, n=1, name="model.22")
    c3k2_22.load_weights(weight_loader, "model.22")

    # Input
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # === PyTorch reference ===
    # Store intermediate outputs for skip connections
    pt_intermediates = {}
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(23):
            layer = torch_model.model.model[i]

            # Handle concat layers
            if layer.__class__.__name__ == "Concat":
                from_indices = layer.f
                tensors = []
                for idx in from_indices:
                    if idx == -1:
                        tensors.append(x_pt)
                    else:
                        tensors.append(pt_intermediates[idx])
                x_pt = torch.cat(tensors, dim=1)
            else:
                x_pt = layer(x_pt)

            pt_intermediates[i] = x_pt.clone()

            if i in [4, 6, 10, 13, 16, 19, 22]:
                logger.info(f"PyTorch model.{i}: shape={x_pt.shape}, mean={x_pt.mean():.4f}")

    # === TTNN forward ===
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run backbone and store intermediates
    tt_intermediates = {}
    h, w = input_size, input_size
    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    for i, layer in enumerate(backbone_layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)

        # Convert and store
        tt_x_conv = to_nhwc(tt_x, batch_size, h, w, out_channels[i])
        tt_intermediates[i] = (ttnn.to_torch(tt_x_conv), h, w, out_channels[i])

        # Prepare for next layer
        tt_x = ttnn.from_torch(tt_intermediates[i][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # model.10: C2PSA
    tt_x, h, w = c2psa_10(tt_x, batch_size, h, w)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[10] = (ttnn.to_torch(tt_x_conv), h, w, 256)

    # Compare model.10
    pt_out_nhwc = pt_intermediates[10].permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, tt_intermediates[10][0].float(), 0.90)
    logger.info(f"model.10 (C2PSA): PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

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

    # Compare model.13
    pt_out_nhwc = pt_intermediates[13].permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, tt_intermediates[13][0].float(), 0.90)
    logger.info(f"model.13 (C3k2): PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

    # model.14: Upsample
    tt_x = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 40, 40, 128)

    # model.15: Concat with model.4
    tt_x4 = ttnn.from_torch(tt_intermediates[4][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x4], dim=3)

    # model.16: C3k2 -> N3 output
    tt_x, h, w = c3k2_16(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_intermediates[16] = (ttnn.to_torch(tt_x_conv), h, w, 64)
    n3 = tt_intermediates[16]

    # Compare model.16
    pt_out_nhwc = pt_intermediates[16].permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, n3[0].float(), 0.90)
    logger.info(f"model.16 (N3): PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

    # model.17: Downsample
    tt_x = ttnn.from_torch(n3[0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_17(tt_x, batch_size, 80, 80)

    # model.18: Concat with model.13
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x13 = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x13], dim=3)

    # model.19: C3k2 -> N4 output
    tt_x, h, w = c3k2_19(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_intermediates[19] = (ttnn.to_torch(tt_x_conv), h, w, 128)
    n4 = tt_intermediates[19]

    # Compare model.19
    pt_out_nhwc = pt_intermediates[19].permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, n4[0].float(), 0.90)
    logger.info(f"model.19 (N4): PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

    # model.20: Downsample
    tt_x = ttnn.from_torch(n4[0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_20(tt_x, batch_size, 40, 40)

    # model.21: Concat with model.10
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x10 = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x10], dim=3)

    # model.22: C3k2 -> N5 output
    tt_x, h, w = c3k2_22(tt_x, batch_size, 20, 20)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[22] = (ttnn.to_torch(tt_x_conv), h, w, 256)
    n5 = tt_intermediates[22]

    # Compare model.22
    pt_out_nhwc = pt_intermediates[22].permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, n5[0].float(), 0.90)
    logger.info(f"model.22 (N5): PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

    # Summary
    logger.info("\n=== Neck Output Summary ===")
    logger.info(f"N3 (model.16): shape={n3[0].shape}, mean={n3[0].float().mean():.4f}")
    logger.info(f"N4 (model.19): shape={n4[0].shape}, mean={n4[0].float().mean():.4f}")
    logger.info(f"N5 (model.22): shape={n5[0].shape}, mean={n5[0].float().mean():.4f}")

    assert passed, f"Neck failed at model.22 with PCC {pcc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

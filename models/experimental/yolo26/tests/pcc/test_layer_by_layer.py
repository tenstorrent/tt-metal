# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-layer test to identify where YOLO26 breaks.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_stem_and_conv1(device):
    """Test stem (model.0) and conv1 (model.1) only."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtConvBNSiLU
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    # Create layers
    stem = TtConvBNSiLU(device, 3, 16, kernel_size=3, stride=2, padding=1, name="model.0")
    conv1 = TtConvBNSiLU(device, 16, 32, kernel_size=3, stride=2, padding=1, name="model.1")

    w, b = weight_loader.get_conv_bn("model.0")
    stem.load_weights(w, b)
    w, b = weight_loader.get_conv_bn("model.1")
    conv1.load_weights(w, b)

    # Test input
    batch_size = 1
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, 640, 640, dtype=torch.bfloat16)
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run stem
    out, h, w = stem(tt_x, batch_size, 640, 640)
    logger.info(f"After stem: h={h}, w={w}, shape={out.shape}")

    # Run conv1
    out, h, w = conv1(out, batch_size, h, w)
    logger.info(f"After conv1: h={h}, w={w}, shape={out.shape}")

    # Compare with PyTorch
    with torch.no_grad():
        pt_out = torch_model.model.model[0](x_torch.float())
        pt_out = torch_model.model.model[1](pt_out)

    # Convert TT output
    if out.memory_config().is_sharded():
        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.reshape(out, [batch_size, h, w, 32])
    tt_out = ttnn.to_torch(out)

    pt_out_nhwc = pt_out.permute(0, 2, 3, 1).contiguous()

    from models.common.utility_functions import comp_pcc

    passed, pcc = comp_pcc(pt_out_nhwc, tt_out.float(), 0.95)
    logger.info(f"PCC: {pcc:.6f} - {'PASS' if passed else 'FAIL'}")
    assert passed


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_c2f_simple(device):
    """Test C2f (model.2) with simplified approach."""
    from ultralytics import YOLO

    torch_model = YOLO("yolo26n.pt")

    # Get the actual C2f layer
    c3k2 = torch_model.model.model[2]

    # Test input after conv1: [1, 32, 160, 160]
    batch_size = 1
    h, w = 160, 160
    in_ch = 32

    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, in_ch, h, w, dtype=torch.float32)

    # Run PyTorch C3k2
    with torch.no_grad():
        pt_out = c3k2(x_torch)

    logger.info(f"PyTorch C3k2 output: {pt_out.shape}")

    # Now test TTNN version
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtC2f
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    # Create TTNN C2f with correct parameters from PyTorch:
    # cv1: 32→32, cv2: 48→64, hidden=16
    tt_c2f = TtC2f(device, in_channels=32, out_channels=64, hidden_channels=16, n=1, name="model.2")
    tt_c2f.load_weights(weight_loader, "model.2")

    # Convert input
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc.to(torch.bfloat16), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run TTNN C2f
    tt_out, out_h, out_w = tt_c2f(tt_x, batch_size, h, w)

    logger.info(f"TTNN C2f output: h={out_h}, w={out_w}")

    # Compare
    if tt_out.memory_config().is_sharded():
        tt_out = ttnn.sharded_to_interleaved(tt_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
    tt_out = ttnn.reshape(tt_out, [batch_size, out_h, out_w, 64])
    tt_out_torch = ttnn.to_torch(tt_out)

    pt_out_nhwc = pt_out.permute(0, 2, 3, 1).contiguous()

    from models.common.utility_functions import comp_pcc

    passed, pcc = comp_pcc(pt_out_nhwc, tt_out_torch.float(), 0.90)
    logger.info(f"PCC: {pcc:.6f} - {'PASS' if passed else 'FAIL'}")
    assert passed


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_backbone_layers_0_to_4(device):
    """Test backbone through model.4 (first 5 layers)."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtConvBNSiLU, TtC2f
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # Create layers
    # model.0: Conv 3→16, stride=2
    stem = TtConvBNSiLU(device, 3, 16, kernel_size=3, stride=2, padding=1, name="model.0")
    # model.1: Conv 16→32, stride=2
    conv1 = TtConvBNSiLU(device, 16, 32, kernel_size=3, stride=2, padding=1, name="model.1")
    # model.2: C3k2 32→64, hidden=16
    c2f_2 = TtC2f(device, 32, 64, hidden_channels=16, n=1, name="model.2")
    # model.3: Conv 64→64, stride=2
    conv3 = TtConvBNSiLU(device, 64, 64, kernel_size=3, stride=2, padding=1, name="model.3")
    # model.4: C3k2 64→128, hidden=32
    c2f_4 = TtC2f(device, 64, 128, hidden_channels=32, n=1, name="model.4")

    # Load weights
    w, b = weight_loader.get_conv_bn("model.0")
    stem.load_weights(w, b)
    w, b = weight_loader.get_conv_bn("model.1")
    conv1.load_weights(w, b)
    c2f_2.load_weights(weight_loader, "model.2")
    w, b = weight_loader.get_conv_bn("model.3")
    conv3.load_weights(w, b)
    c2f_4.load_weights(weight_loader, "model.4")

    # Input
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # PyTorch forward
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(5):
            x_pt = torch_model.model.model[i](x_pt)
        logger.info(f"PyTorch output after model.4: {x_pt.shape}")

    # TTNN forward
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    h, w = input_size, input_size

    tt_x, h, w = stem(tt_x, batch_size, h, w)
    logger.info(f"After model.0: h={h}, w={w}")

    tt_x, h, w = conv1(tt_x, batch_size, h, w)
    logger.info(f"After model.1: h={h}, w={w}")

    tt_x, h, w = c2f_2(tt_x, batch_size, h, w)
    logger.info(f"After model.2: h={h}, w={w}")

    tt_x, h, w = conv3(tt_x, batch_size, h, w)
    logger.info(f"After model.3: h={h}, w={w}")

    tt_x, h, w = c2f_4(tt_x, batch_size, h, w)
    logger.info(f"After model.4: h={h}, w={w}")

    # Compare
    if tt_x.memory_config().is_sharded():
        tt_x = ttnn.sharded_to_interleaved(tt_x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.to_memory_config(tt_x, ttnn.DRAM_MEMORY_CONFIG)
    tt_x = ttnn.reshape(tt_x, [batch_size, h, w, 128])
    tt_out = ttnn.to_torch(tt_x)

    pt_out_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()

    from models.common.utility_functions import comp_pcc

    passed, pcc = comp_pcc(pt_out_nhwc, tt_out.float(), 0.90)
    logger.info(f"PCC after model.4: {pcc:.6f} - {'PASS' if passed else 'FAIL'}")
    assert passed


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_backbone_and_c2psa(device):
    """Test backbone (layers 0-9) + C2PSA (model.10)."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtConvBNSiLU, TtC2f, TtC3k2, TtSPPF, TtC2PSA
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # Create all layers
    layers = [
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
        TtC2PSA(device, 256, 256, n=1, name="model.10"),
    ]

    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256]

    # Load weights
    for i, layer in enumerate(layers):
        if isinstance(layer, TtConvBNSiLU):
            w, b = weight_loader.get_conv_bn(f"model.{i}")
            layer.load_weights(w, b)
        else:
            layer.load_weights(weight_loader, f"model.{i}")

    # Input
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # PyTorch forward
    pt_outputs = []
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(11):  # model.0-10
            x_pt = torch_model.model.model[i](x_pt)
            pt_outputs.append(x_pt.clone())
        logger.info(f"PyTorch model.10 output: {x_pt.shape}, mean={x_pt.mean():.4f}")

    # TTNN forward
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    h, w = input_size, input_size

    for i, layer in enumerate(layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)

        # Convert for comparison
        tt_x_cmp = tt_x
        if tt_x_cmp.memory_config().is_sharded():
            tt_x_cmp = ttnn.sharded_to_interleaved(tt_x_cmp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_x_cmp = ttnn.to_layout(tt_x_cmp, ttnn.ROW_MAJOR_LAYOUT)
        tt_x_cmp = ttnn.reshape(tt_x_cmp, [batch_size, h, w, out_channels[i]])
        tt_out_torch = ttnn.to_torch(tt_x_cmp)

        pt_out_nhwc = pt_outputs[i].permute(0, 2, 3, 1).contiguous()
        passed, pcc = comp_pcc(pt_out_nhwc, tt_out_torch.float(), 0.90)
        logger.info(f"model.{i}: PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

        # Prepare for next layer
        if i < len(layers) - 1:
            tt_x = ttnn.from_torch(tt_out_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    logger.info(f"TTNN model.10 output: mean={tt_out_torch.float().mean():.4f}")
    assert passed, f"Failed at model.10 with PCC {pcc}"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_full_backbone(device):
    """Test full backbone (layers 0-9) - continuous TTNN flow with PCC validation."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtConvBNSiLU, TtC2f, TtC3k2, TtSPPF
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1
    input_size = 640

    # Create all backbone layers
    layers = [
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

    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    # Load weights
    for i, layer in enumerate(layers):
        if isinstance(layer, TtConvBNSiLU):
            w, b = weight_loader.get_conv_bn(f"model.{i}")
            layer.load_weights(w, b)
        else:
            layer.load_weights(weight_loader, f"model.{i}")

    # Input
    torch.manual_seed(42)
    x_torch = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # PyTorch forward through backbone - keep intermediate outputs
    pt_outputs = []
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(10):
            x_pt = torch_model.model.model[i](x_pt)
            pt_outputs.append(x_pt.clone())
        logger.info(f"PyTorch backbone output: {x_pt.shape}, mean={x_pt.mean():.4f}")

    # TTNN forward - check PCC at each layer
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    h, w = input_size, input_size

    for i, layer in enumerate(layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)

        # Convert for comparison
        tt_x_cmp = tt_x
        if tt_x_cmp.memory_config().is_sharded():
            tt_x_cmp = ttnn.sharded_to_interleaved(tt_x_cmp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_x_cmp = ttnn.to_layout(tt_x_cmp, ttnn.ROW_MAJOR_LAYOUT)
        tt_x_cmp = ttnn.reshape(tt_x_cmp, [batch_size, h, w, out_channels[i]])
        tt_out_torch = ttnn.to_torch(tt_x_cmp)

        pt_out_nhwc = pt_outputs[i].permute(0, 2, 3, 1).contiguous()
        passed, pcc = comp_pcc(pt_out_nhwc, tt_out_torch.float(), 0.90)
        logger.info(f"model.{i}: h={h}, w={w}, PCC={pcc:.4f} - {'PASS' if passed else 'FAIL'}")

        # Prepare for next layer
        if i < 9:
            tt_x = ttnn.from_torch(tt_out_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Final result
    logger.info(f"TTNN backbone output: mean={tt_out_torch.float().mean():.4f}")
    logger.info(f"Full backbone PCC: {pcc:.6f} - {'PASS' if passed else 'FAIL'}")
    assert passed, f"Backbone failed with final PCC {pcc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 Basic Performance Test.

Usage:
    pytest models/experimental/yolo26/tests/perf/test_perf.py -v -s
"""

import time
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


def to_nhwc(t, batch_size, h, w, ch):
    """Convert tensor to NHWC format."""
    if t.memory_config().is_sharded():
        t = ttnn.sharded_to_interleaved(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [batch_size, h, w, ch])
    return t


def create_yolo26_model(device, weight_loader):
    """Create YOLO26 TTNN model."""
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

    return {
        "backbone": backbone_layers,
        "c2psa_10": c2psa_10,
        "upsample": upsample,
        "c3k2_13": c3k2_13,
        "c3k2_16": c3k2_16,
        "conv_17": conv_17,
        "c3k2_19": c3k2_19,
        "conv_20": conv_20,
        "c3k2_22": c3k2_22,
        "detect_head": detect_head,
    }


def run_yolo26_forward(model, tt_x, device, batch_size=1):
    """Run single forward pass."""
    backbone_layers = model["backbone"]
    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    tt_intermediates = {}
    h, w = 640, 640

    # Backbone
    for i, layer in enumerate(backbone_layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)
        tt_x_conv = to_nhwc(tt_x, batch_size, h, w, out_channels[i])
        tt_intermediates[i] = (ttnn.to_torch(tt_x_conv), h, w, out_channels[i])
        tt_x = ttnn.from_torch(tt_intermediates[i][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Neck
    tt_x, h, w = model["c2psa_10"](tt_x, batch_size, h, w)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[10] = (ttnn.to_torch(tt_x_conv), h, w, 256)

    tt_x = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = model["upsample"](tt_x, batch_size, 20, 20, 256)
    tt_x6 = ttnn.from_torch(tt_intermediates[6][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x6], dim=3)

    tt_x, h, w = model["c3k2_13"](tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_intermediates[13] = (ttnn.to_torch(tt_x_conv), h, w, 128)

    tt_x = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = model["upsample"](tt_x, batch_size, 40, 40, 128)
    tt_x4 = ttnn.from_torch(tt_intermediates[4][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x4], dim=3)

    tt_x, h, w = model["c3k2_16"](tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_n3 = ttnn.to_torch(tt_x_conv)

    tt_x = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = model["conv_17"](tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x13 = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x13], dim=3)

    tt_x, h, w = model["c3k2_19"](tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_n4 = ttnn.to_torch(tt_x_conv)

    tt_x = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = model["conv_20"](tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x10 = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x10], dim=3)

    tt_x, h, w = model["c3k2_22"](tt_x, batch_size, 20, 20)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_n5 = ttnn.to_torch(tt_x_conv)

    # Detection head
    n3_tensor = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n4_tensor = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n5_tensor = ttnn.from_torch(tt_n5, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_detect_out = model["detect_head"]((n3_tensor, 80, 80), (n4_tensor, 40, 40), (n5_tensor, 20, 20), batch_size)

    return tt_detect_out


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_yolo26_perf(device):
    """
    Basic performance test for YOLO26.
    Measures end-to-end inference time.
    """
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    # Load weights
    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    # Create model
    logger.info("Creating YOLO26 TTNN model...")
    model = create_yolo26_model(device, weight_loader)

    # Create input
    batch_size = 1
    x_torch = torch.randn(batch_size, 640, 640, 3, dtype=torch.bfloat16)

    # Warmup
    logger.info("Warmup runs...")
    for _ in range(3):
        tt_x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        _ = run_yolo26_forward(model, tt_x, device, batch_size)
        ttnn.synchronize_device(device)

    # Benchmark
    num_iterations = 10
    logger.info(f"Running {num_iterations} iterations...")

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        tt_x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        _ = run_yolo26_forward(model, tt_x, device, batch_size)
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time

    logger.info(f"\n{'='*60}")
    logger.info(f"YOLO26 Performance Results")
    logger.info(f"{'='*60}")
    logger.info(f"Input size: 640x640")
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Total time: {total_time*1000:.1f} ms")
    logger.info(f"Avg per inference: {avg_time*1000:.1f} ms")
    logger.info(f"FPS: {fps:.1f}")
    logger.info(f"{'='*60}")

    # Basic sanity check - should be at least 1 FPS
    assert fps > 1.0, f"FPS {fps:.1f} is too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

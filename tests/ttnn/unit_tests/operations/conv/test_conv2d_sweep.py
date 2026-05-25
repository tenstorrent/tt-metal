# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Conv2d shape-sweep harness.

Covers common real-model conv shapes: ResNet50 stages, SDXL UNet body, U-Net
segmentation, YOLO neck, ViT patch projection, plus a couple of edge cases
(wide pointwise, large-spatial small-channel). Designed to be runnable in
parallel with `pytest -n N --dist worksteal` — every parametrize point creates
its own device, so workers are independent.

Filter examples:
  pytest -k "resnet50"
  pytest -k "sdxl or unet"
  pytest -m "not slow"             # skip the largest configs (stem, 256x256)

See scripts/run_conv_sweep.sh for the recommended invocation.
"""

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import (
    run_conv,
    torch_tensor_map,
    HS,
    WS,
    BS,
)


# ---------------------------------------------------------------------------
# Conv presets
# Columns: batch, in_ch, out_ch, h, w, k, stride, pad4, shard_layout
# pad4 = (top, bottom, left, right)
# ---------------------------------------------------------------------------
CONV_CASES = [
    # --- ResNet50 stages ---
    pytest.param(1, 64, 64, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="resnet50_layer1_pw"),
    pytest.param(1, 64, 64, 56, 56, 3, 1, (1, 1, 1, 1), HS, id="resnet50_layer1_3x3"),
    pytest.param(1, 64, 256, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="resnet50_layer1_expand"),
    pytest.param(1, 256, 64, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="resnet50_layer1_reduce"),
    pytest.param(1, 128, 128, 28, 28, 3, 1, (1, 1, 1, 1), BS, id="resnet50_layer2_3x3"),
    pytest.param(1, 128, 512, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="resnet50_layer2_expand"),
    pytest.param(1, 512, 128, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="resnet50_layer2_reduce"),
    pytest.param(1, 256, 256, 14, 14, 3, 1, (1, 1, 1, 1), BS, id="resnet50_layer3_3x3"),
    pytest.param(1, 256, 1024, 14, 14, 1, 1, (0, 0, 0, 0), BS, id="resnet50_layer3_expand"),
    pytest.param(1, 256, 512, 14, 14, 1, 1, (0, 0, 0, 0), BS, id="resnet50_layer3_pw_up"),
    pytest.param(1, 512, 512, 7, 7, 3, 1, (1, 1, 1, 1), WS, id="resnet50_layer4_3x3"),
    pytest.param(1, 512, 2048, 7, 7, 1, 1, (0, 0, 0, 0), WS, id="resnet50_layer4_expand"),
    pytest.param(1, 256, 256, 28, 28, 3, 2, (1, 1, 1, 1), BS, id="resnet50_downsample_s2"),
    # --- SDXL UNet body (heavy channels) ---
    pytest.param(1, 320, 320, 32, 32, 3, 1, (1, 1, 1, 1), BS, id="sdxl_unet_320_32x32"),
    pytest.param(1, 640, 640, 16, 16, 3, 1, (1, 1, 1, 1), BS, id="sdxl_unet_640_16x16"),
    pytest.param(1, 1280, 1280, 16, 16, 3, 1, (1, 1, 1, 1), BS, id="sdxl_unet_1280_16x16", marks=pytest.mark.slow),
    pytest.param(1, 320, 640, 16, 16, 3, 2, (1, 1, 1, 1), BS, id="sdxl_downsample_320_640"),
    pytest.param(1, 640, 1280, 8, 8, 3, 2, (1, 1, 1, 1), WS, id="sdxl_downsample_640_1280"),
    # --- YOLO neck / backbone ---
    pytest.param(1, 32, 64, 40, 40, 3, 1, (1, 1, 1, 1), HS, id="yolo_neck_3x3"),
    pytest.param(1, 128, 256, 40, 40, 1, 1, (0, 0, 0, 0), BS, id="yolo_neck_pw"),
    pytest.param(1, 64, 128, 40, 40, 3, 2, (1, 1, 1, 1), BS, id="yolo_neck_down"),
    pytest.param(1, 256, 256, 20, 20, 3, 1, (1, 1, 1, 1), BS, id="yolo_head_3x3"),
    # --- MobileNetV2-style pointwise (no depthwise; sweep harness keeps groups=1) ---
    pytest.param(1, 16, 96, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="mbnetv2_pw_expand"),
    pytest.param(1, 96, 16, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="mbnetv2_pw_project"),
    pytest.param(1, 64, 384, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="mbnetv2_pw_mid"),
    # --- EfficientNet-style blocks ---
    pytest.param(1, 32, 192, 56, 56, 1, 1, (0, 0, 0, 0), HS, id="effnet_stage1_pw"),
    pytest.param(1, 80, 480, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="effnet_stage2_pw"),
    # --- ConvNext / VGG-ish ---
    pytest.param(1, 96, 96, 56, 56, 3, 1, (1, 1, 1, 1), HS, id="convnext_depth1"),
    pytest.param(1, 192, 384, 28, 28, 3, 1, (1, 1, 1, 1), BS, id="convnext_depth2"),
    pytest.param(1, 64, 128, 56, 56, 3, 1, (1, 1, 1, 1), HS, id="vgg_block1_3x3"),
    pytest.param(1, 128, 256, 28, 28, 3, 1, (1, 1, 1, 1), BS, id="vgg_block2_3x3"),
    # --- U-Net segmentation ---
    pytest.param(1, 32, 64, 64, 64, 3, 1, (1, 1, 1, 1), HS, id="unet_encoder_3x3"),
    pytest.param(1, 64, 64, 32, 32, 3, 1, (1, 1, 1, 1), BS, id="unet_bottleneck"),
    pytest.param(1, 128, 128, 16, 16, 3, 1, (1, 1, 1, 1), BS, id="unet_deep_3x3"),
    # --- ViT / patch-projection (1x1) ---
    pytest.param(1, 16, 192, 32, 32, 1, 1, (0, 0, 0, 0), BS, id="vit_patch_proj_1x1"),
    pytest.param(1, 768, 768, 14, 14, 1, 1, (0, 0, 0, 0), BS, id="vit_base_pw"),
    # --- 1x1 / pointwise variety ---
    pytest.param(1, 192, 384, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="pw_192_384"),
    pytest.param(1, 384, 192, 28, 28, 1, 1, (0, 0, 0, 0), BS, id="pw_384_192"),
    pytest.param(1, 768, 192, 14, 14, 1, 1, (0, 0, 0, 0), BS, id="pw_768_192"),
    # --- 5x5 kernels ---
    pytest.param(1, 32, 64, 32, 32, 5, 1, (2, 2, 2, 2), HS, id="k5x5_32_64"),
    pytest.param(1, 128, 128, 16, 16, 5, 1, (2, 2, 2, 2), BS, id="k5x5_128_128"),
    # --- Batched cases ---
    pytest.param(2, 64, 64, 56, 56, 3, 1, (1, 1, 1, 1), HS, id="batch2_residual"),
    pytest.param(4, 64, 128, 28, 28, 3, 1, (1, 1, 1, 1), BS, id="batch4_3x3"),
    pytest.param(8, 32, 64, 28, 28, 3, 1, (1, 1, 1, 1), BS, id="batch8_3x3"),
    # --- Wide pointwise edge ---
    pytest.param(1, 1024, 1024, 8, 8, 1, 1, (0, 0, 0, 0), WS, id="wide_pw_1024"),
    pytest.param(1, 2048, 512, 7, 7, 1, 1, (0, 0, 0, 0), WS, id="wide_pw_2048_512"),
    # --- Edge / stress (marked slow) ---
    pytest.param(1, 16, 16, 256, 256, 3, 1, (1, 1, 1, 1), HS, id="large_spatial_small_ch", marks=pytest.mark.slow),
    pytest.param(1, 3, 32, 224, 224, 7, 2, (3, 3, 3, 3), HS, id="resnet_stem", marks=pytest.mark.slow),
]


# ---------------------------------------------------------------------------
# Conv sweep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat8_b, ttnn.bfloat16),
    ],
    ids=["bf16_bf16", "bfp8_bf16"],
)
@pytest.mark.parametrize(
    "batch, in_ch, out_ch, h, w, k, stride, pad4, shard_layout",
    CONV_CASES,
)
@pytest.mark.timeout(1800)
def test_conv_sweep(
    device,
    torch_tensor_map,
    batch,
    in_ch,
    out_ch,
    h,
    w,
    k,
    stride,
    pad4,
    shard_layout,
    input_dtype,
    output_dtype,
):
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.HiFi2,
        output_dtype,
        None,
        batch,
        out_ch,
        in_ch,
        h,
        w,
        k,
        k,
        stride,
        stride,
        pad4,
        config_override=None,
        shard_layout=shard_layout,
        output_layout=ttnn.TILE_LAYOUT,
        has_bias=True,
        fp32_accum=False,
        packer_l1_acc=True,
        input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
        input_dtype=input_dtype,
    )

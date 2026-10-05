# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers shared by the backbone and FPN tests."""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.resnet import ResNet
from models.experimental.bevformer.tests.backbone_weights import init_dummy_backbone_weights, init_dummy_fpn_weights
from tests.ttnn.utils_for_testing import assert_with_pcc

NUM_CAMS = 6

# BEVFormer-base's camera images are 1600x900, padded to 1600x928 so the height is a
# multiple of 32 before the backbone.
IMAGE_HEIGHT = 928
IMAGE_WIDTH = 1600

# BEVFormer-base's ResNet101: caffe style, DCNv2 in layer3 and layer4, C3-C5 out.
RESNET_KWARGS = dict(
    depth=101,
    in_channels=3,
    stem_channels=None,
    base_channels=64,
    num_stages=4,
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    out_indices=(1, 2, 3),
    style="caffe",
    deep_stem=False,
    avg_down=False,
    frozen_stages=4,
    conv_cfg=None,
    norm_cfg={"type": "BN2d", "requires_grad": False},
    norm_eval=True,
    dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
    stage_with_dcn=(False, False, True, True),
    plugins=None,
    with_cp=False,
    zero_init_residual=True,
    pretrained=None,
    init_cfg=None,
)

FPN_KWARGS = dict(
    in_channels=[512, 1024, 2048],
    out_channels=256,
    start_level=0,
    add_extra_convs="on_output",
    num_outs=4,
    relu_before_extra_convs=True,
)

# Indices (0 is layer1) of the ResNet layers whose activations are kept in DRAM because
# their convs do not fit in L1. layer1 and layer2 work on 6 x 232 x 400 x 256 tensors
# (285 MB in bfloat16), layer3's convs fit only without fp32 accumulation, and layer4's
# 2048-channel 1x1 convs overflow L1 when sharded.
DRAM_ACTIVATION_STAGES = (0, 1, 2, 3)

# Layers whose convs accumulate in fp32. With trained weights, the error of bfloat16
# accumulation grows through the deep layers' 26 blocks until C5 falls below PCC 0.99.
FP32_ACC_STAGES = (2, 3)

# FPN levels whose convs keep their activations in DRAM because they do not fit in L1:
# C3 is 6 x 116 x 200 x 512 bfloat16 (143 MB) and C4 is 6 x 58 x 100 x 1024 bfloat16 (71 MB).
DRAM_ACTIVATION_LEVELS = (0, 1)

# Width slices for the spatial convs of the DRAM layers and levels, lowered per conv to what
# its output width allows. The tightest is the strided 1x1 downsample that opens layer2: its
# 6 x 232 x 400 x 256 input is in DRAM, and each slice is read into L1 as bfloat16
# ROW_MAJOR for the halo, 3.25 MB per L1 bank in total against 576 KB free, so it needs at
# least 6 slices. 8 leaves margin over that; this conv's 200-wide output caps it at 7.
DRAM_CONV_SLICES = 8

# Layers whose downsample conv is block sharded (layer3, layer4) and FPN levels whose output
# conv is (C3, C4). Both come from the UniAD port and are not re-tuned for 928x1600.
BLOCK_SHARDED_DOWNSAMPLE_STAGES = (2, 3)
BLOCK_SHARDED_LEVELS = (0, 1)


# Dtypes TtResNet emits for C3-C5.
BACKBONE_OUTPUT_DTYPES = [ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16]


# The references only run forward; without autograd they keep no activations for backward.
def build_reference_backbone():
    return init_dummy_backbone_weights(ResNet(**RESNET_KWARGS)).requires_grad_(False)


def build_reference_fpn():
    return init_dummy_fpn_weights(FPN(**FPN_KWARGS)).requires_grad_(False)


def tt_resnet_kwargs():
    """The TtResNet arguments matching RESNET_KWARGS."""
    return dict(
        out_indices=RESNET_KWARGS["out_indices"],
        dram_activation_stages=DRAM_ACTIVATION_STAGES,
        dram_conv_slices=DRAM_CONV_SLICES,
        block_sharded_downsample_stages=BLOCK_SHARDED_DOWNSAMPLE_STAGES,
        fp32_acc_stages=FP32_ACC_STAGES,
    )


def tt_fpn_kwargs():
    """The TtFPN arguments for this configuration."""
    return dict(
        dram_activation_levels=DRAM_ACTIVATION_LEVELS,
        dram_conv_slices=DRAM_CONV_SLICES,
        block_sharded_levels=BLOCK_SHARDED_LEVELS,
    )


def to_conv_layout(nchw, device, dtype, layout=ttnn.TILE_LAYOUT):
    """NCHW torch tensor -> (1, 1, N*H*W, C) device tensor, the layout conv2d reads and writes."""
    n, c, h, w = nchw.shape
    nhwc = nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
    return ttnn.from_torch(nhwc, device=device, dtype=dtype, layout=layout)


def from_conv_layout(tt_tensor, nchw_shape):
    """(1, 1, N*H*W, C) device tensor -> NCHW torch tensor of ``nchw_shape``."""
    n, c, h, w = nchw_shape
    return ttnn.to_torch(tt_tensor).reshape(n, h, w, c).permute(0, 3, 1, 2)


def assert_pcc(expected, actual, pcc):
    passed, message = assert_with_pcc(expected, actual, pcc)
    logger.info(f"PCC {message} (threshold {pcc})")
    return passed, message


# BEVFormer-base's image normalization (img_norm_cfg): BGR pixels minus this mean, std 1.
IMAGE_MEAN_BGR = (103.530, 116.280, 123.675)
# Random structure at these cell sizes and amplitudes, in pixel units, plus per-pixel noise.
IMAGE_NOISE_SCALES = ((8, 80.0), (32, 40.0), (128, 20.0))
IMAGE_PIXEL_NOISE = 8.0


def random_image_batch():
    """Random camera images at the scale BEVFormer feeds the backbone: smooth random structure at
    several scales plus pixel noise, clamped to [0, 255], minus ``IMAGE_MEAN_BGR``.

    Unit-variance noise is two orders of magnitude off that scale; with trained weights it
    measures a different backbone error than real inputs do.
    """
    images = torch.zeros(NUM_CAMS, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    for cell, amplitude in IMAGE_NOISE_SCALES:
        coarse = torch.rand(NUM_CAMS, 3, IMAGE_HEIGHT // cell + 1, IMAGE_WIDTH // cell + 1)
        images += amplitude * F.interpolate(
            coarse, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False
        )
    images += IMAGE_PIXEL_NOISE * torch.randn(images.shape)
    return images.clamp(0, 255) - torch.tensor(IMAGE_MEAN_BGR).view(1, 3, 1, 1)

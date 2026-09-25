# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.tests.backbone_weights import load_fpn_weights
from models.experimental.bevformer.tt.model_preprocessing_backbone import create_fpn_parameters
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from tests.ttnn.utils_for_testing import assert_with_pcc


NUM_CAMS = 6
BACKBONE_CHANNELS = [512, 1024, 2048]

# BEVFormer-base's camera images are 1600x900, padded to 1600x928 so the height is a
# multiple of 32 before the backbone.
IMAGE_HEIGHT = 928
IMAGE_WIDTH = 1600

# Pyramid levels whose convs keep their activations in DRAM because they do not fit in
# L1. The FPN reads the backbone's C3-C5, at 1/8, 1/16 and 1/32 of the image; C3 alone
# is 6 x 116 x 200 x 512 bf16 = 142 MB.
DRAM_ACTIVATION_LEVELS = (0, 1)


def _backbone_output_shapes(height, width):
    """(H, W) of C3, C4, C5: stride 8, then each stride-2 stage rounds up."""
    shapes = [(height // 8, width // 8)]
    for _ in BACKBONE_CHANNELS[1:]:
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    return shapes


def _assert_pcc(expected, actual, pcc):
    passed, message = assert_with_pcc(expected, actual, pcc)
    logger.info(f"PCC {message} (threshold {pcc})")
    return passed, message


def _to_ttnn_nhwc(tensor, device):
    """NCHW torch tensor -> (1, 1, N*H*W, C) bfloat8_b device tensor, the layout the backbone emits."""
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    tensor = torch.reshape(tensor, [1, 1, tensor.shape[0] * tensor.shape[1] * tensor.shape[2], tensor.shape[-1]])
    return ttnn.from_torch(tensor, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_fpn(device, reset_seeds):
    torch_model = FPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    )
    torch_model = load_fpn_weights(torch_model)

    input_tensors = [
        torch.randn(NUM_CAMS, channels, h, w)
        for channels, (h, w) in zip(BACKBONE_CHANNELS, _backbone_output_shapes(IMAGE_HEIGHT, IMAGE_WIDTH))
    ]
    parameters = create_fpn_parameters(torch_model, input_tensors)
    torch_outputs = torch_model(input_tensors)

    tt_model = TtFPN(
        conv_args=parameters.model_args,
        conv_pth=parameters,
        device=device,
        input_dtypes=[ttnn.bfloat8_b] * len(input_tensors),
        dram_activation_levels=DRAM_ACTIVATION_LEVELS,
    )
    tt_outputs = tt_model([_to_ttnn_nhwc(tensor, device) for tensor in input_tensors])

    for torch_output, tt_output in zip(torch_outputs, tt_outputs):
        n, c, h, w = torch_output.shape
        tt_output = ttnn.to_torch(tt_output).reshape(n, h, w, c).permute(0, 3, 1, 2)
        _assert_pcc(torch_output, tt_output, 0.99)

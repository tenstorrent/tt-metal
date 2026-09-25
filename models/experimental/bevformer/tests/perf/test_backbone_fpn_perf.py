# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the BEVFormer backbone (ResNet101-DCN) and FPN neck at 1600x900.

Same models and inputs as ``test_backbone_fpn``: a PCC gate that doubles as the warmup,
then signposted iterations so the report covers already-compiled programs. A
``fpn`` signpost splits each iteration between the backbone and the neck.

One forward runs a few thousand device programs, above the profiler's default
1000-program buffer; run with ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` raised so
none are dropped.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.resnet import ResNet
from models.experimental.bevformer.tests.backbone_weights import load_backbone_weights, load_fpn_weights
from models.experimental.bevformer.tests.pcc.test_backbone_fpn import (
    DRAM_ACTIVATION_LEVELS,
    DRAM_ACTIVATION_STAGES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CAMS,
    RESNET_KWARGS,
)
from models.experimental.bevformer.tt.model_preprocessing_backbone import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from models.experimental.bevformer.tt.tt_resnet import TtResNet
from tests.ttnn.utils_for_testing import assert_with_pcc

DEVICE_PERF_ITERS = 1


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


@torch.no_grad()
@pytest.mark.timeout(1400)
@pytest.mark.parametrize("expected_pcc", [0.99])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_backbone_fpn_perf(device, expected_pcc, reset_seeds):
    logger.info(f"device-perf run of commit {_head_sha()}")

    torch_backbone = ResNet(
        **RESNET_KWARGS,
        norm_cfg={"type": "BN2d", "requires_grad": False},
        norm_eval=True,
        dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
    )
    torch_backbone = load_backbone_weights(torch_backbone)
    torch_fpn = FPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    )
    torch_fpn = load_fpn_weights(torch_fpn)

    torch_input = torch.randn(NUM_CAMS, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)

    tt_backbone = TtResNet(
        backbone_parameters.conv_args,
        backbone_parameters["res_model"],
        device,
        **RESNET_KWARGS,
        dcn=True,
        dram_activation_stages=DRAM_ACTIVATION_STAGES,
    )
    tt_fpn = TtFPN(
        conv_args=fpn_parameters.model_args,
        conv_pth=fpn_parameters,
        device=device,
        input_dtypes=tt_backbone.output_dtypes,
        dram_activation_levels=DRAM_ACTIVATION_LEVELS,
    )

    # Uploaded once per iteration outside the signposted region, so it measures the
    # model and not the transfer. The backbone consumes its input, so each iteration
    # needs a fresh one.
    nhwc = torch_input.permute(0, 2, 3, 1).reshape(1, 1, NUM_CAMS * IMAGE_HEIGHT * IMAGE_WIDTH, 3)

    def upload_input():
        return ttnn.from_torch(nhwc, device=device, dtype=ttnn.bfloat16)

    # Doubles as the warmup: compiles the kernels and fills the program cache, so the
    # signposted iterations already run at steady state.
    tt_outputs = tt_fpn(list(tt_backbone(upload_input())))
    for torch_output, tt_output in zip(torch_outputs, tt_outputs):
        n, c, h, w = torch_output.shape
        tt_output_torch = ttnn.to_torch(tt_output).reshape(n, h, w, c).permute(0, 3, 1, 2)
        _, message = assert_with_pcc(torch_output, tt_output_torch, expected_pcc)
        logger.info(f"PCC gate: {message}")
        ttnn.deallocate(tt_output)

    inputs = [upload_input() for _ in range(DEVICE_PERF_ITERS)]
    ttnn.synchronize_device(device)
    # Drains the profiler buffers so the signposted region starts from empty; the PCC
    # call's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    outputs = []
    signpost("start")
    for tt_input in inputs:
        features = tt_backbone(tt_input)
        signpost("fpn")
        outputs.append(tt_fpn(list(features)))
        ttnn.synchronize_device(device)
    signpost("stop")

    for out in outputs:
        for tensor in out:
            ttnn.deallocate(tensor)

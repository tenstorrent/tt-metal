# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the ResNet101-DCN backbone and FPN neck device path.

Same inputs as ``test_backbone_fpn``: a PCC gate that doubles as the warmup, then the
forward captured as a trace and one signposted replay, so the report covers
already-compiled programs with no host dispatch in between.

The capture is also the check that the forward has no host operations: any read or
write between host and device inside a capture region fails it. The replay's outputs
are checked against the reference after the signposted region.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.tests.backbone_common import (
    build_reference_backbone,
    build_reference_fpn,
    from_conv_layout,
    random_image_batch,
    to_conv_layout,
    tt_fpn_kwargs,
    tt_resnet_kwargs,
)
from models.experimental.bevformer.tests.test_utils import assert_pcc
from models.experimental.bevformer.tt.model_preprocessing_backbone import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from models.experimental.bevformer.tt.tt_resnet import TtResNet


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _check(torch_outputs, tt_outputs):
    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), 0.99)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the backbone's and FPN's recorded commands, not a measured size.
    [{"l1_small_size": 4 * 8192, "trace_region_size": 32 * 1024 * 1024}],
    indirect=True,
)
def test_backbone_fpn_perf(device, reset_seeds):
    logger.info(f"device-perf run of commit {_head_sha()}")

    torch_backbone = build_reference_backbone()
    torch_fpn = build_reference_fpn()
    torch_input = random_image_batch()
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)
    tt_backbone = TtResNet(
        backbone_parameters.conv_args, backbone_parameters["res_model"], device, **tt_resnet_kwargs()
    )
    tt_fpn = TtFPN(
        conv_args=fpn_parameters.conv_args,
        conv_pth=fpn_parameters,
        device=device,
        input_dtypes=tt_backbone.output_dtypes,
        **tt_fpn_kwargs(),
    )

    # Uploaded once so the profiled replay measures the backbone and FPN, not the transfer.
    tt_input = to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def op_fn():
        return tt_fpn(list(tt_backbone(tt_input)))

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    tt_outputs = op_fn()
    _check(torch_outputs, tt_outputs)
    for out in tt_outputs:
        ttnn.deallocate(out)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = op_fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.synchronize_device(device)
    # Drains and resets the device profiler buffers so the signposted region starts from
    # empty; the warmup's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    signpost("stop")

    try:
        _check(torch_outputs, tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)

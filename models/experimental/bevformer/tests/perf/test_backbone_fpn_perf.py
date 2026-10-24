# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the backbone and FPN neck device path.

Same inputs as ``test_backbone_fpn``: a PCC gate that doubles as the warmup, then the
forward captured as a trace and one signposted replay, so the report covers
already-compiled programs with no host dispatch in between.

The capture is also the check that the forward has no host operations: any read or
write between host and device inside a capture region fails it. The replay's outputs
are checked against the reference after the signposted region.
"""

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import RESNET_KWARGS
from models.experimental.bevformer.tests.common import (
    BACKBONE_FPN_PCC,
    signposted_trace,
    assert_pcc,
    build_reference_backbone,
    build_reference_fpn,
    from_conv_layout,
    random_image_batch,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.tt_fpn import TTFPN
from models.experimental.bevformer.tt.tt_resnet import TTResNet


def _check(torch_outputs, tt_outputs):
    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), BACKBONE_FPN_PCC)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the backbone's and FPN's recorded commands, not a measured size.
    [{"l1_small_size": 4 * 8192, "trace_region_size": 32 * 1024 * 1024}],
    indirect=True,
)
def test_backbone_fpn_perf(device, reset_seeds):
    torch_backbone = build_reference_backbone()
    torch_fpn = build_reference_fpn()
    torch_input = random_image_batch()
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)
    tt_backbone = TTResNet(backbone_parameters, device, out_indices=RESNET_KWARGS["out_indices"])
    tt_fpn = TTFPN(fpn_parameters, device, input_dtypes=tt_backbone.output_dtypes)

    # Uploaded once so the profiled replay measures the backbone and FPN, not the transfer.
    tt_input = to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def op_fn():
        return tt_fpn(list(tt_backbone(tt_input)))

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    tt_outputs = op_fn()
    _check(torch_outputs, tt_outputs)
    for out in tt_outputs:
        ttnn.deallocate(out)

    with signposted_trace(device, op_fn) as (_, tt_outputs):
        _check(torch_outputs, tt_outputs)

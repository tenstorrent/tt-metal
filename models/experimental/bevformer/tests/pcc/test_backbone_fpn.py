# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Camera images through the backbone and the FPN neck, end to end.

Checks the backbone's outputs (``RESNET_KWARGS["out_indices"]``) and the FPN's outputs against the reference. The TT backbone's
outputs feed the TT FPN as they are, so this also checks the hand-off between the two
(layout, dtype, memory).
"""

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import RESNET_KWARGS
from models.experimental.bevformer.tests.common import (
    BACKBONE_FPN_PCC,
    BACKBONE_OUTPUT_DTYPES,
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


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_backbone_fpn(device, reset_seeds):
    torch_backbone = build_reference_backbone()
    torch_fpn = build_reference_fpn()

    torch_input = random_image_batch()
    torch_features = torch_backbone(torch_input)
    torch_outputs = torch_fpn(list(torch_features))

    backbone_parameters = create_resnet_parameters(torch_backbone, torch_input)
    fpn_parameters = create_fpn_parameters(torch_fpn, torch_features)

    tt_backbone = TTResNet(backbone_parameters, device, out_indices=RESNET_KWARGS["out_indices"])
    # test_fpn feeds the FPN these dtypes on their own.
    assert tt_backbone.output_dtypes == BACKBONE_OUTPUT_DTYPES
    tt_fpn = TTFPN(fpn_parameters, device, input_dtypes=tt_backbone.output_dtypes)

    tt_input = to_conv_layout(torch_input, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_features = tt_backbone(tt_input)
    # Read before the FPN runs: it deallocates its inputs.
    for torch_feature, tt_feature in zip(torch_features, tt_features, strict=True):
        assert_pcc(torch_feature, from_conv_layout(tt_feature, torch_feature.shape), BACKBONE_FPN_PCC)
    tt_outputs = tt_fpn(list(tt_features))

    for torch_output, tt_output in zip(torch_outputs, tt_outputs, strict=True):
        assert_pcc(torch_output, from_conv_layout(tt_output, torch_output.shape), BACKBONE_FPN_PCC)


def test_to_conv_layout_rounds_as_from_torch():
    """to_conv_layout casts to bfloat16 in torch before from_torch; the images must come out bit for
    bit as from_torch's own float32 conversion makes them. Host only: no device."""
    torch.manual_seed(0)
    nchw = torch.randn(2, 3, 32, 48) * 3
    # Every float32 exactly halfway between two bfloat16 values, which only the tie rule decides.
    ties = (torch.arange(1 << 16, dtype=torch.int32) << 16 | 0x8000).view(torch.float32)
    ties = ties[torch.isfinite(ties)]
    ties = ties[: ties.numel() // 3 * 3].view(1, 3, -1, 1)
    for image in (nchw, ties):
        n, c, h, w = image.shape
        reference = ttnn.from_torch(
            image.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        got = to_conv_layout(image, None, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        assert torch.equal(ttnn.to_torch(got).view(torch.int16), ttnn.to_torch(reference).view(torch.int16))

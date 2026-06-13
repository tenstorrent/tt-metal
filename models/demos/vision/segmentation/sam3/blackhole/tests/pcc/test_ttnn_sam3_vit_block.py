# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_vit_mlp(device, reset_seeds, sam3_vit_backbone):
    """Test ViT MLP block against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_common import (
        preprocess_linear_bias,
        preprocess_linear_weight,
    )
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import tt_vit_mlp

    torch.manual_seed(42)

    ref_mlp = sam3_vit_backbone.blocks[0].mlp
    dim = 1024

    x = torch.randn(1, 196, dim)  # (B, seq_len, dim)

    with torch.no_grad():
        ref_output = ref_mlp(x)

    # Preprocess weights
    fc1_w = preprocess_linear_weight(ref_mlp.fc1.weight.data)
    fc1_b = preprocess_linear_bias(ref_mlp.fc1.bias.data)
    fc2_w = preprocess_linear_weight(ref_mlp.fc2.weight.data)
    fc2_b = preprocess_linear_bias(ref_mlp.fc2.bias.data)

    fc1_w = ttnn.to_device(fc1_w, device)
    fc1_b = ttnn.to_device(fc1_b, device)
    fc2_w = ttnn.to_device(fc2_w, device)
    fc2_b = ttnn.to_device(fc2_b, device)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_vit_mlp(tt_x, fc1_w, fc1_b, fc2_w, fc2_b, device=device)

    output = ttnn.to_torch(tt_output).float()
    assert_with_pcc(ref_output.float(), output, 0.99)

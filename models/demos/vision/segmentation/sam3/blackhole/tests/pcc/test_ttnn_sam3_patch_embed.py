# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_tt_patch_embed(device, reset_seeds, batch_size, sam3_vit_backbone):
    """Test PatchEmbed against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import tt_patch_embed, get_patch_embed_params

    torch.manual_seed(42)

    # Reference
    ref_model = sam3_vit_backbone
    pixel_values = torch.randn(batch_size, 3, 1008, 1008)

    with torch.no_grad():
        ref_output = ref_model.patch_embed(pixel_values)  # (B, H, W, C) = (1, 72, 72, 1024)

    # Our implementation
    params = get_patch_embed_params(ref_model)
    tt_output = tt_patch_embed(pixel_values, params["weight"], params["bias"], device)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare - shape should match
    assert tt_output_torch.shape[-3:] == ref_output.shape[-3:], (
        f"Shape mismatch: {tt_output_torch.shape} vs {ref_output.shape}"
    )

    # PCC check
    assert_with_pcc(ref_output.float(), tt_output_torch.float(), 0.999)

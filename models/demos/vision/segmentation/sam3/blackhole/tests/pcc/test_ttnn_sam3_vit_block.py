# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_tt_vit_block_windowed(device, reset_seeds, batch_size, sam3_vit_backbone):
    """Test a single windowed ViT block against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        tt_vit_block,
        preprocess_vit_block_weights,
        move_block_params_to_device,
    )

    torch.manual_seed(42)

    # Block 0 uses window attention (ws=24)
    ref_block = sam3_vit_backbone.blocks[0]
    assert ref_block.window_size == 24

    # Create input (B, H, W, C)
    H, W, C = 72, 72, 1024
    x = torch.randn(batch_size, H, W, C)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_block(x)

    # Preprocess weights
    block_params = preprocess_vit_block_weights(ref_block)
    block_params = move_block_params_to_device(block_params, device)

    # Run ttnn block
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_vit_block(
        tt_x,
        block_params["tt_params"],
        num_heads=16,
        window_size=block_params["window_size"],
        device=device,
    )

    output = ttnn.to_torch(tt_output).float()
    assert output.shape == ref_output.shape, f"Shape mismatch: {output.shape} vs {ref_output.shape}"
    assert_with_pcc(ref_output.float(), output, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_tt_vit_block_global(device, reset_seeds, batch_size, sam3_vit_backbone):
    """Test a single global attention ViT block against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        tt_vit_block,
        preprocess_vit_block_weights,
        move_block_params_to_device,
    )

    torch.manual_seed(42)

    # Block 7 uses global attention (ws=0)
    ref_block = sam3_vit_backbone.blocks[7]
    assert ref_block.window_size == 0

    # Create input (B, H, W, C)
    H, W, C = 72, 72, 1024
    x = torch.randn(batch_size, H, W, C)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_block(x)

    # Preprocess weights
    block_params = preprocess_vit_block_weights(ref_block)
    block_params = move_block_params_to_device(block_params, device)

    # Run ttnn block
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_vit_block(
        tt_x,
        block_params["tt_params"],
        num_heads=16,
        window_size=block_params["window_size"],
        device=device,
    )

    output = ttnn.to_torch(tt_output).float()
    assert output.shape == ref_output.shape, f"Shape mismatch: {output.shape} vs {ref_output.shape}"
    assert_with_pcc(ref_output.float(), output, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_vit_mlp(device, reset_seeds, sam3_vit_backbone):
    """Test ViT MLP block against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import tt_vit_mlp
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_common import (
        preprocess_linear_weight,
        preprocess_linear_bias,
    )

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

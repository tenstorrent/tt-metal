# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", [576])  # 24*24 = 576 for window attention
@pytest.mark.parametrize("batch_size", [1])
def test_tt_vit_attention(device, reset_seeds, batch_size, seq_len, sam3_vit_backbone):
    """Test ViT attention against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        tt_vit_attention,
        preprocess_vit_attention_weights,
    )

    torch.manual_seed(42)

    # Get a windowed attention block (block 0 uses window attention)
    ref_attn = sam3_vit_backbone.blocks[0].attn
    dim = 1024

    # Create 3D input (B, L, dim) - matching windowed attention input shape
    x = torch.randn(batch_size, seq_len, dim)

    # PyTorch reference forward pass
    with torch.no_grad():
        ref_output = ref_attn(x)

    # Preprocess weights for ttnn
    params = preprocess_vit_attention_weights(ref_attn)

    # Move weights to device
    tt_qkv_w = ttnn.to_device(params["qkv_weight"], device)
    tt_qkv_b = ttnn.to_device(params["qkv_bias"], device)
    tt_proj_w = ttnn.to_device(params["proj_weight"], device)
    tt_proj_b = ttnn.to_device(params["proj_bias"], device)

    # Move input to device
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn attention
    tt_output = tt_vit_attention(
        tt_x,
        tt_qkv_w,
        tt_qkv_b,
        tt_proj_w,
        tt_proj_b,
        num_heads=16,
        freqs_cis=params["freqs_cis"],
        device=device,
    )

    output = ttnn.to_torch(tt_output).float()
    ref_output_float = ref_output.float()

    # Reshape output to match reference if needed
    if output.shape != ref_output_float.shape:
        output = output.reshape(ref_output_float.shape)

    assert_with_pcc(ref_output_float, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("window_size", [24])
@pytest.mark.parametrize("batch_size", [1])
def test_tt_window_partition_unpartition(device, reset_seeds, batch_size, window_size):
    """Test window partition and unpartition round-trip."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        tt_window_partition,
        tt_window_unpartition,
    )

    torch.manual_seed(42)
    H, W, C = 72, 72, 1024
    x = torch.randn(batch_size, H, W, C)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Partition
    tt_windows, (Hp, Wp) = tt_window_partition(tt_x, window_size)

    # Verify shape: 72/24=3 windows per dim -> 3*3=9 windows per batch
    num_windows = (Hp // window_size) * (Wp // window_size)
    windows_torch = ttnn.to_torch(tt_windows)
    assert windows_torch.shape == (batch_size * num_windows, window_size, window_size, C), (
        f"Expected ({batch_size * num_windows}, {window_size}, {window_size}, {C}), "
        f"got {windows_torch.shape}"
    )

    # Unpartition
    tt_restored = tt_window_unpartition(tt_windows, window_size, (Hp, Wp), (H, W))
    restored = ttnn.to_torch(tt_restored).float()

    assert_with_pcc(x.float(), restored, 0.999)


def test_preprocess_vit_attention_weights(sam3_vit_backbone):
    """Test that attention weight preprocessing produces correct shapes."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        preprocess_vit_attention_weights,
    )

    ref_attn = sam3_vit_backbone.blocks[0].attn
    params = preprocess_vit_attention_weights(ref_attn)

    dim = 1024

    # QKV weight: [dim, dim*3]
    qkv_w = ttnn.to_torch(params["qkv_weight"])
    assert qkv_w.shape == (dim, dim * 3), f"Expected ({dim}, {dim*3}), got {qkv_w.shape}"

    # QKV bias: [1, 1, dim*3]
    qkv_b = ttnn.to_torch(params["qkv_bias"])
    assert qkv_b.shape == (1, 1, dim * 3), f"Expected (1, 1, {dim*3}), got {qkv_b.shape}"

    # Proj weight: [dim, dim]
    proj_w = ttnn.to_torch(params["proj_weight"])
    assert proj_w.shape == (dim, dim), f"Expected ({dim}, {dim}), got {proj_w.shape}"

    # Proj bias: [1, 1, dim]
    proj_b = ttnn.to_torch(params["proj_bias"])
    assert proj_b.shape == (1, 1, dim), f"Expected (1, 1, {dim}), got {proj_b.shape}"

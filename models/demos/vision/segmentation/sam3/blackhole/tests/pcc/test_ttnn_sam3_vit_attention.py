# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_preprocess_vit_attention_weights(sam3_vit_backbone):
    """Test that attention weight preprocessing produces correct shapes."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import preprocess_vit_attention_weights

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

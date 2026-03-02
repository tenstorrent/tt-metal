# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Vision Backbone (complete pipeline).

Validates the full vision pipeline: ViT -> Pooling -> Projector.
"""


import torch

import ttnn


def get_vision_backbone_weights(model_id: str = "allenai/Molmo2-8B"):
    """Load all vision backbone weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Get all vision backbone weights
    keys = []

    # ViT weights (25 layers)
    vit_prefix = "model.vision_backbone.image_vit"
    keys.extend(
        [
            f"{vit_prefix}.patch_embedding.weight",
            f"{vit_prefix}.patch_embedding.bias",
            f"{vit_prefix}.pre_ln.weight",
            f"{vit_prefix}.pre_ln.bias",
        ]
    )

    for i in range(25):
        block_prefix = f"{vit_prefix}.transformer.resblocks.{i}"
        keys.extend(
            [
                f"{block_prefix}.attention_norm.weight",
                f"{block_prefix}.attention_norm.bias",
                f"{block_prefix}.attention.wq.weight",
                f"{block_prefix}.attention.wq.bias",
                f"{block_prefix}.attention.wk.weight",
                f"{block_prefix}.attention.wk.bias",
                f"{block_prefix}.attention.wv.weight",
                f"{block_prefix}.attention.wv.bias",
                f"{block_prefix}.attention.wo.weight",
                f"{block_prefix}.attention.wo.bias",
                f"{block_prefix}.ffn_norm.weight",
                f"{block_prefix}.ffn_norm.bias",
                f"{block_prefix}.feed_forward.w1.weight",
                f"{block_prefix}.feed_forward.w1.bias",
                f"{block_prefix}.feed_forward.w2.weight",
                f"{block_prefix}.feed_forward.w2.bias",
            ]
        )

    # Pooling weights
    pool_prefix = "model.vision_backbone.image_pooling_2d"
    keys.extend(
        [
            f"{pool_prefix}.wq.weight",
            f"{pool_prefix}.wq.bias",
            f"{pool_prefix}.wk.weight",
            f"{pool_prefix}.wk.bias",
            f"{pool_prefix}.wv.weight",
            f"{pool_prefix}.wv.bias",
            f"{pool_prefix}.wo.weight",
            f"{pool_prefix}.wo.bias",
        ]
    )

    # Projector weights
    proj_prefix = "model.vision_backbone.image_projector"
    keys.extend(
        [
            f"{proj_prefix}.w1.weight",
            f"{proj_prefix}.w2.weight",
            f"{proj_prefix}.w3.weight",
        ]
    )

    return load_state_dict_from_safetensors(model_id, keys)


def test_vision_backbone_encode_only(device):
    """
    Test VisionBackbone encode_only (ViT forward without pooling/projection).

    This tests the ViT encoder and multi-scale feature extraction.
    """
    from models.demos.molmo2.demo.demo import load_model_weights
    from models.demos.molmo2.tt.vision_backbone import VisionBackbone

    hidden_dim = 1152
    num_patches = 729  # 27x27 for 378x378 image with patch_size=14

    # Load weights using the demo's weight loading function
    state_dict = load_model_weights()

    # Create TTNN backbone
    tt_backbone = VisionBackbone(
        mesh_device=device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
    )

    # Create random input (embedded patches, after patch+pos embedding)
    torch.manual_seed(42)
    x_torch = torch.randn(1, num_patches, hidden_dim, dtype=torch.float32)

    # Convert to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),  # [1, 1, num_patches, hidden_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Forward (encode only)
    features_ttnn = tt_backbone.forward_encode_only(x_ttnn)

    # Convert back to torch
    features_torch = ttnn.to_torch(features_ttnn).squeeze(0).squeeze(0)

    # Check output shape
    expected_shape = (num_patches, hidden_dim * 2)  # Concat of 2 feature layers
    assert features_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {features_torch.shape}"

    print(f"VisionBackbone encode_only output shape: {features_torch.shape}")
    print("VisionBackbone encode_only test passed!")


def test_vision_backbone_projector_integration(device):
    """
    Test ImagePooling + ImageProjector integration.

    Uses pre-computed features to test the adapter pipeline.
    """
    from models.demos.molmo2.tt.image_pooling import ImagePooling
    from models.demos.molmo2.tt.image_projector import ImageProjector
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    model_id = "allenai/Molmo2-8B"
    pool_input_dim = 2304
    adapter_hidden_dim = 1152
    output_dim = 4096
    num_queries = 64
    pool_size = 16

    # Load weights
    pool_keys = [
        "model.vision_backbone.image_pooling_2d.wq.weight",
        "model.vision_backbone.image_pooling_2d.wq.bias",
        "model.vision_backbone.image_pooling_2d.wk.weight",
        "model.vision_backbone.image_pooling_2d.wk.bias",
        "model.vision_backbone.image_pooling_2d.wv.weight",
        "model.vision_backbone.image_pooling_2d.wv.bias",
        "model.vision_backbone.image_pooling_2d.wo.weight",
        "model.vision_backbone.image_pooling_2d.wo.bias",
    ]
    proj_keys = [
        "model.vision_backbone.image_projector.w1.weight",
        "model.vision_backbone.image_projector.w2.weight",
        "model.vision_backbone.image_projector.w3.weight",
    ]
    state_dict = load_state_dict_from_safetensors(model_id, pool_keys + proj_keys)

    # Create modules
    pooling = ImagePooling(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=pool_input_dim,
        hidden_dim=adapter_hidden_dim,
        dtype=ttnn.bfloat8_b,
    )

    projector = ImageProjector(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=adapter_hidden_dim,
        output_dim=output_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Create random inputs
    torch.manual_seed(42)
    query_torch = torch.randn(1, num_queries, pool_input_dim, dtype=torch.float32)
    kv_torch = torch.randn(1, pool_size, pool_input_dim, dtype=torch.float32)

    # Convert to TTNN
    query_ttnn = ttnn.from_torch(
        query_torch.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kv_ttnn = ttnn.from_torch(
        kv_torch.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Forward through pooling
    pooled = pooling(query_ttnn, kv_ttnn)

    # Forward through projector
    output = projector(pooled)

    # Convert back to torch
    output_torch = ttnn.to_torch(output).squeeze(0).squeeze(0)

    # Check output shape
    expected_shape = (num_queries, output_dim)
    assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    print(f"Vision adapter pipeline output shape: {output_torch.shape}")
    print("Vision adapter integration test passed!")


if __name__ == "__main__":
    # Quick standalone test
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_vision_backbone_projector_integration(device)
        print("All tests passed!")
    finally:
        ttnn.close_device(device)

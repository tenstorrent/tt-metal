# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_tt_vit_backbone(device, reset_seeds, batch_size, sam3_vit_backbone):
    """Test the full ViT backbone (32 blocks) against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        tt_vit_backbone,
        preprocess_vit_backbone_weights,
        move_backbone_params_to_device,
    )

    torch.manual_seed(42)

    # Create input
    pixel_values = torch.randn(batch_size, 3, 1008, 1008)

    # Reference forward
    with torch.no_grad():
        ref_outputs = sam3_vit_backbone(pixel_values)
    ref_feats = ref_outputs[-1]  # (B, 1024, 72, 72)

    # Preprocess weights
    backbone_params = preprocess_vit_backbone_weights(sam3_vit_backbone)
    backbone_params = move_backbone_params_to_device(backbone_params, device)

    # Run ttnn backbone
    tt_outputs = tt_vit_backbone(pixel_values, backbone_params, device)
    tt_feats = tt_outputs[-1]  # (B, 1024, 72, 72)

    # Verify shape
    assert tt_feats.shape == ref_feats.shape, f"Shape mismatch: {tt_feats.shape} vs {ref_feats.shape}"

    # PCC check - 32 layers may accumulate some error
    assert_with_pcc(ref_feats.float(), tt_feats.float(), 0.90)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("num_blocks", [4])
def test_tt_vit_backbone_partial(device, reset_seeds, num_blocks, sam3_vit_backbone):
    """Test first N blocks of the ViT backbone for faster iteration."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        preprocess_vit_block_weights,
        move_block_params_to_device,
        tt_patch_embed,
        get_patch_embed_params,
        tt_vit_block,
    )
    import math

    torch.manual_seed(42)

    pixel_values = torch.randn(1, 3, 1008, 1008)

    # Reference: run patch embed + first N blocks
    with torch.no_grad():
        ref_x = sam3_vit_backbone.patch_embed(pixel_values)  # (1, 72, 72, 1024)
        B, H, W, C = ref_x.shape

        # Add pos embed
        if sam3_vit_backbone.pos_embed is not None:
            from sam3.model.vitdet import get_abs_pos
            ref_x = ref_x + get_abs_pos(
                sam3_vit_backbone.pos_embed,
                sam3_vit_backbone.pretrain_use_cls_token,
                (H, W),
                sam3_vit_backbone.retain_cls_token,
                tiling=sam3_vit_backbone.tile_abs_pos,
            )

        ref_x = sam3_vit_backbone.ln_pre(ref_x)

        for i in range(num_blocks):
            ref_x = sam3_vit_backbone.blocks[i](ref_x)

    # Our implementation: patch embed
    params = get_patch_embed_params(sam3_vit_backbone)
    tt_x = tt_patch_embed(pixel_values, params["weight"], params["bias"], device)
    x = ttnn.to_torch(tt_x).float()

    # Add pos embed
    if sam3_vit_backbone.pos_embed is not None:
        from sam3.model.vitdet import get_abs_pos
        x = x + get_abs_pos(
            sam3_vit_backbone.pos_embed,
            sam3_vit_backbone.pretrain_use_cls_token,
            (H, W),
            sam3_vit_backbone.retain_cls_token,
            tiling=sam3_vit_backbone.tile_abs_pos,
        )

    x = torch.nn.functional.layer_norm(
        x, [C],
        sam3_vit_backbone.ln_pre.weight.data,
        sam3_vit_backbone.ln_pre.bias.data,
        eps=1e-5,
    )

    # Run N blocks
    for i in range(num_blocks):
        block_params = preprocess_vit_block_weights(sam3_vit_backbone.blocks[i])
        block_params = move_block_params_to_device(block_params, device)

        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_x = tt_vit_block(
            tt_x,
            block_params["tt_params"],
            num_heads=16,
            window_size=block_params["window_size"],
            device=device,
        )
        x = ttnn.to_torch(tt_x).float()

    assert_with_pcc(ref_x.float(), x.float(), 0.95)

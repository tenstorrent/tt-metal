# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_position_encoding_shape():
    """Test that position encoding has correct shape."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_position_encoding import (
        generate_position_encoding_sine,
    )

    pos = generate_position_encoding_sine(72, 72, num_pos_feats=256)
    assert pos.shape == (1, 512, 72, 72), f"Unexpected shape: {pos.shape}"


def test_position_encoding_matches_reference(sam3_neck):
    """Test position encoding matches SAM3 reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_position_encoding import (
        generate_position_encoding_sine,
    )

    # The neck has a position_encoding module
    ref_pe = sam3_neck.position_encoding

    # Generate position encoding for a specific size
    H, W = 72, 72

    # Reference
    # PositionEmbeddingSine uses mask-based computation; we test shape and type
    with torch.no_grad():
        # The reference needs a NestedTensor or mask
        mask = torch.zeros(1, H, W, dtype=torch.bool)
        ref_pos = ref_pe(mask)

    # Our implementation
    our_pos = generate_position_encoding_sine(
        H, W,
        num_pos_feats=ref_pe.num_pos_feats,
        temperature=ref_pe.temperature,
        normalize=ref_pe.normalize,
        scale=ref_pe.scale,
    )

    assert our_pos.shape == ref_pos.shape, f"Shape mismatch: {our_pos.shape} vs {ref_pos.shape}"
    assert_with_pcc(ref_pos.float(), our_pos.float(), 0.99)


def test_position_encoding_fpn_levels():
    """Test position encodings for multiple FPN levels."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_position_encoding import (
        generate_position_encodings_for_fpn,
    )

    # SAM3 FPN produces features at these spatial sizes (for 1008 input, patch_size=14)
    # 4x: 72*4=288, 2x: 72*2=144, 1x: 72, 0.5x: 36
    feature_shapes = [(288, 288), (144, 144), (72, 72), (36, 36)]

    pos_encodings = generate_position_encodings_for_fpn(feature_shapes)

    assert len(pos_encodings) == 4
    for i, (pos, (h, w)) in enumerate(zip(pos_encodings, feature_shapes)):
        assert pos.shape == (1, 512, h, w), f"Level {i}: expected (1, 512, {h}, {w}), got {pos.shape}"

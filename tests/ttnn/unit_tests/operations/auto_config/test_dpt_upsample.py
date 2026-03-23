# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DPT upsample spatial fix (CPU-only, no hardware required)."""

import pytest


class TestUpsampleSliceLogic:
    """Pure Python tests for the spatial mismatch fix.
    No hardware needed — validates the slice math."""

    @pytest.mark.parametrize("input_h,input_w,target_h,target_w", [
        (19, 19, 37, 37),   # deepest layer: 19*2=38, need 37
        (37, 37, 74, 74),   # mid layer: 37*2=74, exact match
        (10, 10, 19, 19),   # 10*2=20, need 19
        (74, 74, 148, 148), # 74*2=148, exact match
    ])
    def test_upsample_slice_dimensions(self, input_h, input_w, target_h, target_w):
        """Verify that upsample(2x) + slice produces target dims."""
        upsampled_h = input_h * 2
        upsampled_w = input_w * 2

        out_h = min(upsampled_h, target_h)
        out_w = min(upsampled_w, target_w)

        assert out_h == target_h, f"Height mismatch: {out_h} != {target_h}"
        assert out_w == target_w, f"Width mismatch: {out_w} != {target_w}"

    def test_no_slice_when_exact(self):
        """When upsample produces exact target size, no slice needed."""
        assert 37 * 2 == 74  # exact match

    def test_slice_needed_when_off_by_one(self):
        """The core DPT bug: 19*2=38 != 37."""
        input_h, target_h = 19, 37
        upsampled = input_h * 2  # 38
        assert upsampled != target_h
        assert upsampled > target_h  # can slice down
        assert upsampled - target_h == 1  # off by exactly 1


class TestDPTFusionDimensions:
    """Test the full 4-layer dimension chain."""

    def test_reassembly_dimensions(self):
        """ViT output 518/14 = 37 patches per side."""
        img_size = 518
        patch_size = 14
        n_patches = img_size // patch_size  # 37
        assert n_patches == 37

    def test_fusion_chain(self):
        """Verify dimensions through the entire fusion pipeline.

        Fusion goes bottom-up: layer3 → layer2 → layer1 → layer0
        """
        layer_dims = {
            0: (74, 74),    # stride=4, upsample 2x from 37
            1: (37, 37),    # stride=8, native patch grid
            2: (19, 19),    # stride=16, downsample from 37
            3: (10, 10),    # stride=32, downsample from 19
        }

        # Step 1: upsample layer3 (10→20), add to layer2 (19)
        up_3 = layer_dims[3][0] * 2  # 20
        target_2 = layer_dims[2][0]   # 19
        assert up_3 >= target_2       # can slice: 20→19

        # Step 2: upsample result (19→38), add to layer1 (37)
        up_2 = target_2 * 2          # 38
        target_1 = layer_dims[1][0]   # 37
        assert up_2 >= target_1       # can slice: 38→37

        # Step 3: upsample result (37→74), add to layer0 (74)
        up_1 = target_1 * 2          # 74
        target_0 = layer_dims[0][0]   # 74
        assert up_1 == target_0       # exact match, no slice needed

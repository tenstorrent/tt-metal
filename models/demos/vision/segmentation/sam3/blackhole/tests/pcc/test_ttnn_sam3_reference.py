# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the PyTorch reference SAM3 pipeline."""

import pytest
import torch


class TestSam3ReferenceModel:
    """Verify SAM3 PyTorch model loads and runs correctly."""

    def test_model_loads(self, sam3_reference_model):
        """SAM3 model should load without errors."""
        assert sam3_reference_model is not None
        assert not sam3_reference_model.training

    def test_vit_backbone_structure(self, sam3_vit_backbone):
        """Verify ViT backbone has expected structure."""
        vit = sam3_vit_backbone

        # Check patch embedding
        assert hasattr(vit, "patch_embed")
        assert vit.patch_embed.proj.weight.shape == (1024, 3, 14, 14)

        # Check number of blocks
        assert len(vit.blocks) == 32

        # Check block structure
        block = vit.blocks[0]
        assert hasattr(block, "norm1")
        assert hasattr(block, "attn")
        assert hasattr(block, "norm2")
        assert hasattr(block, "mlp")

        # Check attention dimensions
        assert block.attn.qkv.weight.shape == (3072, 1024)  # 3*1024, 1024
        assert block.attn.num_heads == 16

        # Check MLP dimensions - mlp_ratio=4.625, so hidden=4736
        assert block.mlp.fc1.weight.shape[0] == 4736
        assert block.mlp.fc2.weight.shape[1] == 4736

    def test_vit_backbone_window_vs_global(self, sam3_vit_backbone):
        """Verify correct blocks use window vs global attention."""
        global_blocks = {7, 15, 23, 31}
        for i, block in enumerate(sam3_vit_backbone.blocks):
            if i in global_blocks:
                assert block.window_size == 0, f"Block {i} should use global attention"
            else:
                assert block.window_size == 24, f"Block {i} should use window attention (ws=24)"

    def test_vit_backbone_forward_shapes(self, sam3_vit_backbone):
        """Test ViT backbone produces correct output shapes."""
        torch.manual_seed(42)
        pixel_values = torch.randn(1, 3, 1008, 1008)

        with torch.no_grad():
            outputs = sam3_vit_backbone(pixel_values)

        # ViT.forward returns a list of feature tensors
        # With return_interm_layers=False, returns [feats] where feats is NCHW
        assert isinstance(outputs, list), f"Expected list, got {type(outputs)}"
        assert len(outputs) >= 1, f"Expected at least 1 output, got {len(outputs)}"

        # Output is in NCHW format: (B, 1024, 72, 72)
        feats = outputs[-1]
        assert feats.shape[0] == 1, f"Batch dim mismatch: {feats.shape}"
        assert feats.shape[1] == 1024, f"Channel dim should be 1024, got {feats.shape[1]}"
        assert feats.shape[2] == 72, f"Height should be 72, got {feats.shape[2]}"
        assert feats.shape[3] == 72, f"Width should be 72, got {feats.shape[3]}"

    def test_patch_embed_output(self, sam3_vit_backbone):
        """Test PatchEmbed produces correct shape."""
        torch.manual_seed(42)
        pixel_values = torch.randn(1, 3, 1008, 1008)

        with torch.no_grad():
            patch_output = sam3_vit_backbone.patch_embed(pixel_values)

        # PatchEmbed outputs BHWC: (B, 72, 72, 1024)
        assert patch_output.shape == (1, 72, 72, 1024), f"Unexpected shape: {patch_output.shape}"

    def test_single_block_forward(self, sam3_vit_backbone):
        """Test a single ViT block produces correct output shape."""
        torch.manual_seed(42)
        block = sam3_vit_backbone.blocks[0]

        # Window attention block expects (B, H, W, C)
        hidden_states = torch.randn(1, 72, 72, 1024)

        with torch.no_grad():
            output = block(hidden_states)

        assert output.shape == (1, 72, 72, 1024), f"Unexpected shape: {output.shape}"

    def test_vit_rope_present(self, sam3_vit_backbone):
        """Verify RoPE is configured in attention blocks."""
        block = sam3_vit_backbone.blocks[0]
        assert block.attn.use_rope, "Block should use RoPE"
        assert block.attn.freqs_cis is not None, "RoPE frequencies should be precomputed"

    def test_extract_block_params(self, sam3_vit_backbone):
        """Test parameter extraction from ViT block."""
        from models.demos.vision.segmentation.sam3.common.reference.torch_sam3 import extract_vit_block_params

        params = extract_vit_block_params(sam3_vit_backbone.blocks[0])

        assert "norm1_weight" in params
        assert "attn_qkv_weight" in params
        assert "mlp_fc1_weight" in params
        assert params["attn_qkv_weight"].shape == (3072, 1024)
        assert params["mlp_fc1_weight"].shape == (4736, 1024)
        assert params["window_size"] == 24

    def test_neck_structure(self, sam3_neck):
        """Verify FPN neck has expected structure."""
        assert len(sam3_neck.convs) == 4

    def test_transformer_structure(self, sam3_transformer):
        """Verify transformer encoder/decoder structure."""
        assert sam3_transformer.d_model == 256
        assert hasattr(sam3_transformer, "encoder")
        assert hasattr(sam3_transformer, "decoder")

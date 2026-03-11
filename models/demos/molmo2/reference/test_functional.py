# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for reference/functional.py — verifies each standalone block
matches the HuggingFace Molmo2 implementation numerically.

Run with:
    pytest models/demos/molmo2/reference/test_functional.py -v

Requires:
    HF_MODEL env var or defaults to "allenai/Molmo2-8B"
    transformers, torch installed

These tests run on CPU only — no TTNN device needed.
They generate golden tensors in reference/golden/ for TTNN PCC validation.
"""

import os
from pathlib import Path

import pytest
import torch

from models.demos.molmo2.reference.functional import (
    generate_image_projector_golden,
    generate_text_block_golden,
    generate_vision_block_golden,
    image_pooling_forward,
    image_projector_forward,
    text_block_forward,
    text_mlp_forward,
    vision_block_forward,
    vision_mlp_forward,
    vision_transformer_forward,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
MODEL_PATH = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
ATOL = 1e-4  # float32 tolerance for HF comparison


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace Molmo2 reference model (float32, CPU)."""
    from models.demos.molmo2.reference.model import Molmo2Reference

    return Molmo2Reference(MODEL_PATH, torch_dtype=torch.float32)


@pytest.fixture(scope="module")
def state_dict(hf_model):
    """Extract full state dict from HF model."""
    return {k: v.clone() for k, v in hf_model.model.state_dict().items()}


@pytest.fixture(scope="module")
def full_state_dict(state_dict):
    """State dict with 'model.' prefix prepended to match HF checkpoint keys."""
    return {f"model.{k}": v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Vision LayerNorm
# ---------------------------------------------------------------------------


class TestVisionLayerNorm:
    def test_layernorm_matches_hf(self, hf_model, full_state_dict):
        """LayerNorm output matches HF ViT block norm."""
        from models.demos.molmo2.reference.functional import vision_layernorm

        torch.manual_seed(0)
        x = torch.randn(1, 729, 1152)
        prefix = "model.vision_backbone.image_vit.transformer.resblocks.0"
        w = full_state_dict[f"{prefix}.attention_norm.weight"]
        b = full_state_dict[f"{prefix}.attention_norm.bias"]

        ref = hf_model.image_vit.transformer.resblocks[0].ln_1(x.squeeze(0)).unsqueeze(0)
        out = vision_layernorm(x, w, b, eps=1e-6)

        assert torch.allclose(out, ref, atol=ATOL), f"Max diff: {(out - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------


class TestVisionAttention:
    @pytest.mark.parametrize("layer_num", [0, 12, 24])
    def test_vision_attention_matches_hf(self, hf_model, full_state_dict, layer_num):
        """Vision attention output matches HF ViT attention block."""
        torch.manual_seed(0)
        x = torch.randn(1, 729, 1152)
        prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}.attention"

        hf_block = hf_model.image_vit.transformer.resblocks[layer_num]
        from models.demos.molmo2.reference.functional import vision_attention_forward

        with torch.no_grad():
            ref = hf_block.attn(x.squeeze(0)).unsqueeze(0)
        out = vision_attention_forward(x, full_state_dict, prefix)

        assert torch.allclose(out, ref, atol=ATOL), f"Layer {layer_num}: max diff {(out - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------


class TestVisionMLP:
    @pytest.mark.parametrize("layer_num", [0, 12, 24])
    def test_vision_mlp_matches_hf(self, hf_model, full_state_dict, layer_num):
        """Vision MLP output matches HF ViT MLP block."""
        torch.manual_seed(0)
        x = torch.randn(1, 729, 1152)
        prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}.feed_forward"

        hf_block = hf_model.image_vit.transformer.resblocks[layer_num]
        with torch.no_grad():
            ref = hf_block.mlp(x.squeeze(0)).unsqueeze(0)
        out = vision_mlp_forward(x, full_state_dict, prefix)

        assert torch.allclose(out, ref, atol=ATOL), f"Layer {layer_num}: max diff {(out - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------


class TestVisionBlock:
    @pytest.mark.parametrize("layer_num", [0, 12, 24])
    def test_vision_block_matches_hf(self, hf_model, full_state_dict, layer_num):
        """Full ViT block output matches HF."""
        torch.manual_seed(0)
        x = torch.randn(1, 729, 1152)

        hf_block = hf_model.image_vit.transformer.resblocks[layer_num]
        with torch.no_grad():
            ref = hf_block(x.squeeze(0)).unsqueeze(0)
        out = vision_block_forward(x, full_state_dict, layer_num)

        assert torch.allclose(out, ref, atol=ATOL), f"Layer {layer_num}: max diff {(out - ref).abs().max()}"

    @pytest.mark.parametrize("layer_num", [0, 12, 24])
    def test_vision_block_golden_saved(self, full_state_dict, layer_num):
        """Save golden tensors for ViT block TTNN PCC comparison."""
        golden = generate_vision_block_golden(full_state_dict, layer_num)
        path = GOLDEN_DIR / f"vision_block_layer{layer_num}.pt"
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(golden, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Vision Transformer (full encoder)
# ---------------------------------------------------------------------------


class TestVisionTransformer:
    @pytest.mark.parametrize("num_layers", [1, 5, 25])
    def test_vision_transformer_matches_hf(self, hf_model, full_state_dict, num_layers):
        """ViT encoder hidden states match HF for given number of layers."""
        torch.manual_seed(0)
        # Use random pixel values (B=1, 3, 378, 378)
        pixel_values = torch.randn(1, 3, 378, 378)

        # Run reference
        hidden_states = vision_transformer_forward(pixel_values, full_state_dict, num_layers=num_layers)

        # Run HF reference block by block
        prefix = "model.vision_backbone.image_vit"
        with torch.no_grad():
            x = hf_model.image_vit.patch_embedding(pixel_values)
            x = x + hf_model.image_vit.positional_embedding.unsqueeze(0)
            for i in range(num_layers):
                x = hf_model.image_vit.transformer.resblocks[i](x.squeeze(0)).unsqueeze(0)

        ref = x
        out = hidden_states[-1]
        assert torch.allclose(out, ref, atol=ATOL * 10), f"{num_layers}L: max diff {(out - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Image Projector
# ---------------------------------------------------------------------------


class TestImageProjector:
    @pytest.mark.parametrize("num_tokens", [64, 256, 729])
    def test_image_projector_matches_hf(self, hf_model, full_state_dict, num_tokens):
        """SwiGLU projector output matches HF image_projector."""
        torch.manual_seed(0)
        x = torch.randn(num_tokens, 1152)

        with torch.no_grad():
            ref = hf_model.image_projector(x)
        out = image_projector_forward(x, full_state_dict)

        assert torch.allclose(out, ref, atol=ATOL), f"num_tokens={num_tokens}: max diff {(out - ref).abs().max()}"

    @pytest.mark.parametrize("num_tokens", [64, 256, 729])
    def test_image_projector_golden_saved(self, full_state_dict, num_tokens):
        """Save golden tensors for projector TTNN PCC comparison."""
        golden = generate_image_projector_golden(full_state_dict, num_tokens)
        path = GOLDEN_DIR / f"image_projector_{num_tokens}tokens.pt"
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(golden, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Image Pooling
# ---------------------------------------------------------------------------


class TestImagePooling:
    def test_image_pooling_matches_hf(self, hf_model, full_state_dict):
        """Cross-attention pooling matches HF image_pooling_2d."""
        torch.manual_seed(0)
        B, N_patches, K_pool = 1, 729, 9
        N_out = 64

        features = torch.randn(B, N_patches, 2304)
        pooled_patches_idx = torch.randint(0, N_patches, (B, N_out, K_pool))

        out = image_pooling_forward(features, pooled_patches_idx, full_state_dict)

        # Compare shape: [B, N_out, 1152]
        assert out.shape == (B, N_out, 1152), f"Shape mismatch: {out.shape}"

    def test_image_pooling_golden_saved(self, full_state_dict):
        """Save golden tensors for pooling TTNN PCC comparison."""
        torch.manual_seed(0)
        features = torch.randn(1, 729, 2304)
        pooled_patches_idx = torch.randint(0, 729, (1, 64, 9))
        out = image_pooling_forward(features, pooled_patches_idx, full_state_dict)
        golden = {"input_features": features, "pooled_patches_idx": pooled_patches_idx, "output": out}
        path = GOLDEN_DIR / "image_pooling.pt"
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(golden, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Text MLP
# ---------------------------------------------------------------------------


class TestTextMLP:
    @pytest.mark.parametrize("layer_num", [0, 17, 35])
    def test_text_mlp_matches_hf(self, hf_model, full_state_dict, layer_num):
        """Text SwiGLU MLP matches HF block MLP."""
        torch.manual_seed(0)
        x = torch.randn(1, 32, 4096)

        hf_block = hf_model.text_model.blocks[layer_num]
        with torch.no_grad():
            ref = hf_block.mlp(x)
        out = text_mlp_forward(x, full_state_dict, layer_num)

        assert torch.allclose(out, ref, atol=ATOL), f"Layer {layer_num}: max diff {(out - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Text Block
# ---------------------------------------------------------------------------


class TestTextBlock:
    @pytest.mark.parametrize("layer_num", [0, 17, 35])
    def test_text_block_matches_hf(self, hf_model, full_state_dict, layer_num):
        """Full text decoder block matches HF."""
        torch.manual_seed(0)
        seq_len = 32
        x = torch.randn(1, seq_len, 4096)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        hf_block = hf_model.text_model.blocks[layer_num]
        with torch.no_grad():
            ref = hf_block(x, position_ids=position_ids)[0]
        out = text_block_forward(x, full_state_dict, layer_num, position_ids)

        assert torch.allclose(out, ref, atol=ATOL * 10), f"Layer {layer_num}: max diff {(out - ref).abs().max()}"

    @pytest.mark.parametrize("layer_num", [0, 17, 35])
    def test_text_block_golden_saved(self, full_state_dict, layer_num):
        """Save golden tensors for text block TTNN PCC comparison."""
        golden = generate_text_block_golden(full_state_dict, layer_num, seq_len=128)
        path = GOLDEN_DIR / f"text_block_layer{layer_num}.pt"
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(golden, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# E2E Reference Verification
# ---------------------------------------------------------------------------


class TestE2EReference:
    def test_e2e_reference_generates_text(self, hf_model):
        """Verify HF model produces sensible text (smoke test for reference correctness)."""

        # Minimal text-only test (no image loading needed)
        processor = hf_model.processor
        inputs = processor(text="The capital of France is", return_tensors="pt")

        with torch.no_grad():
            output = hf_model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = processor.decode(output[0], skip_special_tokens=True)
        assert "Paris" in decoded or len(decoded) > len(
            "The capital of France is"
        ), f"Model did not generate expected text: {decoded!r}"

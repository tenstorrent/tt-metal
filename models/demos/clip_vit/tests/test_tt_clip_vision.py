import pytest
import torch
from transformers import CLIPModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.clip_vit.tt.tt_clip_vision import (
    TtCLIPVisionAttention,
    TtCLIPVisionEmbeddings,
    TtCLIPVisionEncoderLayer,
    TtCLIPVisionMLP,
    TtCLIPVisionModel,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 50, 0.99, 0), (32, 50, 0.99, 0)])
def test_clip_vision_mlp(batch_size, seq_len, pcc, layer_idx):
    """
    seq_len for vision is num_patches + 1 = (224/32)^2 + 1 = 49 + 1 = 50 for ViT-B/32
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config.vision_config

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        torch_mlp = torch_model.vision_model.encoder.layers[layer_idx].mlp
        ttnn_mlp = TtCLIPVisionMLP(config, torch_mlp, device)

        torch_input = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        with torch.no_grad():
            torch_output = torch_mlp(torch_input)

        ttnn_output = ttnn_mlp(ttnn_input)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 50, 0.97, 0), (4, 50, 0.97, 0)])
def test_clip_vision_attention(batch_size, seq_len, pcc, layer_idx):
    """
    Vision attention does not use causal masking.
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.vision_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_attention = torch_model.vision_model.encoder.layers[layer_idx].self_attn

        ttnn_attention = TtCLIPVisionAttention(config, torch_attention, device)

        torch_input = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        with torch.no_grad():
            torch_output = torch_attention(torch_input)[0]

        ttnn_output = ttnn_attention(ttnn_input)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 50, 0.97, 0), (4, 50, 0.97, 0)])
def test_clip_vision_encoder(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.vision_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_encoder = torch_model.vision_model.encoder.layers[layer_idx]

        ttnn_encoder = TtCLIPVisionEncoderLayer(config, torch_encoder, device)

        torch_input = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        with torch.no_grad():
            torch_output = torch_encoder(torch_input, attention_mask=None, causal_attention_mask=None)[0]

        ttnn_output = ttnn_encoder(ttnn_input)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, pcc", [(1, 0.97), (4, 0.97)])
def test_clip_vision_embeddings(batch_size, pcc):
    """
    Input is pixel values of shape (batch_size, num_channels, height, width).
    For ViT-B/32: image_size=224, patch_size=32, so num_patches = (224/32)^2 = 49
    Output shape: (batch_size, num_patches + 1, hidden_size) = (batch_size, 50, 768)
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.vision_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_embedding = torch_model.vision_model.embeddings

        ttnn_embedding = TtCLIPVisionEmbeddings(config, torch_embedding, device)

        torch_pixel_values = torch.randn(
            batch_size, config.num_channels, config.image_size, config.image_size, dtype=torch.bfloat16
        )
        ttnn_pixel_values = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )

        ttnn_pixel_values = ttnn.permute(ttnn_pixel_values, [0, 2, 3, 1])

        with torch.no_grad():
            torch_output = torch_embedding(torch_pixel_values).to(torch.bfloat16)

        ttnn_output = ttnn_embedding(ttnn_pixel_values)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, pcc", [(1, 0.95), (4, 0.95)])
def test_clip_vision_model(batch_size, pcc):
    """
    Test the full CLIP Vision Transformer (equivalent to vision_model in HuggingFace).
    Input: pixel_values of shape (batch_size, num_channels, height, width)
    Output: last_hidden_state of shape (batch_size, num_patches + 1, hidden_size)
            pooler_output of shape (batch_size, hidden_size)
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.vision_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_vision_model = torch_model.vision_model

        ttnn_model = TtCLIPVisionModel(config, torch_vision_model, device)

        torch_pixel_values = torch.randn(
            batch_size, config.num_channels, config.image_size, config.image_size, dtype=torch.bfloat16
        )
        ttnn_pixel_values = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )

        ttnn_pixel_values = ttnn.permute(ttnn_pixel_values, [0, 2, 3, 1])

        with torch.no_grad():
            torch_outputs = torch_vision_model(pixel_values=torch_pixel_values)
            torch_last_hidden_state = torch_outputs.last_hidden_state.to(torch.bfloat16)
            torch_pooler_output = torch_outputs.pooler_output.to(torch.bfloat16)

        ttnn_last_hidden_state, ttnn_pooler_output = ttnn_model(pixel_values=ttnn_pixel_values)

        ttnn_last_hidden_torch = ttnn.to_torch(ttnn_last_hidden_state)
        ttnn_pooler_torch = ttnn.to_torch(ttnn_pooler_output)

        # Check last hidden state
        passed_lhs, pcc_lhs = comp_pcc(torch_last_hidden_state, ttnn_last_hidden_torch, pcc=pcc)
        assert_with_pcc(torch_last_hidden_state, ttnn_last_hidden_torch, pcc=pcc)

        # Check pooler output
        passed_pool, pcc_pool = comp_pcc(torch_pooler_output, ttnn_pooler_torch, pcc=pcc)
        assert_with_pcc(torch_pooler_output, ttnn_pooler_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)

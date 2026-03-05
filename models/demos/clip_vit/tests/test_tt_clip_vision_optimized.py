import torch

import ttnn
from models.demos.clip_vit.tt.tt_clip_vision_optimized import (
    TtCLIPVisionAttention,
    TtCLIPVisionEmbeddings,
    TtCLIPVisionEncoderLayer,
    TtCLIPVisionMLP,
    TtCLIPVisionModel,
)


def test_clip_vision_embeddings(torch_model, vision_config, pixel_values, device):
    torch_embedding = torch_model.embeddings
    embedding_config = vision_config

    ttnn_embedding = TtCLIPVisionEmbeddings(vision_config, torch_embedding, device)

    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_embedding(pixel_values)

    ttnn_output = ttnn_embedding(ttnn_pixel_values)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_vision_mlp(torch_model, vision_config, pixel_values, device):
    torch_embedding = torch_model.embeddings
    torch_mlp = torch_model.encoder.layers[0].mlp
    ttnn_mlp = TtCLIPVisionMLP(vision_config, torch_mlp, device)

    with torch.no_grad():
        pixel_values = torch_embedding(pixel_values)

    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(pixel_values)

    ttnn_output = ttnn_mlp(ttnn_pixel_values)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_vision_attention(torch_model, vision_config, pixel_values, device):
    torch_embedding = torch_model.embeddings
    torch_attention = torch_model.encoder.layers[0].self_attn
    ttnn_attention = TtCLIPVisionAttention(vision_config, torch_attention, device)
    with torch.no_grad():
        pixel_values = torch_embedding(pixel_values)

    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_attention(pixel_values)[0]

    ttnn_output = ttnn_attention(ttnn_pixel_values)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_vision_encoder_layer(torch_model, vision_config, pixel_values, device):
    torch_embedding = torch_model.embeddings
    torch_encoder = torch_model.encoder.layers[0]
    with torch.no_grad():
        pixel_embeddings = torch_embedding(pixel_values)

    ttnn_encoder = TtCLIPVisionEncoderLayer(vision_config, torch_encoder, device)

    ttnn_pixel_embeddings = ttnn.from_torch(
        pixel_embeddings, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT
    )

    with torch.no_grad():
        torch_output = torch_encoder(pixel_embeddings, attention_mask=None, causal_attention_mask=None)[0]

    ttnn_output = ttnn_encoder(ttnn_pixel_embeddings)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_vision_encoder(torch_vision_encoder, vision_config, pixel_values, device):
    ttnn_vision_encoder = TtCLIPVisionModel(vision_config, torch_vision_encoder, device)

    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_outputs = torch_vision_encoder(pixel_values=pixel_values)
        torch_last_hidden_state = torch_outputs.last_hidden_state
        torch_pooler_output = torch_outputs.pooler_output

    ttnn_pooler_output = ttnn_vision_encoder(pixel_values=ttnn_pixel_values)

    ttnn_pooler_torch = ttnn.to_torch(ttnn_pooler_output)

    return torch_pooler_output, ttnn_pooler_torch

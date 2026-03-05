import torch

import ttnn
from models.demos.clip_vit.tt.tt_clip_text import (
    TtCLIPAttention,
    TtCLIPEncoderLayer,
    TtCLIPMLP,
    TtCLIPTextEmbeddings,
    TtCLIPTextModel,
)
from models.demos.clip_vit.tt.tt_clip_text_optimized import (
    TtCLIPAttentionOptimized,
    TtCLIPEncoderLayerOptimized,
    TtCLIPMLPOptimized,
    TtCLIPTextEmbeddingsOptimized,
    TtCLIPTextModelOptimized,
)


def test_clip_text_embeddings(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings

    ttnn_embedding = TtCLIPTextEmbeddings(text_config, torch_embedding, device)

    batch_size, seq_len = input_ids.shape
    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with torch.no_grad():
        torch_output = torch_embedding(input_ids=input_ids, position_ids=torch_position_ids).to(torch.bfloat16)

    ttnn_output = ttnn_embedding(ttnn_input_ids, ttnn_position_ids)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_mlp(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_mlp = torch_model.encoder.layers[0].mlp

    ttnn_mlp = TtCLIPMLP(text_config, torch_mlp, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states)

    ttnn_output = ttnn_mlp(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_attention(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_attention = torch_model.encoder.layers[0].self_attn

    ttnn_attention = TtCLIPAttention(text_config, torch_attention, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_attention(hidden_states)[0]

    ttnn_output = ttnn_attention(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_encoder_layer(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_encoder_layer = torch_model.encoder.layers[0]

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_encoder_layer = TtCLIPEncoderLayer(text_config, torch_encoder_layer, device)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_encoder_layer(hidden_states, attention_mask=None, causal_attention_mask=None)[0]

    ttnn_output = ttnn_encoder_layer(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_encoder(torch_text_model, text_config, input_ids, device):
    ttnn_text_model = TtCLIPTextModel(text_config, torch_text_model, device)

    batch_size, seq_len = input_ids.shape
    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with torch.no_grad():
        torch_outputs = torch_text_model(input_ids=input_ids, position_ids=torch_position_ids)
        torch_last_hidden_state = torch_outputs.last_hidden_state.to(torch.bfloat16)

    ttnn_output = ttnn_text_model(input_ids=ttnn_input_ids, position_ids=ttnn_position_ids)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_last_hidden_state, ttnn_output_torch


def test_clip_text_embeddings_optimized(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings

    ttnn_embedding = TtCLIPTextEmbeddingsOptimized(text_config, torch_embedding, device)

    batch_size, seq_len = input_ids.shape
    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with torch.no_grad():
        torch_output = torch_embedding(input_ids=input_ids, position_ids=torch_position_ids).to(torch.bfloat16)

    ttnn_output = ttnn_embedding(ttnn_input_ids, ttnn_position_ids)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_mlp_optimized(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_mlp = torch_model.encoder.layers[0].mlp

    ttnn_mlp = TtCLIPMLPOptimized(text_config, torch_mlp, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states.float()).to(torch.bfloat16)

    ttnn_output = ttnn_mlp(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_attention_optimized(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_attention = torch_model.encoder.layers[0].self_attn

    ttnn_attention = TtCLIPAttentionOptimized(text_config, torch_attention, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_attention(hidden_states.float())[0].to(torch.bfloat16)

    ttnn_output = ttnn_attention(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_encoder_layer_optimized(torch_model, text_config, input_ids, device):
    torch_embedding = torch_model.embeddings
    torch_encoder_layer = torch_model.encoder.layers[0]

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_encoder_layer = TtCLIPEncoderLayerOptimized(text_config, torch_encoder_layer, device)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_encoder_layer(hidden_states.float(), attention_mask=None, causal_attention_mask=None)[
            0
        ].to(torch.bfloat16)

    ttnn_output = ttnn_encoder_layer(ttnn_hidden_states)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_output, ttnn_output_torch


def test_clip_text_encoder_optimized(torch_text_model, text_config, input_ids, device):
    ttnn_text_model = TtCLIPTextModelOptimized(text_config, torch_text_model, device)

    batch_size, seq_len = input_ids.shape
    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with torch.no_grad():
        torch_outputs = torch_text_model(input_ids=input_ids, position_ids=torch_position_ids)
        torch_last_hidden_state = torch_outputs.last_hidden_state.to(torch.bfloat16)

    ttnn_output = ttnn_text_model(input_ids=ttnn_input_ids, position_ids=ttnn_position_ids)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    return torch_last_hidden_state, ttnn_output_torch

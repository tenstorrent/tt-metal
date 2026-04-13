# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import copy

import torch

import ttnn
from models.demos.clip_vit.tests.pcc.conftest import PCC_BATCH_SIZE, opt_pcc_check, pcc_check

# ---- Base model imports ----
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel

# ---- Optimized model imports ----
from models.demos.clip_vit.tt.tt_clip_model_optimized import TtCLIPModelOptimized
from models.demos.clip_vit.tt.tt_clip_text import (
    TtCLIPAttention,
    TtCLIPEncoderLayer,
    TtCLIPMLP,
    TtCLIPTextEmbeddings,
    TtCLIPTextModel,
)
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextAttention as TtCLIPTextAttentionOpt
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextEmbeddings as TtCLIPTextEmbeddingsOpt
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextEncoderLayer as TtCLIPTextEncoderLayerOpt
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextMLP as TtCLIPTextMLPOpt
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextModel as TtCLIPTextModelOpt
from models.demos.clip_vit.tt.tt_clip_text_optimized import build_text_encoder_configs
from models.demos.clip_vit.tt.tt_clip_vision import (
    TtCLIPVisionAttention,
    TtCLIPVisionEmbeddings,
    TtCLIPVisionEncoderLayer,
    TtCLIPVisionMLP,
    TtCLIPVisionModel,
)
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionAttention as TtCLIPVisionAttentionOpt
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionEmbeddings as TtCLIPVisionEmbeddingsOpt
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionEncoderLayer as TtCLIPVisionEncoderLayerOpt
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionMLP as TtCLIPVisionMLPOpt
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionModel as TtCLIPVisionModelOpt
from models.demos.clip_vit.tt.tt_clip_vision_optimized import build_vision_encoder_configs

# ==========================================================================
#  Base Vision PCC Tests
# ==========================================================================


def test_pcc_vision_embeddings(torch_model, inputs, device):
    config = torch_model.config.vision_config
    torch_embedding = torch_model.vision_model.embeddings
    ttnn_embedding = TtCLIPVisionEmbeddings(config, torch_embedding, device)

    ttnn_pixel_values = ttnn.from_torch(
        inputs["pixel_values"], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    with torch.no_grad():
        torch_output = torch_embedding(inputs["pixel_values"])

    ttnn_output = ttnn_embedding(ttnn_pixel_values)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_vision_mlp(torch_model, inputs, device):
    config = torch_model.config.vision_config
    torch_embedding = torch_model.vision_model.embeddings
    torch_mlp = torch_model.vision_model.encoder.layers[0].mlp
    ttnn_mlp = TtCLIPVisionMLP(config, torch_mlp, device)

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states)

    ttnn_output = ttnn_mlp(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_vision_attention(torch_model, inputs, device):
    config = torch_model.config.vision_config
    torch_embedding = torch_model.vision_model.embeddings
    torch_attn = torch_model.vision_model.encoder.layers[0].self_attn
    ttnn_attn = TtCLIPVisionAttention(config, torch_attn, device)

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_attn(hidden_states)[0]

    ttnn_output = ttnn_attn(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_vision_encoder_layer(torch_model, inputs, device):
    config = torch_model.config.vision_config
    torch_embedding = torch_model.vision_model.embeddings
    torch_layer = torch_model.vision_model.encoder.layers[0]
    ttnn_layer = TtCLIPVisionEncoderLayer(config, torch_layer, device)

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_layer(hidden_states, attention_mask=None, causal_attention_mask=None)[0]

    ttnn_output = ttnn_layer(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_vision_model(torch_model, inputs, device):
    config = torch_model.config.vision_config
    vision_model = torch_model.vision_model
    ttnn_model = TtCLIPVisionModel(config, vision_model, device)

    ttnn_pixel_values = ttnn.from_torch(
        inputs["pixel_values"], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    with torch.no_grad():
        torch_output = vision_model(pixel_values=inputs["pixel_values"]).pooler_output

    ttnn_output = ttnn_model(pixel_values=ttnn_pixel_values)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


# ==========================================================================
#  Base Text PCC Tests
# ==========================================================================


def test_pcc_text_embeddings(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    torch_embedding = torch_model.text_model.embeddings
    ttnn_embedding = TtCLIPTextEmbeddings(config, torch_embedding, device)

    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        torch_output = torch_embedding(input_ids=input_ids, position_ids=position_ids).to(torch.bfloat16)

    ttnn_output = ttnn_embedding(ttnn_input_ids, ttnn_position_ids)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_text_mlp(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    torch_embedding = torch_model.text_model.embeddings
    torch_mlp = torch_model.text_model.encoder.layers[0].mlp
    ttnn_mlp = TtCLIPMLP(config, torch_mlp, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states.float()).to(torch.bfloat16)

    ttnn_output = ttnn_mlp(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_text_attention(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    torch_embedding = torch_model.text_model.embeddings
    torch_attn = torch_model.text_model.encoder.layers[0].self_attn
    ttnn_attn = TtCLIPAttention(config, torch_attn, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    seq_len = hidden_states.shape[1]
    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        torch_output = torch_attn(hidden_states.float(), causal_attention_mask=causal_mask)[0].to(torch.bfloat16)

    ttnn_output = ttnn_attn(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_text_encoder_layer(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    torch_embedding = torch_model.text_model.embeddings
    torch_layer = torch_model.text_model.encoder.layers[0]
    ttnn_layer = TtCLIPEncoderLayer(config, torch_layer, device)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    seq_len = hidden_states.shape[1]
    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        torch_output = torch_layer(hidden_states.float(), attention_mask=None, causal_attention_mask=causal_mask)[0].to(
            torch.bfloat16
        )

    ttnn_output = ttnn_layer(ttnn_hidden)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_text_model(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    text_model = torch_model.text_model
    ttnn_model = TtCLIPTextModel(config, text_model, device)

    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        torch_output = text_model(input_ids=input_ids, position_ids=position_ids).last_hidden_state.to(torch.bfloat16)

    ttnn_output = ttnn_model(input_ids=ttnn_input_ids, position_ids=ttnn_position_ids)
    pcc_check(torch_output, ttnn.to_torch(ttnn_output))


# ==========================================================================
#  Base Full Model PCC Test
# ==========================================================================


def test_pcc_full_model(torch_model, inputs, device):
    # Deep-copy so the .to(bfloat16) cast doesn't mutate the session fixture.
    torch_model_bf16 = copy.deepcopy(torch_model).to(torch.bfloat16)
    config = torch_model_bf16.config

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]

    ttnn_model = TtCLIPModel(config, torch_model_bf16, device)

    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_outputs = torch_model_bf16(input_ids=input_ids, pixel_values=pixel_values)

    logits_per_image, logits_per_text = ttnn_model(
        input_ids=ttnn_input_ids,
        pixel_values=ttnn_pixel_values,
        position_ids=ttnn_position_ids,
    )

    pcc_check(torch_outputs.logits_per_image.to(torch.bfloat16), ttnn.to_torch(logits_per_image))
    pcc_check(torch_outputs.logits_per_text.to(torch.bfloat16), ttnn.to_torch(logits_per_text))


# ==========================================================================
#  Optimized Vision PCC Tests
# ==========================================================================


def test_pcc_opt_vision_embeddings(torch_model, inputs, device):
    config = torch_model.config.vision_config
    torch_embedding = torch_model.vision_model.embeddings
    ttnn_embedding = TtCLIPVisionEmbeddingsOpt(config, torch_embedding, device, dtype=ttnn.bfloat16)

    ttnn_pixel_values = ttnn.from_torch(
        inputs["pixel_values"], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    with torch.no_grad():
        torch_output = torch_embedding(inputs["pixel_values"])

    ttnn_output = ttnn_embedding(ttnn_pixel_values)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_vision_mlp(torch_model, inputs, device):
    config = torch_model.config.vision_config
    memory_configs, program_configs = build_vision_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.vision_model.embeddings
    torch_mlp = torch_model.vision_model.encoder.layers[0].mlp
    ttnn_mlp = TtCLIPVisionMLPOpt(
        config,
        torch_mlp,
        device,
        memory_configs,
        program_configs,
        dtype=ttnn.bfloat16,
    )

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    # Flatten to 2D [batch*seq, hidden] for sharded MLP
    batch, seq, hidden = hidden_states.shape
    hidden_states_2d = hidden_states.reshape(batch * seq, hidden)

    ttnn_hidden = ttnn.from_torch(hidden_states_2d, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states)

    ttnn_output = ttnn_mlp(ttnn_hidden)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = ttnn_output_torch[: batch * seq, :]  # trim any tile padding
    ttnn_output_torch = ttnn_output_torch.reshape(batch, seq, -1)
    opt_pcc_check(torch_output, ttnn_output_torch)


def test_pcc_opt_vision_attention(torch_model, inputs, device):
    config = torch_model.config.vision_config
    memory_configs, program_configs = build_vision_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.vision_model.embeddings
    torch_attn = torch_model.vision_model.encoder.layers[0].self_attn
    ttnn_attn = TtCLIPVisionAttentionOpt(
        config,
        torch_attn,
        device,
        memory_configs,
        program_configs,
        dtype=ttnn.bfloat16,
    )

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_attn(hidden_states)[0]

    ttnn_output = ttnn_attn(ttnn_hidden)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_vision_encoder_layer(torch_model, inputs, device):
    config = torch_model.config.vision_config
    memory_configs, program_configs = build_vision_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.vision_model.embeddings
    torch_layer = torch_model.vision_model.encoder.layers[0]
    ttnn_layer = TtCLIPVisionEncoderLayerOpt(
        config,
        torch_layer,
        device,
        memory_configs,
        program_configs,
        dtype=ttnn.bfloat16,
    )

    with torch.no_grad():
        hidden_states = torch_embedding(inputs["pixel_values"])

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden = ttnn.to_memory_config(ttnn_hidden, memory_configs["hidden"])

    with torch.no_grad():
        torch_output = torch_layer(hidden_states, attention_mask=None, causal_attention_mask=None)[0]

    ttnn_output = ttnn_layer(ttnn_hidden)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_vision_model(torch_model, inputs, device):
    config = torch_model.config.vision_config
    vision_model = torch_model.vision_model
    memory_configs, program_configs = build_vision_encoder_configs(config, device, PCC_BATCH_SIZE)
    ttnn_model = TtCLIPVisionModelOpt(
        config,
        vision_model,
        device,
        memory_configs,
        program_configs,
        dtype=ttnn.bfloat16,
    )

    ttnn_pixel_values = ttnn.from_torch(
        inputs["pixel_values"], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    with torch.no_grad():
        torch_output = vision_model(pixel_values=inputs["pixel_values"]).pooler_output

    ttnn_output = ttnn_model(pixel_values=ttnn_pixel_values)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


# ==========================================================================
#  Optimized Text PCC Tests
# ==========================================================================


def test_pcc_opt_text_embeddings(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]

    torch_embedding = torch_model.text_model.embeddings
    ttnn_embedding = TtCLIPTextEmbeddingsOpt(config, torch_embedding, device, dtype=ttnn.bfloat16)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_embedding(input_ids=input_ids).to(torch.bfloat16)

    ttnn_output = ttnn_embedding(ttnn_input_ids)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_text_mlp(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]
    text_mem, text_prog = build_text_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.text_model.embeddings
    torch_mlp = torch_model.text_model.encoder.layers[0].mlp
    ttnn_mlp = TtCLIPTextMLPOpt(config, torch_mlp, device, text_mem, text_prog, dtype=ttnn.bfloat16)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_mlp(hidden_states.float()).to(torch.bfloat16)

    ttnn_output = ttnn_mlp(ttnn_hidden)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_text_attention(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]
    text_mem, text_prog = build_text_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.text_model.embeddings
    torch_attn = torch_model.text_model.encoder.layers[0].self_attn
    ttnn_attn = TtCLIPTextAttentionOpt(config, torch_attn, device, text_mem, text_prog, dtype=ttnn.bfloat16)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Build causal mask to match is_causal=True in ttnn attention
    seq_len = hidden_states.shape[1]
    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        torch_output = torch_attn(hidden_states.float(), causal_attention_mask=causal_mask)[0].to(torch.bfloat16)

    ttnn_output = ttnn_attn(ttnn_hidden)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_text_encoder_layer(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]
    text_mem, text_prog = build_text_encoder_configs(config, device, PCC_BATCH_SIZE)

    torch_embedding = torch_model.text_model.embeddings
    torch_layer = torch_model.text_model.encoder.layers[0]
    ttnn_layer = TtCLIPTextEncoderLayerOpt(config, torch_layer, device, text_mem, text_prog, dtype=ttnn.bfloat16)

    with torch.no_grad():
        hidden_states = torch_embedding(input_ids).to(torch.bfloat16)

    ttnn_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden = ttnn.to_memory_config(ttnn_hidden, text_mem["hidden"])

    # Build causal mask to match is_causal=True in ttnn attention
    seq_len = hidden_states.shape[1]
    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        torch_output = torch_layer(hidden_states.float(), attention_mask=None, causal_attention_mask=causal_mask)[0].to(
            torch.bfloat16
        )

    ttnn_output = ttnn_layer(ttnn_hidden)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


def test_pcc_opt_text_model(torch_model, inputs, device):
    config = torch_model.config.text_config
    input_ids = inputs["input_ids"]
    text_mem, text_prog = build_text_encoder_configs(config, device, PCC_BATCH_SIZE)

    text_model = torch_model.text_model
    ttnn_model = TtCLIPTextModelOpt(config, text_model, device, text_mem, text_prog, dtype=ttnn.bfloat16)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = text_model(input_ids=input_ids).last_hidden_state.to(torch.bfloat16)

    ttnn_output = ttnn_model(input_ids=ttnn_input_ids)
    opt_pcc_check(torch_output, ttnn.to_torch(ttnn_output))


# ==========================================================================
#  Optimized Full Model PCC Test
# ==========================================================================


def test_pcc_opt_full_model(torch_model, inputs, device):
    # Deep-copy so the .to(bfloat16) cast doesn't mutate the session fixture.
    torch_model_bf16 = copy.deepcopy(torch_model).to(torch.bfloat16)
    config = torch_model_bf16.config

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]

    ttnn_model = TtCLIPModelOptimized(
        config,
        torch_model_bf16,
        device,
        vision_batch=PCC_BATCH_SIZE,
        text_batch=PCC_BATCH_SIZE,
        dtype=ttnn.bfloat16,
    )

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        torch_outputs = torch_model_bf16(input_ids=input_ids, pixel_values=pixel_values)

    logits_per_image, logits_per_text = ttnn_model(
        input_ids=ttnn_input_ids,
        pixel_values=ttnn_pixel_values,
    )

    opt_pcc_check(torch_outputs.logits_per_image.to(torch.bfloat16), ttnn.to_torch(logits_per_image))
    opt_pcc_check(torch_outputs.logits_per_text.to(torch.bfloat16), ttnn.to_torch(logits_per_text))

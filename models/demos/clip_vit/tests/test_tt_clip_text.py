import pytest
import torch
from transformers import CLIPModel

import ttnn
from models.common.utility_functions import comp_pcc

# Text model components to test
from models.demos.clip_vit.tt.tt_clip_text import (
    TtCLIPAttention,
    TtCLIPEncoderLayer,
    TtCLIPMLP,
    TtCLIPTextEmbeddings,
    TtCLIPTextModel,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.99, 0), (32, 77, 0.99, 0)])
def test_clip_text_mlp(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config.text_config

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        torch_mlp = torch_model.text_model.encoder.layers[layer_idx].mlp
        ttnn_mlp = TtCLIPMLP(config, torch_mlp, device)

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


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.97, 0), (32, 77, 0.97, 0)])
def test_clip_text_attention(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.text_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_attention = torch_model.text_model.encoder.layers[layer_idx].self_attn

        ttnn_attention = TtCLIPAttention(config, torch_attention, device)

        torch_input = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        """
        torch_attention_mask = torch.ones(batch_size, seq_len)
        ttnn_attention_mask = ttnn.from_torch(
            torch_attention_mask,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT
        )
        """

        with torch.no_grad():
            torch_output = torch_attention(torch_input)[0]

        ttnn_output = ttnn_attention(ttnn_input)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.97, 0), (32, 77, 0.97, 0)])
def test_clip_text_encoder(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.text_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_encoder = torch_model.text_model.encoder.layers[layer_idx]

        ttnn_encoder = TtCLIPEncoderLayer(config, torch_encoder, device)

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


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.97, 0), (32, 77, 0.97, 0)])
def test_clip_text_embeddings(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.text_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_embedding = torch_model.text_model.embeddings

        ttnn_embedding = TtCLIPTextEmbeddings(config, torch_embedding, device)

        torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

        torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        ttnn_position_ids = ttnn.from_torch(
            torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        with torch.no_grad():
            torch_output = torch_embedding(input_ids=torch_input_ids, position_ids=torch_position_ids).to(
                torch.bfloat16
            )

        ttnn_output = ttnn_embedding(ttnn_input_ids, ttnn_position_ids)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc, layer_idx", [(1, 77, 0.97, 0), (32, 77, 0.97, 0)])
def test_clip_text_model(batch_size, seq_len, pcc, layer_idx):
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()
    config = torch_model.config.text_config
    device = None

    try:
        device = ttnn.open_device(device_id=0)

        torch_model = torch_model.text_model

        ttnn_model = TtCLIPTextModel(config, torch_model, device)

        torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_position_ids = ttnn.from_torch(
            torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        with torch.no_grad():
            output = torch_model(input_ids=torch_input_ids, position_ids=torch_position_ids)
            torch_output = output.last_hidden_state.to(torch.bfloat16)

        ttnn_output = ttnn_model(input_ids=ttnn_input_ids, position_ids=ttnn_position_ids)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=pcc)
        assert_with_pcc(torch_output, ttnn_output_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)

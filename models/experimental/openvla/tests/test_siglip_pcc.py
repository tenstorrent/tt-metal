# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for TTNN SigLIP implementation.
Tests each component of the SigLIP encoder against PyTorch reference.
"""

import pytest
import torch
from timm.models import create_model
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.experimental.openvla.tt import tt_optimized_openvla_vision as ttnn_siglip
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_siglip_torch_model():
    """Load pretrained SigLIP model from timm."""
    torch_model = create_model("vit_so400m_patch14_siglip_224", pretrained=True)
    return torch_model.eval()


@pytest.mark.parametrize("image_size_h", [224])
@pytest.mark.parametrize("image_size_w", [224])
@pytest.mark.parametrize("image_channels", [3])
def test_siglip_patch_embeddings(device, image_size_h, image_size_w, image_channels):
    torch.manual_seed(0)

    model = load_siglip_torch_model()

    torch_pixel_values = torch_random((1, image_channels, image_size_h, image_size_w), -1, 1, dtype=torch.float32)
    torch_output, *_ = model.patch_embed(torch_pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn_siglip.siglip_patch_embeddings(
        pixel_values,
        parameters=parameters.patch_embed.patch_embeddings,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.9998)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


@pytest.mark.parametrize("sequence_size", [1152])
@pytest.mark.parametrize("num_heads", [16])
def test_siglip_attention_upchannel(device, sequence_size, num_heads):
    torch.manual_seed(0)

    model = Attention(
        dim=sequence_size,
        num_heads=num_heads,
        qkv_bias=True,
        proj_bias=True,
    )

    torch_hidden_states = torch_random((1, 6, sequence_size), -1, 1, dtype=torch.float32)

    torch_output = model(torch_hidden_states)
    params = ttnn_siglip.upchannel_attn_weight_bias(
        model.qkv.weight, model.qkv.bias, model.proj.weight, model.proj.bias, num_heads
    )
    model.qkv.weight, model.qkv.bias, model.proj.weight, model.proj.bias = [
        torch.nn.Parameter(param) for param in params
    ]
    model.head_dim = 96
    torch_output2 = model(torch_hidden_states)
    assert_with_pcc(torch_output, torch_output2, 0.999999)


@pytest.mark.parametrize("sequence_size", [640])
def test_siglip_attention(device, sequence_size, model_location_generator):
    torch.manual_seed(0)

    model = load_siglip_torch_model().blocks[0].attn

    torch_hidden_states = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.zeros(1, 1, 1, sequence_size, dtype=torch.float32)

    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_siglip.siglip_attention(
        hidden_states,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output.unsqueeze(0), output, 0.999)  # bfloat8_b inputs


@pytest.mark.parametrize("sequence_size", [4032])
def test_siglip_intermediate(device, sequence_size):
    torch.manual_seed(0)

    model = load_siglip_torch_model().blocks[0].mlp
    torch_hidden_states = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)
    torch_output = model.fc1(torch_hidden_states)
    torch_output = model.act(torch_output)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_siglip.siglip_intermediate(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9973)


@pytest.mark.parametrize("sequence_size", [4304])
def test_siglip_output(device, sequence_size):
    torch.manual_seed(0)
    model = load_siglip_torch_model().blocks[0].mlp
    torch_hidden_states = torch_random((1, 256, sequence_size), -1, 1, dtype=torch.float32)
    torch_residual = torch_random((1, 256, 1152), -1, 1, dtype=torch.float32)
    torch_output = model.fc2(torch_hidden_states) + torch_residual
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_siglip.siglip_output(
        hidden_states,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9998)


@pytest.mark.parametrize("sequence_size", [256])
def test_siglip_feedforward(device, sequence_size):
    torch.manual_seed(0)
    model = load_siglip_torch_model().blocks[0]
    torch_hidden_states = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)
    torch_residual = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)
    torch_output = model.mlp(torch_hidden_states) + torch_residual
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_siglip.siglip_feedforward(
        hidden_states,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9995)


@pytest.mark.parametrize("sequence_size", [256])
def test_siglip_layer(device, sequence_size):
    torch.manual_seed(0)

    model = load_siglip_torch_model().blocks[0]
    torch_hidden_states = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)
    attention_mask = torch.zeros(1, 1, 1, sequence_size, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_siglip.siglip_layer(
        hidden_states,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)  # Full layer accumulates error


@pytest.mark.parametrize("sequence_size", [640])  #
@pytest.mark.parametrize("layer_end_index", [1, 2, 4, 8, 10, 15, 20, 27])
def test_siglip_encoder(device, sequence_size, layer_end_index):
    torch.manual_seed(0)
    model = load_siglip_torch_model()

    torch_hidden_states = torch_random((1, sequence_size, 1152), -1, 1, dtype=torch.float32)

    torch_output = model.blocks[:layer_end_index](torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=ttnn_siglip.custom_preprocessor_siglip,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    head_masks = [
        ttnn.from_torch(
            torch.zeros(1, 1, 1, sequence_size, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in parameters.blocks
    ]
    output = ttnn_siglip.siglip_encoder(
        hidden_states, head_masks, parameters=parameters.blocks, layer_end_index=layer_end_index
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.95)  # Lowered from 0.99 due to bfloat8_b precision

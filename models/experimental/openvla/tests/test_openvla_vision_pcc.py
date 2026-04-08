# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for OpenVLA Vision Backbone and Projector.

Note: These tests use RANDOM WEIGHTS for fast, self-contained testing.
For tests with actual OpenVLA weights (higher PCC ~0.99+), see the full model test.

Test order follows the data flow:
1. DINOv2 attention
2. DINOv2 feedforward
3. SigLIP layer
4. Full vision encoder
5. Projector
"""

import pytest
import torch
import torch.nn as nn

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.openvla.tt import tt_optimized_openvla_vision
from tests.ttnn.utils_for_testing import assert_with_pcc


def ttnn_to_torch_safe(tensor, mesh_device=None):
    """Convert ttnn tensor to torch, handling multi-device mesh."""
    try:
        device_tensors = ttnn.get_device_tensors(tensor)
        return ttnn.to_torch(device_tensors[0]).float()
    except (RuntimeError, TypeError, AttributeError):
        return ttnn.to_torch(tensor).float()


# =============================================================================
# TEST 1: DINOv2 Attention
# =============================================================================
@pytest.mark.parametrize("batch_size", [1])
def test_dinov2_attention_pcc(device, batch_size):
    """
    Test DINOv2 attention from tt_optimized_openvla_vision (using random weights).

    DINOv2-Large: hidden_dim=1024, num_heads=16, head_dim=64.
    """
    torch.manual_seed(42)

    hidden_dim = 1024
    seq_len = 261  # 256 patches + 1 CLS + 4 register tokens
    num_heads = 16
    head_dim = hidden_dim // num_heads  # 64

    torch_input = torch.randn(batch_size, seq_len, hidden_dim)

    # Create random weights for attention (in PyTorch linear format: [out_features, in_features])
    norm1_weight = torch.randn(hidden_dim)
    norm1_bias = torch.randn(hidden_dim)
    qkv_weight = torch.randn(hidden_dim * 3, hidden_dim)  # [3072, 1024] PyTorch format
    qkv_bias = torch.randn(hidden_dim * 3)
    proj_weight = torch.randn(hidden_dim, hidden_dim)  # [1024, 1024]
    proj_bias = torch.randn(hidden_dim)
    ls1_scale = torch.ones(hidden_dim) * 0.1

    # PyTorch reference
    def pytorch_dinov2_attention(x, params):
        normed = torch.nn.functional.layer_norm(x, [hidden_dim], params[0], params[1], eps=1e-6)
        qkv = torch.nn.functional.linear(normed, params[3], params[2])  # Use weight directly (PyTorch format)
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = head_dim**-0.5
        q = q * scale
        attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        out = torch.nn.functional.linear(out, params[5], params[4])  # Use weight directly
        out = out * params[6]
        return x + out

    with torch.no_grad():
        torch_output = pytorch_dinov2_attention(
            torch_input, [norm1_weight, norm1_bias, qkv_bias, qkv_weight, proj_bias, proj_weight, ls1_scale]
        )

    # TTNN - weights need to be transposed (TTNN uses [in_features, out_features])
    tensors = [
        norm1_weight.unsqueeze(0),
        norm1_bias.unsqueeze(0),
        qkv_bias.unsqueeze(0),
        qkv_weight.T.contiguous(),  # [1024, 3072] for TTNN
        proj_bias.unsqueeze(0),
        proj_weight.T.contiguous(),  # [1024, 1024] for TTNN
        ls1_scale.reshape(1, 1, -1),
    ]
    tensors_tt = tt_optimized_openvla_vision.prepare_dinov2_attention_constants(tensors, device)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_output = tt_optimized_openvla_vision.dinov2_attention(tt_input, *tensors_tt)
    tt_output_torch = ttnn_to_torch_safe(tt_output, device)

    _, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.90)

    print(f"\n{'='*60}")
    print(f"DINOv2 ATTENTION PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.90, f"DINOv2 attention PCC {pcc_value} too low"


# =============================================================================
# TEST 2: DINOv2 Feedforward
# =============================================================================
@pytest.mark.parametrize("batch_size", [1])
def test_dinov2_feedforward_pcc(device, batch_size):
    """
    Test DINOv2 feedforward from tt_optimized_openvla_vision (using random weights).
    """
    torch.manual_seed(42)

    hidden_dim = 1024
    seq_len = 261
    mlp_dim = hidden_dim * 4

    torch_input = torch.randn(batch_size, seq_len, hidden_dim)

    # Create random weights (in PyTorch linear format: [out_features, in_features])
    norm2_weight = torch.randn(hidden_dim)
    norm2_bias = torch.randn(hidden_dim)
    fc1_weight = torch.randn(mlp_dim, hidden_dim)  # [4096, 1024] PyTorch format
    fc1_bias = torch.randn(mlp_dim)
    fc2_weight = torch.randn(hidden_dim, mlp_dim)  # [1024, 4096] PyTorch format
    fc2_bias = torch.randn(hidden_dim)
    ls2_scale = torch.ones(hidden_dim) * 0.1

    # PyTorch reference
    def pytorch_dinov2_feedforward(x, params):
        normed = torch.nn.functional.layer_norm(x, [hidden_dim], params[0], params[1], eps=1e-6)
        out = torch.nn.functional.linear(normed, params[3], params[2])  # Use weight directly
        out = torch.nn.functional.gelu(out)
        out = torch.nn.functional.linear(out, params[5], params[4])  # Use weight directly
        out = out * params[6]
        return x + out

    with torch.no_grad():
        torch_output = pytorch_dinov2_feedforward(
            torch_input, [norm2_weight, norm2_bias, fc1_bias, fc1_weight, fc2_bias, fc2_weight, ls2_scale]
        )

    # TTNN - weights need to be transposed (TTNN uses [in_features, out_features])
    tensors = [
        norm2_weight.unsqueeze(0),
        norm2_bias.unsqueeze(0),
        fc1_bias.unsqueeze(0),
        fc1_weight.T.contiguous(),  # [1024, 4096] for TTNN
        fc2_bias.unsqueeze(0),
        fc2_weight.T.contiguous(),  # [4096, 1024] for TTNN
        ls2_scale.reshape(1, 1, -1),
    ]
    tensors_tt = tt_optimized_openvla_vision.prepare_dinov2_feedforward_constants(tensors, device)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_output = tt_optimized_openvla_vision.dinov2_feedforward(tt_input, *tensors_tt)
    tt_output_torch = ttnn_to_torch_safe(tt_output, device)

    _, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.90)

    print(f"\n{'='*60}")
    print(f"DINOv2 FEEDFORWARD PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.90, f"DINOv2 feedforward PCC {pcc_value} too low"


# =============================================================================
# TEST 3: SigLIP Layer
# =============================================================================
@pytest.mark.parametrize("batch_size", [1])
def test_siglip_layer_pcc(device, batch_size):
    """
    Test SigLIP layer from tt_optimized_openvla_vision (using pretrained weights).
    """
    from timm.models import create_model
    from ttnn.model_preprocessing import preprocess_model_parameters

    torch.manual_seed(42)

    hidden_dim = 1152
    seq_len = 256

    # Load pretrained model for reference
    model = create_model("vit_so400m_patch14_siglip_224", pretrained=True).eval()
    block = model.blocks[0]

    torch_input = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.zeros(1, 1, 1, seq_len)

    with torch.no_grad():
        torch_output = block(torch_input)

    # TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: block.to(torch.bfloat16),
        device=device,
        custom_preprocessor=tt_optimized_openvla_vision.custom_preprocessor_siglip,
    )

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_mask = ttnn.from_torch(
        attention_mask.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_output = tt_optimized_openvla_vision.siglip_layer(tt_input, tt_mask, parameters)
    tt_output_torch = ttnn_to_torch_safe(tt_output, device)

    _, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.90)

    print(f"\n{'='*60}")
    print(f"SigLIP LAYER PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.90, f"SigLIP layer PCC {pcc_value} too low"


# =============================================================================
# TEST 4: Projector
# =============================================================================
class SimpleProjector(nn.Module):
    """MLP projector matching OpenVLA architecture."""

    def __init__(self, input_dim=2176, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TTSimpleProjector:
    """TTNN implementation of projector."""

    def __init__(self, device, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        self.device = device

        mesh_mapper = None
        if hasattr(device, "shape") and tuple(device.shape) != (1, 1):
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        self.fc1_weight = ttnn.from_torch(
            fc1_weight.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
        self.fc1_bias = ttnn.from_torch(
            fc1_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
        )
        self.fc2_weight = ttnn.from_torch(
            fc2_weight.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
        self.fc2_bias = ttnn.from_torch(
            fc2_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
        )

    def forward(self, x):
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias, activation="gelu")
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias)
        return x


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("input_dim", [2176])  # DINOv2(1024) + SigLIP(1152)
@pytest.mark.parametrize("output_dim", [4096])
def test_projector_pcc(device, batch_size, seq_len, input_dim, output_dim):
    """
    Test OpenVLA projector (using random weights).

    Input: fused vision features [batch, 256, 2176]
    Output: projected features [batch, 256, 4096]
    """
    torch.manual_seed(42)

    torch_model = SimpleProjector(input_dim, output_dim)
    torch_model.eval()

    torch_input = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_model = TTSimpleProjector(
        device,
        torch_model.fc1.weight.data,
        torch_model.fc1.bias.data,
        torch_model.fc2.weight.data,
        torch_model.fc2.bias.data,
    )

    mesh_mapper = None
    if hasattr(device, "shape") and tuple(device.shape) != (1, 1):
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
    )

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn_to_torch_safe(tt_output, device)

    _, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.99)

    print(f"\n{'='*60}")
    print(f"PROJECTOR PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert_with_pcc(torch_output, tt_output_torch, 0.99)


# =============================================================================
# TEST 5: Fused Vision Backbone (DINOv2 + SigLIP)
# =============================================================================
@pytest.mark.parametrize("batch_size", [1])
def test_fused_vision_backbone_pcc(device, batch_size):
    """
    Test fused vision backbone: DINOv2 features + SigLIP features concatenated.

    This tests the concatenation logic used in OpenVLA's vision backbone:
    - DINOv2: [batch, 256, 1024] (after dropping CLS + register tokens)
    - SigLIP: [batch, 256, 1152]
    - Fused: [batch, 256, 2176]

    Uses random features to test the concatenation logic (not actual encoder outputs).
    """
    torch.manual_seed(42)

    seq_len = 256  # 16x16 patches
    dinov2_dim = 1024
    siglip_dim = 1152  # fused_dim = dinov2_dim + siglip_dim = 2176

    # Create random features simulating encoder outputs
    dinov2_features = torch.randn(batch_size, seq_len, dinov2_dim)
    siglip_features = torch.randn(batch_size, seq_len, siglip_dim)

    # PyTorch reference: concatenate
    torch_fused = torch.cat([dinov2_features, siglip_features], dim=2)

    # TTNN
    mesh_mapper = None
    if hasattr(device, "shape") and tuple(device.shape) != (1, 1):
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    tt_dinov2 = ttnn.from_torch(
        dinov2_features.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=mesh_mapper,
    )
    tt_siglip = ttnn.from_torch(
        siglip_features.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=mesh_mapper,
    )

    # Concatenate (same as open_vla.py vision backbone)
    tt_fused = ttnn.concat([tt_dinov2, tt_siglip], dim=2)
    tt_fused_torch = ttnn_to_torch_safe(tt_fused, device)

    _, pcc_value = comp_pcc(torch_fused, tt_fused_torch, 0.99)

    print(f"\n{'='*60}")
    print(f"FUSED VISION BACKBONE PCC RESULTS")
    print(f"{'='*60}")
    print(f"  DINOv2 features shape: {dinov2_features.shape}")
    print(f"  SigLIP features shape: {siglip_features.shape}")
    print(f"  Fused shape: {torch_fused.shape}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert_with_pcc(torch_fused, tt_fused_torch, 0.99)

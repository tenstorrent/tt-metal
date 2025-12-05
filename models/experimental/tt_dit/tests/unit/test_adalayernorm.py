# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test AdaLayerNorm implementations for SD3.5 Medium
"""

import pytest
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.layers.adalayernorm import (
    AdaLayerNormZero,
    AdaLayerNormContinuous,
    SD35AdaLayerNormZeroX,
)


class TorchAdaLayerNormZero(torch.nn.Module):
    """PyTorch reference implementation of AdaLayerNormZero"""

    def __init__(self, hidden_size: int, conditioning_size: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, conditioning_size, bias=bias)
        self.norm = torch.nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, x, c):
        normalized_x = self.norm(x)
        scale = torch.nn.functional.silu(self.linear(c))
        # Reshape to split into 6 chunks
        B = scale.shape[0]
        scale = scale.view(B, 1, 6, -1)
        return normalized_x, scale


class TorchAdaLayerNormContinuous(torch.nn.Module):
    """PyTorch reference implementation of AdaLayerNormContinuous"""

    def __init__(self, hidden_size: int, conditioning_size: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, conditioning_size, bias=bias)
        self.norm = torch.nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, x, c):
        normalized_x = self.norm(x)
        scale = torch.nn.functional.silu(self.linear(c))
        # Reshape to split into 2 chunks
        B = scale.shape[0]
        scale = scale.view(B, 1, 2, -1)
        return normalized_x, scale


class TorchSD35AdaLayerNormZeroX(torch.nn.Module):
    """PyTorch reference implementation of SD35AdaLayerNormZeroX"""

    def __init__(self, hidden_size: int, conditioning_size: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, conditioning_size, bias=bias)
        self.norm = torch.nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, x, c):
        normalized_x = self.norm(x)
        scale = torch.nn.functional.silu(self.linear(c))
        # Reshape to split into 9 chunks
        B = scale.shape[0]
        scale = scale.view(B, 1, 9, -1)
        return normalized_x, scale


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["n150"],
    indirect=True,
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 1, 4096, 1536),  # Typical SD3.5 Medium shape
    ],
)
@pytest.mark.parametrize(
    ("hidden_size", "conditioning_size"),
    [
        (1536, 9216),  # AdaLayerNormZero (6x)
        (1536, 3072),  # AdaLayerNormContinuous (2x)
        (1536, 13824),  # SD35AdaLayerNormZeroX (9x)
    ],
)
def test_adalayernorm_zero(
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    hidden_size: int,
    conditioning_size: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create reference model based on conditioning size
    if conditioning_size == 9216:
        torch_model = TorchAdaLayerNormZero(hidden_size, conditioning_size).to(dtype=torch_dtype)
        tt_model = AdaLayerNormZero(hidden_size, conditioning_size, mesh_device=mesh_device)
    elif conditioning_size == 3072:
        torch_model = TorchAdaLayerNormContinuous(hidden_size, conditioning_size).to(dtype=torch_dtype)
        tt_model = AdaLayerNormContinuous(hidden_size, conditioning_size, mesh_device=mesh_device)
    elif conditioning_size == 13824:
        torch_model = TorchSD35AdaLayerNormZeroX(hidden_size, conditioning_size).to(dtype=torch_dtype)
        tt_model = SD35AdaLayerNormZeroX(hidden_size, conditioning_size, mesh_device=mesh_device)
    else:
        raise ValueError(f"Unknown conditioning size: {conditioning_size}")

    torch_model.eval()
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Create test inputs
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4
    torch_conditioning = torch.randn((input_shape[0], input_shape[1], hidden_size), dtype=torch_dtype)

    # Convert to TTNN tensors
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    tt_conditioning = ttnn.from_torch(
        torch_conditioning,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    # Run forward pass
    with torch.no_grad():
        torch_normalized, torch_scale = torch_model(torch_input_tensor, torch_conditioning)

    tt_normalized, tt_scale = tt_model(tt_input_tensor, tt_conditioning)

    # Compare normalized output
    for t in ttnn.get_device_tensors(tt_normalized):
        t = ttnn.to_torch(t)
        assert_quality(torch_normalized, t, pcc=0.999_500)

    # Compare scale output
    for t in ttnn.get_device_tensors(tt_scale):
        t = ttnn.to_torch(t)
        # TTNN adds a leading dimension, so t has shape [1, B, chunks, hidden_size]
        # torch_scale has shape [B, 1, chunks, hidden_size]
        B = torch_scale.shape[0]
        expected_chunks = 6 if conditioning_size == 9216 else (2 if conditioning_size == 3072 else 9)

        # Reshape torch_scale to match TTNN format by adding leading dimension
        torch_scale_reshaped = torch_scale.view(1, B, expected_chunks, hidden_size)
        assert_quality(torch_scale_reshaped, t, pcc=0.999_500)


def assert_quality(torch_tensor: torch.Tensor, tt_tensor: torch.Tensor, pcc: float):
    """Helper function to assert PCC quality"""
    passing, output = comp_pcc(torch_tensor, tt_tensor, pcc)
    logger.info(f"PCC: {output}")
    assert passing, f"PCC value {output} is lower than required {pcc}"


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["n150"],
    indirect=True,
)
def test_adalayernorm_block_configurations(mesh_device: ttnn.MeshDevice) -> None:
    """Test all three AdaLayerNorm types as they would be used in SD3.5 Medium blocks"""

    hidden_size = 1536
    torch_dtype = torch.bfloat16

    # Test Block 0-12 configuration (SD35AdaLayerNormZeroX + AdaLayerNormZero)
    norm1 = SD35AdaLayerNormZeroX(hidden_size, hidden_size * 9, mesh_device=mesh_device)
    norm1_context = AdaLayerNormZero(hidden_size, hidden_size * 6, mesh_device=mesh_device)

    # Test Block 13-22 configuration (AdaLayerNormZero + AdaLayerNormZero)
    norm1_standard = AdaLayerNormZero(hidden_size, hidden_size * 6, mesh_device=mesh_device)
    norm1_context_standard = AdaLayerNormZero(hidden_size, hidden_size * 6, mesh_device=mesh_device)

    # Test Block 23 configuration (AdaLayerNormZero + AdaLayerNormContinuous)
    norm1_last = AdaLayerNormZero(hidden_size, hidden_size * 6, mesh_device=mesh_device)
    norm1_context_last = AdaLayerNormContinuous(hidden_size, hidden_size * 2, mesh_device=mesh_device)

    # Create and load dummy state dicts with correct shapes
    dummy_state_dict_9x = {
        "linear.weight": torch.randn(hidden_size * 9, hidden_size, dtype=torch_dtype),  # Transposed
        "linear.bias": torch.randn(hidden_size * 9, dtype=torch_dtype),
    }
    dummy_state_dict_6x = {
        "linear.weight": torch.randn(hidden_size * 6, hidden_size, dtype=torch_dtype),  # Transposed
        "linear.bias": torch.randn(hidden_size * 6, dtype=torch_dtype),
    }
    dummy_state_dict_2x = {
        "linear.weight": torch.randn(hidden_size * 2, hidden_size, dtype=torch_dtype),  # Transposed
        "linear.bias": torch.randn(hidden_size * 2, dtype=torch_dtype),
    }

    norm1.load_torch_state_dict(dummy_state_dict_9x)
    norm1_context.load_torch_state_dict(dummy_state_dict_6x)
    norm1_standard.load_torch_state_dict(dummy_state_dict_6x)
    norm1_context_standard.load_torch_state_dict(dummy_state_dict_6x)
    norm1_last.load_torch_state_dict(dummy_state_dict_6x)
    norm1_context_last.load_torch_state_dict(dummy_state_dict_2x)

    # Create test inputs
    x = torch.randn(1, 1, 1024, hidden_size, dtype=torch_dtype)
    c = torch.randn(1, 1, hidden_size, dtype=torch_dtype)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    tt_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    # Test all configurations run without error
    norm1(tt_x, tt_c)
    norm1_context(tt_x, tt_c)
    norm1_standard(tt_x, tt_c)
    norm1_context_standard(tt_x, tt_c)
    norm1_last(tt_x, tt_c)
    norm1_context_last(tt_x, tt_c)

    logger.info("All AdaLayerNorm configurations tested successfully")

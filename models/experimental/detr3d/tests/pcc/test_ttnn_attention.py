# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from models.common.utility_functions import comp_pcc
from models.experimental.detr3d.ttnn.multihead_attention import TTNNMultiheadAttention

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger
from models.experimental.detr3d.common import load_torch_model_state


@pytest.mark.parametrize("device_params", [{"l1_small_size": 13684}], indirect=True)
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("d_model", [256])
@pytest.mark.parametrize("nhead", [4])
def test_multihead_attention(device, batch_size, seq_len, d_model, nhead, reset_seeds):
    """Test TTNN MultiheadAttention against PyTorch reference implementation"""

    # Create PyTorch reference model
    torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False)
    load_torch_model_state(torch_mha, "encoder.layers.0.self_attn")

    # Create TTNN model
    ttnn_mha = TTNNMultiheadAttention(d_model, nhead, device)

    # Extract and convert PyTorch weights to TTNN format
    with torch.no_grad():
        # Convert to TTNN tensors
        ttnn_mha.q_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[:d_model].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.k_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[d_model : 2 * d_model].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.v_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[2 * d_model :].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.q_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[:d_model].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_mha.k_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[d_model : 2 * d_model].reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.v_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[2 * d_model :].reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.out_weight = ttnn.from_torch(
            torch_mha.out_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_mha.out_bias = ttnn.from_torch(
            torch_mha.out_proj.bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    # Create test inputs
    torch_input = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output, _ = torch_mha(torch_input, torch_input, torch_input)

    # TTNN forward pass
    ttnn_input = ttnn.from_torch(
        torch_input.permute(1, 0, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_output = ttnn_mha(ttnn_input, ttnn_input, ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = torch.permute(ttnn_output_torch, (1, 0, 2))

    # Calculate and log PCC before assertion
    passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, 0.99)
    logger.info(f"PCC: {pcc_message}")
    logger.info(f"PCC: {torch_output.shape}")
    logger.info(f"PCC: {ttnn_output_torch.shape}")

    # Compare outputs with PCC (Pearson Correlation Coefficient)
    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)

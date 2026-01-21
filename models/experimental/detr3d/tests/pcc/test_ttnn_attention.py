# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.multihead_attention import TtnnMultiheadAttention
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead",
    [
        (128, 1, 256, 4),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_multihead_attention(device, batch_size, seq_len, d_model, nhead, reset_seeds):
    # Create PyTorch reference model
    torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False)
    load_torch_model_state(torch_mha, "encoder.layers.0.self_attn")

    # Create test inputs
    torch_input = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output, _ = torch_mha(torch_input, torch_input, torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_mha,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    # Create TTNN model
    ttnn_mha = TtnnMultiheadAttention(d_model, nhead, device, parameters=parameters)

    # TTNN forward pass
    ttnn_input = ttnn.from_torch(
        torch_input.permute(1, 0, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_output = ttnn_mha(ttnn_input, ttnn_input, ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = torch.permute(ttnn_output_torch, (1, 0, 2))

    # Compare outputs with PCC (Pearson Correlation Coefficient)
    passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, 0.99)
    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(torch_output, ttnn_output_torch))
    logger.info(f"Output shape - PyTorch: {torch_output.shape}, TTNN: {ttnn_output_torch.shape}")
    logger.info(f"Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}, Heads: {nhead}")

    if passing:
        logger.info("MultiheadAttention Test Passed!")
    else:
        logger.warning("MultiheadAttention Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"

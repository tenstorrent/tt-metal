from typing import Literal
import torch
import torch.nn as nn
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
import math


@pytest.mark.parametrize(
    "B,C,T,H,W",
    [
        # From blocks.0 and blocks.1 (first section)
        (1, 768, 28, 60, 106),
        # From blocks.2 (second section)
        (1, 512, 82, 120, 212),
        # From blocks.3 (third section)
        (1, 256, 163, 240, 424),
        # From blocks.4 (fourth section)
        (1, 128, 163, 480, 848),
    ],
)
@pytest.mark.parametrize("parallel_factor", [8])
def test_silu_tt(device, B, C, T, H, W, parallel_factor):
    # Set a manual seed for reproducibility
    torch.manual_seed(42)
    T = math.ceil(T / parallel_factor)

    input_shape = (B, C, T, H, W)
    # Create random input tensor
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Create torch SiLU for ground truth
    silu = nn.SiLU()
    torch_output = silu(input_tensor)

    # Attain my format
    inp = input_tensor.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, C)
    # Convert to ttnn tensor
    tt_input_tensor = ttnn.from_torch(
        inp,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute SiLU using ttnn operations
    tt_output = ttnn.silu(tt_input_tensor)

    # Convert back to torch
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # Undo my format
    tt_output = tt_output.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    # Compare outputs
    pcc, mse, mae = compute_metrics(torch_output, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Assertions to verify correctness
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_l1_loss(device):
    # Create reference and prediction tensors
    input_reference = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
    input_prediction = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)

    # Compute L1 loss
    output = ttnn.l1_loss(input_reference, input_prediction)
    logger.info(f"L1 Loss result: {output}")


def test_mse_loss(device):
    # Create reference and prediction tensors
    input_reference = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
    input_prediction = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)

    # Compute MSE loss
    output = ttnn.mse_loss(input_reference, input_prediction)
    logger.info(f"MSE Loss result: {output}")

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger


def test_addcdiv(device):
    # Create three tensors and a value for the operation
    value = 1.0
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the addcdiv operation: tensor1 + value * (tensor2 / tensor3)
    output = ttnn.addcdiv(tensor1, tensor2, tensor3, value=value)
    logger.info(f"Addcdiv result: {output}")


def test_addcmul(device):
    # Create three tensors and a value for the operation
    value = 1.0
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the addcmul operation: tensor1 + value * (tensor2 * tensor3)
    output = ttnn.addcmul(tensor1, tensor2, tensor3, value=value)
    logger.info(f"Addcmul result: {output}")


def test_mac(device):
    # Create three tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[5, 6], [7, 8]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[9, 10], [11, 12]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the mac operation
    output = ttnn.mac(tensor1, tensor2, tensor3)
    logger.info(f"MAC result: {output}")


def test_where(device):
    # Create predicate tensor of 0's and 1's and two value tensors
    predicate = ttnn.from_torch(
        torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    true_value = ttnn.from_torch(
        torch.tensor([[5, 6], [7, 8]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    false_value = ttnn.from_torch(
        torch.tensor([[9, 10], [11, 12]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the where operation, giving [[5, 10], [11, 8]]
    output = ttnn.where(predicate, true_value, false_value)
    logger.info(f"Where result: {output}")


def test_lerp(device):
    # Create three tensors
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the lerp operation
    output = ttnn.lerp(tensor1, tensor2, tensor3)
    logger.info(f"Lerp result: {output}")


def test_addcmul_bw(device):
    # Create three tensors and a value for the operation
    value = 1.0
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the addcmul backward operation
    output = ttnn.addcmul_bw(grad_tensor, tensor1, tensor2, tensor3, alpha=value)
    logger.info(f"Addcmul Backward result: {output}")


def test_addcdiv_bw(device):
    # Create three tensors and a value for the operation
    value = 1.0
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the addcdiv backward operation
    output = ttnn.addcdiv_bw(grad_tensor, tensor1, tensor2, tensor3, alpha=value)
    logger.info(f"Addcdiv Backward result: {output}")


def test_where_bw(device):
    # Create the forward inputs, with a predicate of 0's and 1's, and a gradient tensor
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    predicate = ttnn.from_torch(
        torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    true_value = ttnn.from_torch(
        torch.tensor([[5, 6], [7, 8]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    false_value = ttnn.from_torch(
        torch.tensor([[9, 10], [11, 12]], dtype=torch.bfloat16, requires_grad=True),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Perform the where backward operation. Each gradient depends only on grad_tensor and the predicate,
    # giving [[1, 0], [3, 0]] for true_value and [[0, 2], [0, 4]] for false_value
    output = ttnn.where_bw(grad_tensor, predicate, true_value, false_value)
    logger.info(f"Where Backward result: {output}")


def test_lerp_bw(device):
    # Create three tensors and a gradient tensor for the operation
    grad_tensor = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor1 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor2 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor3 = ttnn.from_torch(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Perform the lerp backward operation
    output = ttnn.lerp_bw(grad_tensor, tensor1, tensor2, tensor3)
    logger.info(f"Lerp Backward result: {output}")

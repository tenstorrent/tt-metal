#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Linear Regression with Tensor Parallelism (TP) and Data Parallelism (DP).

This is a Python port of the C++ linear_regression_tp_dp example, demonstrating
combined TP+DP training using Python bindings.

The mesh is organized as:
- DP dimension (mesh dim 0): Groups for data parallelism
- TP dimension (mesh dim 1): Devices per group for tensor parallelism

Example: 8x4 mesh = 32 devices total (8 DP groups × 4 TP devices each)

Supported parallelism strategies:
- ColumnParallelLinear (default): Shards output features across TP devices.
  Input is broadcast, output remains sharded.
- RowParallelLinear (--row flag): Shards input features across TP devices.
  Output is all-reduced to produce full result.

Usage:
    python linear_regression_tp_dp.py [--row] [-b BATCH_SIZE] [--mesh_shape RxC]

Requirements:
    - TT-Metal with fabric enabled
    - Sufficient devices for the mesh (dp_size × tp_size)
"""

from __future__ import annotations

import argparse
import math
import numpy as np
from typing import Optional
from ttml.autograd import DistributedConfig

import ttnn
import ttml


# ---------------------------------------------------------------------------
# Distributed Linear Modules (Python implementations matching C++)
# ---------------------------------------------------------------------------


class RowParallelLinear(ttml.modules.ModuleBase):
    """Row-parallel linear layer.

    Shards input features across TP devices.
    Each device computes partial output, then all-reduces to get final output.

    For row-parallel: Y = XW^T + b, where X is sharded along feature dim.
    Each device has W[in_features/tp_size, out_features].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        shard_dim: Optional[int] = None,
        weight_seed: int = 12345,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.shard_dim = shard_dim

        # Get device and TP size
        device = ttml.autograd.AutoContext.get_instance().get_device()
        if shard_dim is not None:
            self.tp_size = device.shape[shard_dim]
        else:
            self.tp_size = device.get_num_devices()

        # Each device has a shard of the weight matrix
        # TTML weight shape convention: [1, 1, out_features, in_features]
        # For RowParallel: shard along in_features (dim 3)
        # Per-device weight shape: [1, 1, out_features, in_features/tp_size]

        # Initialize weight with uniform distribution matching C++ (with fixed seed for reproducibility)
        rng = np.random.default_rng(weight_seed)
        init_k = math.sqrt(1.0 / in_features)
        weight_data = rng.uniform(-init_k, init_k, (1, 1, out_features, in_features)).astype(np.float32)

        # Create sharded weight tensor - shard along last dim (in_features)
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, shard_dim)
        self.weight = ttml.autograd.Tensor.from_numpy(weight_data, ttnn.Layout.TILE, None, mapper)

        self.bias = None
        if has_bias:
            # Bias is replicated across TP devices, uniform init like C++ (using same rng for reproducibility)
            bias_data = rng.uniform(-init_k, init_k, (1, 1, 1, out_features)).astype(np.float32)
            self.bias = ttml.autograd.Tensor.from_numpy(bias_data, ttnn.Layout.TILE)

        self.create_name("row_parallel_linear")
        self.register_tensor(self.weight, "weight")
        if self.bias is not None:
            self.register_tensor(self.bias, "bias")

    def __call__(self, tensor):
        x = tensor

        if not self.input_is_parallel:
            # Scatter input along TP dimension
            x = ttml.ops.distributed.scatter(x, tensor.get_rank() - 1, self.shard_dim)

        # Linear operation (no bias here)
        x = ttml.ops.linear.linear(x, self.weight, None)

        # All-reduce with noop_backward=input_is_parallel to avoid double all-reduce in backward
        # (broadcast does all-reduce in backward, so if input_is_parallel we skip that)
        x = ttml.ops.distributed.all_reduce(x, self.input_is_parallel, self.shard_dim)

        # Add bias after all-reduce
        if self.bias is not None:
            x = ttml.ops.binary.add(x, self.bias)

        return x


class ColumnParallelLinear(ttml.modules.ModuleBase):
    """Column-parallel linear layer.

    Broadcasts input to all TP devices.
    Each device computes a shard of output features.
    Optionally gathers output.

    For column-parallel: Y = XW^T + b, where W is sharded along output dim.
    Each device has W[in_features, out_features/tp_size].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = False,
        shard_dim: Optional[int] = None,
        weight_seed: int = 12345,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.shard_dim = shard_dim

        # Get device and TP size
        device = ttml.autograd.AutoContext.get_instance().get_device()
        if shard_dim is not None:
            self.tp_size = device.shape[shard_dim]
        else:
            self.tp_size = device.get_num_devices()

        # Each device has a shard of the weight matrix
        # TTML weight shape convention: [1, 1, out_features, in_features]
        # For ColumnParallel: shard along out_features (dim 2)
        # Per-device weight shape: [1, 1, out_features/tp_size, in_features]

        # Initialize weight with uniform distribution matching C++ (with fixed seed for reproducibility)
        rng = np.random.default_rng(weight_seed)
        init_k = math.sqrt(1.0 / in_features)
        weight_data = rng.uniform(-init_k, init_k, (1, 1, out_features, in_features)).astype(np.float32)

        # Create sharded weight tensor - shard along out_features (dim 2)
        weight_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 2, shard_dim)
        self.weight = ttml.autograd.Tensor.from_numpy(weight_data, ttnn.Layout.TILE, None, weight_mapper)

        self.bias = None
        if has_bias:
            # Bias shape: [1, 1, 1, out_features] - shard along last dim, uniform init like C++ (using same rng)
            bias_data = rng.uniform(-init_k, init_k, (1, 1, 1, out_features)).astype(np.float32)
            bias_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, shard_dim)
            self.bias = ttml.autograd.Tensor.from_numpy(bias_data, ttnn.Layout.TILE, None, bias_mapper)

        self.create_name("column_parallel_linear")
        self.register_tensor(self.weight, "weight")
        if self.bias is not None:
            self.register_tensor(self.bias, "bias")

    def __call__(self, tensor):
        # Broadcast input along TP dimension
        x = ttml.ops.distributed.broadcast(tensor, self.shard_dim)

        # Linear operation with sharded weight and bias
        x = ttml.ops.linear.linear(x, self.weight, self.bias)

        if self.gather_output:
            # All-gather output along TP dimension
            x = ttml.ops.distributed.all_gather(x, tensor.get_rank() - 1, self.shard_dim)

        return x


# ---------------------------------------------------------------------------
# Data generation (matching C++ make_regression)
# ---------------------------------------------------------------------------


def make_regression(
    n_samples: int,
    n_features: int,
    n_targets: int,
    noise: float = 0.0,
    bias: bool = True,
    seed: int = 42,
):
    """Generate synthetic regression data."""
    rng = np.random.default_rng(seed)

    # Generate random weight matrix
    W = rng.standard_normal((n_features, n_targets)).astype(np.float32)

    # Generate random features
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Compute targets: Y = X @ W
    Y = X @ W

    if bias:
        b = rng.standard_normal(n_targets).astype(np.float32)
        Y += b

    if noise > 0:
        Y += noise * rng.standard_normal(Y.shape).astype(np.float32)

    return X, Y


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Linear Regression TP+DP Example")
    parser.add_argument(
        "--row",
        action="store_true",
        help="Use RowParallelLinear (shards input features), default is ColumnParallelLinear",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="8x4",
        help="Logical mesh shape RxC (e.g. 8x4) where R=DP size, C=TP size",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Parse mesh shape
    try:
        mesh_parts = args.mesh_shape.lower().split("x")
        if len(mesh_parts) != 2:
            raise ValueError()
        mesh_rows = int(mesh_parts[0])
        mesh_cols = int(mesh_parts[1])
        if mesh_rows <= 0 or mesh_cols <= 0:
            raise ValueError()
    except (ValueError, IndexError):
        print(f"Error: invalid --mesh_shape '{args.mesh_shape}', expected RxC like 8x4")
        return 1

    if mesh_rows * mesh_cols != 32:
        print("Error: mesh_rows and mesh_cols must be their product must be 32 (whole galaxy).")
        return 1

    # Mesh configuration
    # - DP groups (data parallelism) along mesh dimension 0
    # - TP devices per group (tensor parallelism) along mesh dimension 1
    logical_mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
    num_devices = mesh_rows * mesh_cols

    # Training hyperparameters
    training_samples_count = 100000
    num_features = 64
    num_targets = 64
    noise = 0.0
    has_bias = True
    batch_size = args.batch_size
    use_row_parallel = args.row

    # Enable fabric BEFORE opening the device
    ttml.core.distributed.enable_fabric(num_devices)

    # Open device with mesh shape
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device([mesh_rows, mesh_cols])
    device = autograd_ctx.get_device()

    # Initialize parallelism context for TP+DP
    autograd_ctx.initialize_parallelism_context(DistributedConfig(enable_ddp=True, enable_tp=True))

    # Get parallelism parameters from context
    pctx = autograd_ctx.get_parallelism_context()
    tp_axis = pctx.get_tp_axis()
    dp_size = pctx.get_ddp_size()
    tp_size = pctx.get_tp_size()

    # Validate dimensions
    if num_features % tp_size != 0:
        print(f"Error: num_features ({num_features}) must be divisible by tp_size ({tp_size})")
        return 1
    if num_targets % tp_size != 0:
        print(f"Error: num_targets ({num_targets}) must be divisible by tp_size ({tp_size})")
        return 1
    if batch_size == 0:
        print("Error: batch_size must be > 0")
        return 1
    if batch_size % dp_size != 0:
        print(f"Error: batch_size ({batch_size}) must be divisible by dp_size ({dp_size})")
        return 1

    # Generate training dataset
    print(f"Generating {training_samples_count} training samples...")
    X_train, Y_train = make_regression(
        n_samples=training_samples_count,
        n_features=num_features,
        n_targets=num_targets,
        noise=noise,
        bias=has_bias,
    )

    # Create model based on parallelism type
    # shard_dim=tp_axis specifies weights should be sharded along TP dimension
    # Use fixed weight_seed so Row and Column parallel have identical initial weights
    weight_seed = 12345
    if use_row_parallel:
        print("Using RowParallelLinear: shards input features, all_reduces output")
        model = RowParallelLinear(
            num_features,
            num_targets,
            has_bias=has_bias,
            input_is_parallel=True,  # Input already sharded from data distribution
            shard_dim=tp_axis,
            weight_seed=weight_seed,
        )
    else:
        print("Using ColumnParallelLinear: shards output features, sharded output")
        model = ColumnParallelLinear(
            num_features,
            num_targets,
            has_bias=has_bias,
            gather_output=False,  # Keep output sharded
            shard_dim=tp_axis,
            weight_seed=weight_seed,
        )

    print(f"Batch size: {batch_size}, DP groups: {dp_size}, TP size: {tp_size}")

    # Configure optimizer - use same formula as C++
    learning_rate = 0.1 * num_targets * (batch_size / 128.0)
    if not use_row_parallel:
        # Loss is calculated for each TP partition, so it's averaged over 1/tp_size
        # times fewer samples, making gradient tp_size times greater
        learning_rate /= tp_size

    sgd_config = ttml.optimizers.SGDConfig.make(learning_rate, 0.0, 0.0, 0.0, False)
    optimizer = ttml.optimizers.SGD(model.parameters(), sgd_config)

    def create_batch_tensors(x_batch: np.ndarray, y_batch: np.ndarray):
        """Create distributed tensors for a batch using proper 2D mesh mapping.

        For ColumnParallel (default):
        - Data: Shard{0} on DP (batch dim), Replicate on TP (broadcast to all TP devices)
        - Targets: Shard{0} on DP (batch dim), Shard{3} on TP (shard output features)

        For RowParallel:
        - Data: Shard{0} on DP (batch dim), Shard{3} on TP (shard input features)
        - Targets: Shard{0} on DP (batch dim), Replicate on TP (matches all-reduced output)
        """
        actual_batch_size = x_batch.shape[0]

        # Reshape to [batch, 1, 1, features] to match C++ convention
        x_batch = x_batch.reshape(actual_batch_size, 1, 1, num_features).astype(np.float32)
        y_batch = y_batch.reshape(actual_batch_size, 1, 1, num_targets).astype(np.float32)

        # Configure data mapper based on parallelism type
        # Placements list: [DP placement, TP placement]
        if use_row_parallel:
            # RowParallelLinear: input is sharded across TP, so data should be sharded
            data_config = ttnn.MeshMapperConfig(
                [
                    ttnn.PlacementShard(0),
                    ttnn.PlacementShard(3),
                ],  # DP: shard batch, TP: shard features
                logical_mesh_shape,
            )
        else:
            # ColumnParallelLinear: input is broadcast, so data should be replicated on TP
            data_config = ttnn.MeshMapperConfig(
                [
                    ttnn.PlacementShard(0),
                    ttnn.PlacementReplicate(),
                ],  # DP: shard batch, TP: replicate
                logical_mesh_shape,
            )

        data_mapper = ttnn.create_mesh_mapper(device, data_config)

        # Create data tensor using ttml.autograd.Tensor.from_numpy with mapper
        data_tensor = ttml.autograd.Tensor.from_numpy(x_batch, ttnn.Layout.TILE, None, data_mapper)

        # Configure targets mapper based on parallelism type
        if use_row_parallel:
            # RowParallelLinear: output is all-reduced (replicated), so targets should be replicated on TP
            targets_config = ttnn.MeshMapperConfig(
                [
                    ttnn.PlacementShard(0),
                    ttnn.PlacementReplicate(),
                ],  # DP: shard batch, TP: replicate
                logical_mesh_shape,
            )
        else:
            # ColumnParallelLinear with gather_output=false: output is sharded, so targets should be sharded
            targets_config = ttnn.MeshMapperConfig(
                [
                    ttnn.PlacementShard(0),
                    ttnn.PlacementShard(3),
                ],  # DP: shard batch, TP: shard features
                logical_mesh_shape,
            )

        targets_mapper = ttnn.create_mesh_mapper(device, targets_config)

        # Create targets tensor (no gradient needed for targets)
        targets_tensor = ttml.autograd.Tensor.from_numpy(y_batch, ttnn.Layout.TILE, None, targets_mapper)
        targets_tensor.set_requires_grad(False)

        return data_tensor, targets_tensor

    def get_loss_value(loss):
        """Aggregate loss over all devices and return mean."""
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        loss_numpy = loss.to_numpy(composer=composer)
        return float(loss_numpy.mean())

    # Training loop
    training_step = 0
    num_epochs = args.epochs
    num_batches = training_samples_count // batch_size

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(training_samples_count)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            x_batch = X_shuffled[start_idx:end_idx]
            y_batch = Y_shuffled[start_idx:end_idx]

            # Create distributed tensors with proper 2D mesh mapping
            data, targets = create_batch_tensors(x_batch, y_batch)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute loss
            loss = ttml.ops.loss.mse_loss(output, targets, ttml.ops.ReduceType.MEAN)

            # Log loss
            loss_val = get_loss_value(loss)
            print(f"Step: {training_step} Loss: {loss_val:.6f}")
            training_step += 1

            # Backward pass (retain_graph=False to free graph after backward)
            loss.backward(False)

            # Synchronize gradients across DP groups
            ttml.core.distributed.synchronize_gradients(model.parameters())

            # Optimizer step
            optimizer.step()

            # Reset autograd graph
            autograd_ctx.reset_graph()

    print("\nTraining complete!")

    # Cleanup
    autograd_ctx.close_device()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Training functions for hierarchical parallel transformer training.

This module contains the training logic for both 2-tier and 3-tier architectures:
- worker(): Worker training loop using RemoteOptimizer
- aggregator(): Aggregates gradients from workers (3-tier only)
- optimizer(): Applies optimizer updates (3-tier only)
- aggregator_optimizer(): Combined aggregator+optimizer for 2-tier architecture
"""
from time import time

import numpy as np
import ttnn
import ttml
from ttml.common.data import get_batch, build_causal_mask
from ttml.common.utils import no_grad, PerformanceMeter


def get_batch_ttml(
    ids: np.ndarray, seq_len: int, batch_size: int, use_ddp: bool = False
):
    """Prepare a batch of data for TTML training.

    Args:
        ids: Array of token IDs
        seq_len: Sequence length
        batch_size: Batch size
        use_ddp: Whether to use distributed data parallel

    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    x_u32, y_u32 = get_batch(ids, seq_len, batch_size)

    # TTML shapes: inputs [B,1,1,T] (uint32), targets [B,T] (uint32)
    if use_ddp:
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
        tt_x = ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(batch_size, 1, 1, seq_len),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            mapper,
        )
        tt_y = ttml.autograd.Tensor.from_numpy(
            y_u32, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper
        )
    else:
        tt_x = ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(batch_size, 1, 1, seq_len),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
        )
        tt_y = ttml.autograd.Tensor.from_numpy(
            y_u32, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        )
    return tt_x, tt_y


def worker(
    cfg,
    model,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    use_ddp: bool = False,
    use_tp: bool = False,
    num_workers: int = None,
):
    """Execute worker training loop.

    Workers compute forward/backward passes and use RemoteOptimizer to:
    - Send gradients to aggregator
    - Receive updated weights from aggregator

    Args:
        cfg: Training configuration object
        model: Model to train
        train_ids: Training data token IDs
        val_ids: Validation data token IDs (unused, for API compatibility)
        use_ddp: Whether to use distributed data parallel
        use_tp: Whether to use tensor parallel
        num_workers: Number of worker ranks (if None, assumes 3-tier with world_size - 2)

    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    # Setup loss function and causal mask
    loss_fn = ttml.ops.loss.cross_entropy_loss
    reduce = ttml.ops.ReduceType.MEAN

    causal_mask = build_causal_mask(cfg.seq_len)
    tt_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )

    # Setup distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    distributed_ctx = autograd_ctx.get_distributed_context()
    device = autograd_ctx.get_device()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    # Determine num_workers if not provided
    if num_workers is None:
        # Default to 3-tier: workers are ranks 0 to num_workers-1, aggregator is num_workers, optimizer is num_workers+1
        num_workers = world_size - 2  # Subtract aggregator and optimizer

    aggregator_rank = num_workers

    assert rank < num_workers, f"Worker rank {rank} must be < num_workers {num_workers}"

    # Create RemoteOptimizer that communicates with aggregator (or aggregator_optimizer in 2-tier)
    optimizer = ttml.optimizers.RemoteOptimizer(model.parameters(), aggregator_rank)

    # Create composer for distributed tensors if using DDP or TP
    composer = None
    if use_ddp or use_tp:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Training state
    model.train()
    train_losses = []
    val_losses = []  # Unused, kept for API compatibility

    # Receive initial weights from aggregator (or aggregator_optimizer in 2-tier)
    print(f"[Worker {rank}] Receiving initial weights from rank {aggregator_rank}")
    optimizer.receive_weights()
    print(f"[Worker {rank}] Received initial weights")

    performance_meter = PerformanceMeter(cfg)

    # Training loop: outer loop = optimizer steps
    for step in range(1, cfg.steps + 1):
        performance_meter.step()
        optimizer.zero_grad()
        accum_loss = 0.0

        # Gradient accumulation loop
        for _ in range(cfg.gradient_accumulation_steps):
            tt_x, tt_y = get_batch_ttml(train_ids, cfg.seq_len, cfg.batch_size, use_ddp)

            # Forward and backward pass
            logits = model(tt_x, tt_mask)
            loss = loss_fn(logits, tt_y, reduce)

            # Scale loss by accumulation steps for proper gradient averaging
            if cfg.gradient_accumulation_steps > 1:
                loss = loss * (1.0 / cfg.gradient_accumulation_steps)

            loss.backward(False)

            # Convert loss to numpy for logging
            loss_numpy = loss.to_numpy(composer=composer)
            train_loss = loss_numpy.mean()
            accum_loss += train_loss

            # Reset computation graph after each micro-batch
            autograd_ctx.reset_graph()

        # RemoteOptimizer.step() sends gradients to aggregator and receives updated weights
        optimizer.step()

        # Log training progress
        train_losses.append(accum_loss)
        if step > 10:
            samples_per_second, tokens_per_second = performance_meter.get_metrics()
            # scale by number of workers
            samples_per_second = samples_per_second * num_workers
            tokens_per_second = tokens_per_second * num_workers
            print(
                f"[Worker {rank}] Step {step}/{cfg.steps}: Loss = {accum_loss:.4f}, "
                f"Samples/s = {samples_per_second:.2f}, Tokens/s = {tokens_per_second:.2f}"
            )
        else:
            print(f"[Worker {rank}] Step {step}/{cfg.steps}: Loss = {accum_loss:.4f}")

    print(f"[Worker {rank}] Training finished")
    return train_losses, val_losses


@no_grad()
def aggregator(model, cfg, use_ddp: bool = False):
    """Aggregator that averages gradients from workers and broadcasts weights.

    The aggregator sits between workers and the optimizer:
    1. Receives gradients from all workers
    2. Averages the gradients
    3. Optionally applies DDP reduction across devices
    4. Sends averaged gradients to optimizer
    5. Receives updated weights from optimizer
    6. Broadcasts updated weights to all workers

    Args:
        model: Model instance (for getting parameters)
        cfg: Training configuration
        use_ddp: Whether to apply DDP reduction on gradients
    """
    # Setup distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    distributed_ctx = autograd_ctx.get_distributed_context()
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    # Calculate number of workers (exclude aggregator and optimizer)
    num_workers = world_size - 2

    # Get sorted parameters for consistent ordering
    parameters = model.parameters()
    sorted_parameters = dict(sorted(parameters.items()))

    # Create sub-contexts for communication
    workers_and_aggregator_ranks = list(range(num_workers + 1))
    workers_and_aggregator_ctx = distributed_ctx.create_sub_context(
        workers_and_aggregator_ranks
    )

    aggregator_and_optimizer_ranks = [rank, rank + 1]
    aggregator_and_optimizer_ctx = distributed_ctx.create_sub_context(
        aggregator_and_optimizer_ranks
    )

    # In sub-context: aggregator is local rank 0, optimizer is local rank 1
    optimizer_local_rank = 1

    # Receive and broadcast initial weights from optimizer to workers
    print(f"[Aggregator {rank}] Waiting for initial weights from optimizer {rank + 1}")
    for name, tensor_ptr in sorted_parameters.items():
        socket_manager.recv(
            tensor_ptr, aggregator_and_optimizer_ctx, optimizer_local_rank
        )

        # Broadcast to all workers
        for worker_id in range(num_workers):
            socket_manager.send(tensor_ptr, workers_and_aggregator_ctx, worker_id)

    print(f"[Aggregator {rank}] Starting training loop for {cfg.steps} steps")

    # Training loop
    for step in range(cfg.steps):
        # Receive and average gradients from all workers
        for name, tensor_ptr in sorted_parameters.items():
            # Receive gradient from first worker
            socket_manager.recv(tensor_ptr, workers_and_aggregator_ctx, 0)

            # Receive and accumulate gradients from remaining workers
            to_add = ttml.core.empty_like(tensor_ptr)
            for worker_id in range(1, num_workers):
                socket_manager.recv(to_add, workers_and_aggregator_ctx, worker_id)
                tensor_ptr = tensor_ptr + to_add

            # Average the gradients (in-place on tensor_ptr's grad)
            tensor_ptr = tensor_ptr * (1.0 / num_workers)

            # Apply DDP reduction across devices if enabled
            if use_ddp:
                tensor_ptr = ttml.ops.distributed.all_reduce(tensor_ptr)

            # Send averaged gradient to optimizer (local rank 1 in sub-context)
            socket_manager.send(
                tensor_ptr, aggregator_and_optimizer_ctx, optimizer_local_rank
            )

        # Receive updated weights from optimizer and broadcast to all workers
        for name, tensor_ptr in sorted_parameters.items():
            socket_manager.recv(
                tensor_ptr, aggregator_and_optimizer_ctx, optimizer_local_rank
            )

            # Broadcast to all workers
            for worker_id in range(num_workers):
                socket_manager.send(tensor_ptr, workers_and_aggregator_ctx, worker_id)

    print(f"[Aggregator {rank}] Completed {cfg.steps} steps")


def optimizer(model, cfg, optimizer_instance):
    """Optimizer that applies optimizer updates.

    The optimizer is responsible for:
    1. Sending initial weights to aggregator
    2. Receiving averaged gradients from aggregator
    3. Applying optimizer step (e.g., AdamW with momentum)
    4. Sending updated weights back to aggregator

    Args:
        model: Model instance (for getting parameters)
        cfg: Training configuration
        optimizer_instance: Optimizer instance to use for updates
    """
    # Setup distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    distributed_ctx = autograd_ctx.get_distributed_context()
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()

    # Get sorted parameters for consistent ordering
    parameters = model.parameters()
    sorted_parameters = dict(sorted(parameters.items()))

    # Create sub-context for aggregator-optimizer communication
    aggregator_global_rank = rank - 1
    aggregator_and_optimizer_ranks = [aggregator_global_rank, rank]
    aggregator_and_optimizer_ctx = distributed_ctx.create_sub_context(
        aggregator_and_optimizer_ranks
    )

    # In sub-context: aggregator is local rank 0, optimizer is local rank 1
    aggregator_local_rank = 0

    # Send initial weights to aggregator
    print(
        f"[Optimizer {rank}] Sending initial weights to aggregator {aggregator_global_rank}"
    )
    for name, tensor_ptr in sorted_parameters.items():
        socket_manager.send(
            tensor_ptr, aggregator_and_optimizer_ctx, aggregator_local_rank
        )

    print(f"[Optimizer {rank}] Starting training loop for {cfg.steps} steps")

    # Training loop
    for step in range(cfg.steps):
        # Receive gradients from aggregator (local rank 0 in sub-context)
        for name, tensor_ptr in sorted_parameters.items():
            socket_manager.recv(
                tensor_ptr,
                aggregator_and_optimizer_ctx,
                aggregator_local_rank,
                use_grad=True,
            )

        # Apply optimizer step
        optimizer_instance.step()

        # Send updated weights back to aggregator (local rank 0 in sub-context)
        for name, tensor_ptr in sorted_parameters.items():
            socket_manager.send(
                tensor_ptr, aggregator_and_optimizer_ctx, aggregator_local_rank
            )

    print(f"[Optimizer {rank}] Completed {cfg.steps} steps")


def aggregator_optimizer(model, cfg, optimizer_instance, use_ddp: bool = False):
    """Combined aggregator and optimizer for 2-tier hierarchical parallel training.

    This function combines the roles of aggregator and optimizer:
    1. Receives gradients from all workers
    2. Averages the gradients
    3. Optionally applies DDP reduction across devices
    4. Sets the averaged gradients on model parameters
    5. Applies optimizer step
    6. Broadcasts updated weights to all workers

    Args:
        model: Model instance (for getting parameters)
        cfg: Training configuration
        optimizer_instance: Optimizer instance to use for updates
        use_ddp: Whether to apply DDP reduction on gradients
    """
    # Setup distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    distributed_ctx = autograd_ctx.get_distributed_context()
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    # Calculate number of workers (exclude aggregator_optimizer)
    num_workers = world_size - 1

    # Get sorted parameters for consistent ordering
    parameters = model.parameters()
    sorted_parameters = dict(sorted(parameters.items()))

    # We need to create a sub-context for all workers to receive gradients from because of RemoteOptimizer that creates
    # a sub-context for all workers and aggregator. In our case with aggregator_optimizer.
    all_ranks = list(range(world_size))
    all_ctx = distributed_ctx.create_sub_context(all_ranks)

    # Send initial weights to all workers
    print(
        f"[AggregatorOptimizer {rank}] Sending initial weights to {num_workers} workers"
    )
    for worker_id in range(num_workers):
        for name, tensor_ptr in sorted_parameters.items():
            socket_manager.send(tensor_ptr, all_ctx, worker_id)

    print(f"[AggregatorOptimizer {rank}] Starting training loop for {cfg.steps} steps")

    # Training loop
    for step in range(cfg.steps):
        # Receive and average gradients from all workers
        for name, tensor_ptr in sorted_parameters.items():
            # Receive gradient from first worker
            with no_grad():
                grad_tensor = ttml.core.empty_like(tensor_ptr)
                socket_manager.recv(grad_tensor, all_ctx, 0)

                # Receive and accumulate gradients from remaining workers
                for worker_id in range(1, num_workers):
                    to_add = ttml.core.empty_like(tensor_ptr)
                    socket_manager.recv(to_add, all_ctx, worker_id)
                    grad_tensor = grad_tensor + to_add

                # Average the gradients
                grad_tensor = grad_tensor * (1.0 / num_workers)

                # Apply DDP reduction across devices if enabled
                if use_ddp:
                    grad_tensor = ttml.ops.distributed.all_reduce(grad_tensor)

            # Set the gradient on the parameter for optimizer step
            tensor_ptr.set_grad_from_tensor(grad_tensor)

        # Apply optimizer step
        optimizer_instance.step()

        # Broadcast updated weights to all workers
        for name, tensor_ptr in sorted_parameters.items():
            for worker_id in range(num_workers):
                socket_manager.send(tensor_ptr, all_ctx, worker_id)

        if (step + 1) % 10 == 0:
            print(f"[AggregatorOptimizer {rank}] Completed step {step + 1}/{cfg.steps}")

    print(f"[AggregatorOptimizer {rank}] Completed {cfg.steps} steps")

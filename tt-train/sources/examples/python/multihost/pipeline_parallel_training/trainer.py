# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Training loop and batch preparation for pipeline parallel transformer models."""

import numpy as np
import ttnn
import ttml
from ttml.common.data import get_batch, build_causal_mask
from ttml.common.utils import PerformanceMeter, no_grad


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


def train(
    cfg,
    model,
    optim,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    use_ddp: bool = False,
    use_tp: bool = False,
):
    """Execute pipeline parallel training loop.

    In pipeline parallelism:
    - First rank (rank 0) receives input data and processes through initial layers
    - Middle ranks receive activations from previous rank and process through their layers
    - Final rank receives activations, processes through final layers, and computes loss

    Args:
        cfg: Training configuration object
        model: Pipeline parallel model to train
        optim: Optimizer
        train_ids: Training data token IDs
        val_ids: Validation data token IDs (unused, for API compatibility)
        use_ddp: Whether to use distributed data parallel
        use_tp: Whether to use tensor parallel

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
    socket_manager = autograd_ctx.get_socket_manager()
    device = autograd_ctx.get_device()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()
    is_first_stage = rank == 0
    is_final_stage = rank == world_size - 1

    assert (
        world_size > 1
    ), f"Pipeline parallel requires world_size > 1, got {world_size}"

    # Create composer for distributed tensors if using DDP or TP
    composer = None
    if use_ddp or use_tp:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Training state
    model.train()
    train_losses = []
    val_losses = []  # Unused, kept for API compatibility

    performance_meter = PerformanceMeter(cfg)
    # Training loop: outer loop = optimizer steps, inner loop = gradient accumulation
    for step in range(1, cfg.steps + 1):
        performance_meter.step()
        optim.zero_grad()
        accum_loss = 0.0

        # Gradient accumulation loop
        for _ in range(cfg.gradient_accumulation_steps):
            # Generate batch data
            # Note: All ranks generate batches, but only rank 0's input is used.
            # Pipeline model handles activation passing between ranks automatically.
            tt_x, tt_y = get_batch_ttml(train_ids, cfg.seq_len, cfg.batch_size, use_ddp)

            # Transfer targets from first stage to final stage
            # Only the final stage computes loss, so it needs the correct targets
            if is_first_stage:
                socket_manager.send(tt_y, distributed_ctx, world_size - 1)
            elif is_final_stage:
                tt_y = socket_manager.recv(tt_y, distributed_ctx, 0)

            # Forward and backward pass
            # Pipeline model automatically handles inter-stage communication
            logits = model(tt_x, tt_mask)

            if is_final_stage:
                # Only final stage computes loss
                loss = loss_fn(logits, tt_y, reduce)

                # Scale loss by accumulation steps for proper gradient averaging
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss * (1.0 / cfg.gradient_accumulation_steps)

                loss.backward(False)

                # Convert loss to numpy for logging
                loss_numpy = loss.to_numpy(composer=composer)
                train_loss = loss_numpy.mean()
                accum_loss += train_loss
            else:
                # Non-final stages only propagate gradients backward
                logits.backward(False)

            # Reset computation graph after each micro-batch
            autograd_ctx.reset_graph()

        # Synchronize gradients across data parallel dimension (if enabled)
        if use_ddp:
            ttml.core.distributed.synchronize_gradients(model.parameters())

        # Update model parameters
        optim.step()

        # Log training progress (only on final stage)
        if is_final_stage:
            train_losses.append(accum_loss)
            samples_per_second, tokens_per_second = performance_meter.get_metrics()
            print(
                f"Step {step}/{cfg.steps}: Loss = {accum_loss:.4f}, Samples/s = {samples_per_second:.2f}, Tokens/s = {tokens_per_second:.2f}"
            )

    return train_losses, val_losses

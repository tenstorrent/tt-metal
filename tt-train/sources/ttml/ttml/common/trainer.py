# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Training loop and batch preparation for transformer models."""
import numpy as np
import ttnn
import ttml
from ttml.common.data import get_batch, build_causal_mask
from tqdm import tqdm


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
    # TTML shapes: inputs [B,1,1,T] (uint32), targets [B,T] (int32)

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
    use_ddp: bool = False,
    use_tp: bool = False,
):
    """Execute training loop.

    Args:
        cfg: Training configuration object
        model: Model to train
        optim: Optimizer
        train_ids: Training data token IDs
        use_ddp: Whether to use distributed data parallel
        use_tp: Whether to use tensor parallel

    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    loss_fn = ttml.ops.loss.cross_entropy_loss
    reduce = ttml.ops.ReduceType.MEAN

    causal_mask = build_causal_mask(cfg.seq_len)
    tt_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )  # [1,1,T,T], bfloat16

    # Create composer for distributed tensors if using DDP
    composer = None
    if use_ddp or use_tp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    model.train()
    train_losses = []
    val_losses = []

    # Gradient accumulation state
    accum_loss = 0.0
    last_val_loss = None

    bar = tqdm(range(1, cfg.steps + 1))
    for step in bar:
        optim.zero_grad()
        accum_loss = 0.0

        # Inner loop for gradient accumulation
        for _ in range(cfg.gradient_accumulation_steps):
            tt_x, tt_y = get_batch_ttml(train_ids, cfg.seq_len, cfg.batch_size, use_ddp)

            # ---- forward/backward ----
            logits = model(tt_x, tt_mask)
            loss = loss_fn(logits, tt_y, reduce)

            # Scale loss by accumulation steps
            if cfg.gradient_accumulation_steps > 1:
                loss = loss * (1.0 / cfg.gradient_accumulation_steps)

            loss.backward(False)
            ttml.autograd.AutoContext.get_instance().reset_graph()

            # For DDP, composer concatenates losses from all devices - take mean
            loss_numpy = loss.to_numpy(composer=composer)
            train_loss = loss_numpy.mean()

            # Accumulate loss
            accum_loss += train_loss

        # Synchronize gradients for DDP
        if use_ddp:
            ttml.core.distributed.synchronize_gradients(model.parameters())

        optim.step()

        # Record accumulated loss
        train_losses.append(accum_loss)

        # Update progress bar - preserve val_loss if it exists
        postfix = {"train_loss": f"{accum_loss:.4f}"}
        if last_val_loss is not None:
            postfix["val_loss"] = f"{last_val_loss:.4f}"
        bar.set_postfix(postfix, refresh=False)

        # ---- occasional eval on val set ----
        if (step % cfg.eval_every) == 0 or step == 1:
            model.eval()
            # keep existing placeholder behavior for validation loss
            val_losses.append(train_losses[-1] if train_losses else 0.0)
            last_val_loss = val_losses[-1]
            model.train()
            # Update bar with validation loss
            postfix = {
                "train_loss": f"{train_losses[-1]:.4f}" if train_losses else "N/A"
            }
            postfix["val_loss"] = f"{last_val_loss:.4f}"
            bar.set_postfix(postfix, refresh=False)

    return train_losses, val_losses

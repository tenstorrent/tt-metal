# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Training loop and batch preparation for transformer models."""
import numpy as np
import ttml
from tqdm import tqdm
from data import get_batch, build_causal_mask


def get_batch_ttml(ids: np.ndarray, seq_len: int, batch_size: int, use_ddp: bool = False):
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
            x_u32.reshape(batch_size, 1, 1, seq_len), ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper
        )
        tt_y = ttml.autograd.Tensor.from_numpy(y_u32, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper)
    else:
        tt_x = ttml.autograd.Tensor.from_numpy(
            x_u32.reshape(batch_size, 1, 1, seq_len), ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )
        tt_y = ttml.autograd.Tensor.from_numpy(y_u32, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)
    return tt_x, tt_y

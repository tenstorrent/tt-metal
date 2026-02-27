# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Custom distributed autograd ops for tensor-parallel Qwen3.

Provides:
  - AllGatherFwdScatterBwd / all_gather_fwd_scatter_bwd:
      all_gather in forward, scatter (reduce_scatter / tp_size) in backward.
      Correctly averages gradients when the gathered output feeds replicated
      per-device computation (e.g. a replicated loss).
  - _BroadcastMul:
      Multiply by a broadcast-compatible constant with no gradient for the
      second operand.  Works around ttml.ops.binary.mul not reducing broadcast
      dims in backward.
  - _vocab_parallel_embedding:
      Megatron-LM style VocabParallelEmbedding for tied weights.  Each TP
      device owns a vocab shard, masks out-of-range tokens, then all-reduces
      hidden vectors.
"""

import numpy as np
import ttnn
import ttml

from utils.tensor_utils import get_device, get_tp_size


# ---------------------------------------------------------------------------
# AllGatherFwdScatterBwd — custom autograd op
# ---------------------------------------------------------------------------


class AllGatherFwdScatterBwd(ttml.autograd.Function):
    """all_gather forward, scatter (reduce_scatter / tp_size) backward.

    The standard all_gather backward is reduce_scatter, which sums tp_size
    identical gradients when the gathered output feeds into identical
    per-device computation (e.g. a replicated loss).  This op uses scatter
    (reduce_scatter / tp_size) as backward, which correctly averages them.
    """

    @staticmethod
    def forward(ctx, x, dim, shard_dim):
        ctx.dim = dim
        ctx.shard_dim = shard_dim
        # all_gather on a detached tensor (bypass autograd of the library op)
        temp = ttml.autograd.create_tensor(x.get_value(), requires_grad=False)
        gathered = ttml.ops.distributed.all_gather(temp, dim, shard_dim)
        return ttml.autograd.create_tensor(gathered.get_value())

    @staticmethod
    def backward(ctx, grad_output):
        # scatter = reduce_scatter / tp_size (bypass autograd)
        temp = ttml.autograd.create_tensor(grad_output, requires_grad=False)
        scattered = ttml.ops.distributed.scatter(temp, ctx.dim, ctx.shard_dim)
        return scattered.get_value()


def all_gather_fwd_scatter_bwd(x, dim, shard_dim):
    """all_gather forward, scatter backward — drop-in replacement."""
    return AllGatherFwdScatterBwd.apply(x, dim, shard_dim)


# ---------------------------------------------------------------------------
# Vocab-parallel embedding (Megatron-LM style)
# ---------------------------------------------------------------------------


class _BroadcastMul(ttml.autograd.Function):
    """Multiply by a broadcast-compatible constant (no gradient for the mask).

    ``ttml.ops.binary.mul`` does not reduce broadcast dims in backward, so
    ``[B,1,T,H] * [B,1,T,1]`` crashes.  This function treats the second
    operand as a non-differentiable constant and only returns gradient for
    the first operand.
    """

    @staticmethod
    def forward(ctx, h, mask_val):
        ctx.mask_val = mask_val
        return ttnn.multiply(h.get_value(), mask_val)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx.mask_val)


def _vocab_parallel_embedding(input_ids_np, sharded_weight, vocab_size, shard_dim):
    """Megatron-LM style VocabParallelEmbedding for tied weights.

    Each TP device owns vocab range ``[rank*local_V, (rank+1)*local_V)``.
    Forward: local lookup → mask invalid positions → all-reduce (SUM).

    Only communicates hidden-dim vectors (small), never the weight table.
    The same ``[local_V, hidden]`` shard serves as the ColumnParallel LM head
    weight for the output projection.

    Args:
        input_ids_np: numpy uint32 token IDs, shape ``[B, 1, 1, T]`` or ``[B, T]``.
        sharded_weight: Per-device weight ``[1, 1, local_V, hidden]``
            (lm_head ColumnParallel weight).
        vocab_size: Full (un-padded) vocabulary size.
        shard_dim: Mesh dimension for TP communication.
    """
    device = get_device()
    tp_size = get_tp_size(shard_dim)
    local_V = int(sharded_weight.shape()[2])

    ids = input_ids_np.reshape(input_ids_np.shape[0], -1)  # [B, T]
    B, T = ids.shape

    all_local_ids = np.zeros((tp_size * B, 1, 1, T), dtype=np.uint32)
    all_valid_mask = np.zeros((tp_size * B, 1, T, 1), dtype=np.float32)

    for k in range(tp_size):
        offset = k * local_V
        local = ids.astype(np.int64) - offset
        valid = (local >= 0) & (local < local_V)
        local = np.clip(local, 0, local_V - 1).astype(np.uint32)
        all_local_ids[k * B : (k + 1) * B, 0, 0, :] = local
        all_valid_mask[k * B : (k + 1) * B, 0, :, 0] = valid.astype(np.float32)

    shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
        device, 0, shard_dim
    )
    local_ids_t = ttml.autograd.Tensor.from_numpy(
        all_local_ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, shard_mapper
    )
    valid_mask_t = ttml.autograd.Tensor.from_numpy(
        all_valid_mask, ttnn.Layout.TILE, ttnn.bfloat16, shard_mapper
    )

    h = ttml.ops.embedding.embedding(local_ids_t, sharded_weight)
    h = _BroadcastMul.apply(h, valid_mask_t.get_value())
    h = ttml.ops.distributed.all_reduce(h, True, shard_dim)
    return h

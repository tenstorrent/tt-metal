# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sharded cross-entropy loss for vocab-sharded logits.

Standard cross-entropy and log-softmax compute softmax over the full vocabulary,
which requires gathering all vocab logits on every device.  For large vocabularies
this is extremely expensive in memory.

This module implements a numerically-stable sharded cross-entropy that keeps
logits sharded along the vocabulary dimension and only communicates tiny
[B,1,S,1] correction tensors (max and sum) via all-gather / all-reduce.

Algorithm (forward):
    1. Each device computes local max and local exp-sum.
    2. All-gather local max (tiny) → compute global max locally.
    3. Recompute local exp-sum with global max shift → all-reduce sum (tiny).
    4. Extract target logit from the shard that holds it (one-hot × dot + all-reduce).
    5. loss = global_log_sum_exp - target_logit

Algorithm (backward):
    dL/dx_k = (1/N) * (softmax_k - onehot_k) * grad_output
    Entirely local — no inter-device communication required.
"""

import numpy as np

import ttnn
import ttml


def sharded_cross_entropy_loss(
    logits, targets_np, vocab_size, tp_size, tp_axis=1, dp_size=1
):
    """Cross-entropy loss for vocab-sharded logits.

    Parameters
    ----------
    logits : ttml.autograd.Tensor
        Logits sharded along dim 3 (vocab).  Shape per device: [B, 1, S, V/tp].
    targets_np : np.ndarray
        Target token IDs, shape [dp_size * B, S] (or [B, S] when dp_size=1),
        dtype uint32.  Lives on CPU; will be pre-processed and sharded to
        devices internally.  When dp_size > 1, rows are ordered by DP group:
        the first B rows go to DP group 0, the next B to DP group 1, etc.
    vocab_size : int
        Full (unsharded) vocabulary size.
    tp_size : int
        Tensor-parallelism degree.
    tp_axis : int
        Mesh dimension used for tensor parallelism (default 1).
    dp_size : int
        Data-parallelism degree (default 1).

    Returns
    -------
    ttml.autograd.Tensor
        Scalar loss (mean-reduced), with backward support.
    """
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    x = logits.get_value()
    shape = logits.shape()
    B, S, local_V = int(shape[0]), int(shape[2]), int(shape[3])
    N = B * S

    # ------------------------------------------------------------------
    # Forward  — only [B,1,S,1] tensors are communicated
    # ------------------------------------------------------------------

    # 1. Local max per device  →  [B,1,S,1]
    local_max = ttnn.max(x, dim=3, keepdim=True)

    # 2. All-gather local maxes in float32 for precision  →  [B,1,S,tp]  then local max  →  [B,1,S,1]
    lm = ttml.autograd.create_tensor(
        ttnn.typecast(local_max, ttnn.DataType.FLOAT32), False
    )
    all_max = ttml.ops.distributed.all_gather(lm, dim=3, cluster_axis=tp_axis)
    global_max = ttnn.max(all_max.get_value(), dim=3, keepdim=True)  # float32

    # 3. Shifted exp + local sum in float32 for numerical stability  →  all-reduce SUM (tiny!)
    # exp(bfloat16) has only ~7 mantissa bits; accumulating many such values wrecks the sum.
    # Upcasting to float32 before exp and moreh_sum gives full single-precision accuracy.
    shifted = ttnn.subtract(
        ttnn.typecast(x, ttnn.DataType.FLOAT32), global_max
    )  # float32
    local_exp = ttnn.exp(shifted)  # float32
    local_sum = ttnn.sum(local_exp, dim=3, keepdim=True)  # float32, [B,1,S,1]

    ls = ttml.autograd.create_tensor(local_sum, False)
    global_sum = ttml.ops.distributed.all_reduce(
        ls, noop_backward=True, cluster_axis=tp_axis
    ).get_value()  # float32, [B,1,S,1]

    log_normalizer = ttnn.add(global_max, ttnn.log(global_sum))  # float32, [B,1,S,1]

    # 4. Extract the logit value at the target position.
    #    Build a per-device one-hot mask on CPU, shard it, then dot-product.
    if dp_size > 1:
        num_devices = dp_size * tp_size
        onehot_np = np.zeros((num_devices * B, 1, S, local_V), dtype=np.float32)
        for d in range(dp_size):
            for k in range(tp_size):
                offset = k * local_V
                dev_idx = d * tp_size + k
                for b in range(B):
                    for s in range(S):
                        t = int(targets_np[d * B + b, s])
                        if offset <= t < offset + local_V:
                            onehot_np[dev_idx * B + b, 0, s, t - offset] = 1.0
        shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    else:
        onehot_np = np.zeros((tp_size * B, 1, S, local_V), dtype=np.float32)
        for k in range(tp_size):
            offset = k * local_V
            for b in range(B):
                for s in range(S):
                    t = int(targets_np[b, s])
                    if offset <= t < offset + local_V:
                        onehot_np[k * B + b, 0, s, t - offset] = 1.0
        shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            device, 0, cluster_axis=tp_axis
        )

    onehot_val = ttml.autograd.Tensor.from_numpy(
        onehot_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, shard_mapper
    ).get_value()  # [B,1,S,V/tp]

    target_logit_local = ttnn.sum(
        ttnn.multiply(x, onehot_val), dim=3, keepdim=True
    )  # [B,1,S,1]

    tl = ttml.autograd.create_tensor(
        ttnn.typecast(target_logit_local, ttnn.DataType.FLOAT32), False
    )
    target_logit = ttml.ops.distributed.all_reduce(
        tl, noop_backward=True, cluster_axis=tp_axis
    ).get_value()  # float32, [B,1,S,1]

    # 5. Per-position loss = log_normalizer − target_logit  →  mean  (all float32)
    per_pos = ttnn.subtract(log_normalizer, target_logit)  # float32, [B,1,S,1]

    # Mean = sum / N  (reduce dim 0 then dim 2 to get scalar [1,1,1,1])
    s = ttnn.sum(per_pos, dim=0, keepdim=True)  # [1,1,S,1]
    s = ttnn.sum(s, dim=2, keepdim=True)  # [1,1,1,1]
    loss_val = ttnn.multiply(s, 1.0 / float(N))  # [1,1,1,1]

    loss = ttml.autograd.create_tensor(loss_val)

    # ------------------------------------------------------------------
    # Backward  — purely local, zero communication
    # ------------------------------------------------------------------
    # dCE/dx_k = (softmax_k − onehot_k) / N  *  grad_output
    # softmax_k = exp(shifted_k) / global_sum
    #           = exp(x_k − global_max) / global_sum

    saved_shifted = shifted
    saved_global_sum = global_sum
    saved_onehot = onehot_val

    def backward():
        grad_out = loss.get_grad()  # [1,1,1,1]
        # saved_shifted and saved_global_sum are already float32 from the forward pass.
        # Re-computing exp in float32 keeps the softmax gradient numerically stable.
        softmax_k = ttnn.multiply(
            ttnn.exp(saved_shifted), ttnn.reciprocal(saved_global_sum)
        )  # float32
        diff = ttnn.subtract(
            softmax_k, ttnn.typecast(saved_onehot, ttnn.DataType.FLOAT32)
        )  # float32, [B,1,S,V/tp]
        grad = ttnn.multiply(diff, 1.0 / float(N))
        grad = ttnn.multiply(grad, ttnn.typecast(grad_out, ttnn.DataType.FLOAT32))
        logits.add_grad(ttnn.typecast(grad, ttnn.DataType.BFLOAT16))

    node = logits.get_node()
    node_id = ctx.add_backward_node(backward, [node] if node is not None else [])
    if node_id is not None:
        loss.set_node(node_id)

    return loss

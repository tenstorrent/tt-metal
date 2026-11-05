# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-token vectors to one embedding per text, the TTNN form of reference/postprocessing.py.

    last_hidden_state (B, 1, S, H)
      -> mean_pool           -> (B, 1, 1, H)   fp32, padding excluded
      -> matryoshka_truncate -> (B, 1, 1, dim) optional, feature axis
      -> l2_normalize        -> (B, 1, 1, dim) unit norm

Functions rather than a class: there are no weights here. The sequence axis disappears at
mean_pool, which is why this stage needs the batch-separated (B, 1, S, H) layout rather than the
flat token axis the MoE uses.

The pooled embedding is fp32 from mean_pool on. In bfloat16, l2_normalize left the norm anywhere
in 0.9952..1.0047, which the tests' 1 - dot against the unit reference counts as error; in fp32 it
is 1.000000.
"""

from __future__ import annotations

from typing import Optional

import ttnn

# Matches reference/postprocessing.mean_pool's own clamp on the keep count.
MASK_SUM_FLOOR = 1e-9

# Matches reference/postprocessing.L2_EPS, which reaches F.normalize as its eps. Applied to the
# sum of squares rather than to the norm, since that is what rsqrt consumes, so it is squared.
L2_EPS = 1e-12

# Only ttnn.matmul and ttnn.sum among the operators here accept a compute_kernel_config;
# ttnn.multiply, ttnn.rsqrt and ttnn.clamp take none, so the port's HiFi4 plus fp32-accumulation
# setting is applied to the mean's matmul and the norm's sum and is not expressible on the rest.


def mean_pool(hidden_states: ttnn.Tensor, mask_weights: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
    """Average each text's token vectors, ignoring padding.

    Mask-weighted rather than a plain mean over S, and not the CLS token: the checkpoint's
    1_Pooling/config.json sets pooling_mode_mean_tokens. Padding has to be excluded because the
    <pad> embedding is trained and non-zero, so counting it would make one text's embedding
    depend on how long its batch-mates are.

    One batched matmul, (B, 1, 1, S) weights against (B, 1, S, H) tokens, in place of a multiply,
    two sums, a clamp and a divide. Its K runs over the tile padding of S, where the hidden states
    can hold anything (the MoE layer's output padding is unwritten) and the weights are zero; the
    FPU's product of zero and inf or NaN is zero, so the padding drops out whatever it holds
    (test_mean_pool_ignores_non_finite_tile_padding).

    Args:
        hidden_states: (B, 1, S, H) encoder output.
        mask_weights: (B, 1, 1, S) from tt.common.pooling_mask: 1/count at real tokens, 0 at padding,
            the count floored the way the reference floors it, so a row that keeps nothing pools to
            zeros.
        compute_kernel_config: The port's device compute kernel config, applied to the matmul.

    Returns:
        ttnn.Tensor: (B, 1, 1, H) fp32, one vector per text. Not unit norm.
    """
    return ttnn.matmul(mask_weights, hidden_states, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config)


def matryoshka_truncate(embeddings: ttnn.Tensor, dim: Optional[int]) -> ttnn.Tensor:
    """Keep the leading dim features of each embedding.

    The feature axis, not the sequence axis. Upstream's own matryoshka_dim slices the sequence
    instead, dropping tokens at full feature width; that is a different operation and not what
    the published embeddings use.

    Args:
        embeddings: (B, 1, 1, H) pooled embeddings.
        dim: Target width, at most H, or None to pass through unchanged.

    Returns:
        ttnn.Tensor: (B, 1, 1, dim), or the input unchanged when dim is None.

    Raises:
        ValueError: If dim exceeds the embedding width.
    """
    if dim is None:
        return embeddings
    width = embeddings.shape[-1]
    if dim > width:
        raise ValueError(f"matryoshka dim {dim} exceeds embedding width {width}")
    if dim == width:
        return embeddings
    return ttnn.slice(embeddings, [0, 0, 0, 0], [embeddings.shape[0], 1, 1, dim])


def l2_normalize(embeddings: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
    """Scale each row to unit norm, so a dot product of two rows is their cosine similarity.

    rsqrt of the sum of squares rather than a divide by a sqrt: one fewer op, and no reciprocal
    of a value that could round to zero in bfloat16.

    The sum of squares is floored first. Without it a zero row gives rsqrt(0) = inf and then
    0 * inf = NaN, which would undo the zero weights of a fully padded row one operator later: such
    a row pools to zeros and must stay zeros rather than turning into NaN here. The reference gets
    this from F.normalize's eps.

    On mean_pool's fp32 output the norm comes out 1.000000 over 64 random rows; on a bfloat16
    input every step rounds and it lands anywhere in 0.9959..1.0043.

    Args:
        embeddings: (B, 1, 1, dim) pooled, optionally truncated embeddings.
        compute_kernel_config: The port's device compute kernel config, applied to the sum.

    Returns:
        ttnn.Tensor: (B, 1, 1, dim) with unit norm along the feature axis, or zeros for a row
        that arrived as zeros.
    """
    squared = ttnn.multiply(embeddings, embeddings)
    sum_of_squares = ttnn.sum(squared, dim=-1, keepdim=True, compute_kernel_config=compute_kernel_config)
    ttnn.deallocate(squared)

    scale = ttnn.rsqrt(ttnn.clamp(sum_of_squares, min=L2_EPS * L2_EPS))
    ttnn.deallocate(sum_of_squares)
    return ttnn.multiply(embeddings, scale)

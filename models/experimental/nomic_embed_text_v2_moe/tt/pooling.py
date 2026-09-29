# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-token vectors to one embedding per text, the TTNN form of reference/postprocessing.py.

    last_hidden_state (B, 1, S, H)
      -> mean_pool           -> (B, 1, 1, H)   padding excluded
      -> matryoshka_truncate -> (B, 1, 1, dim) optional, feature axis
      -> l2_normalize        -> (B, 1, 1, dim) unit norm

Functions rather than a class: there are no weights here. The sequence axis disappears at
mean_pool, which is why this stage needs the batch-separated (B, 1, S, H) layout rather than the
flat token axis the MoE uses.
"""

from __future__ import annotations

from typing import Optional

import ttnn

# Matches reference/postprocessing.mean_pool's own clamp on the keep count.
MASK_SUM_FLOOR = 1e-9

# Matches reference/postprocessing.L2_EPS, which reaches F.normalize as its eps. Applied to the
# sum of squares rather than to the norm, since that is what rsqrt consumes, so it is squared.
L2_EPS = 1e-12

# Only ttnn.sum among the operators here accepts a compute_kernel_config; ttnn.multiply,
# ttnn.divide, ttnn.rsqrt and ttnn.clamp take none, so the port's HiFi4 plus fp32-accumulation
# setting is applied to the two reduces and is simply not expressible on the rest.


def mean_pool(hidden_states: ttnn.Tensor, mask_weights: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
    """Average each text's token vectors, ignoring padding.

    Mask-weighted rather than a plain mean over S, and not the CLS token: the checkpoint's
    1_Pooling/config.json sets pooling_mode_mean_tokens. Padding has to be excluded because the
    <pad> embedding is trained and non-zero, so counting it would make one text's embedding
    depend on how long its batch-mates are.

    Args:
        hidden_states: (B, 1, S, H) encoder output.
        mask_weights: (B, 1, S, 1) from tt.common.pooling_mask, 1.0 at real tokens and 0.0 at
            padding. Its trailing singleton broadcasts over the hidden axis.
        compute_kernel_config: The port's device compute kernel config, applied to the two sums.

    Returns:
        ttnn.Tensor: (B, 1, 1, H), one vector per text. Not unit norm.
    """
    weighted = ttnn.multiply(hidden_states, mask_weights)
    # The divisor is floored the way the reference floors it. A fully padded row has a keep count
    # of zero, and dividing by it returns inf across all 768 features rather than raising, so the
    # row would leave here looking like data. Such a row cannot come from the tokenizer, which
    # always emits bos, but mean_pool takes any mask its caller builds.
    kept = ttnn.clamp(
        ttnn.sum(mask_weights, dim=2, keepdim=True, compute_kernel_config=compute_kernel_config),
        min=MASK_SUM_FLOOR,
    )
    pooled = ttnn.divide(ttnn.sum(weighted, dim=2, keepdim=True, compute_kernel_config=compute_kernel_config), kept)
    ttnn.deallocate(weighted)
    ttnn.deallocate(kept)
    return pooled


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
    0 * inf = NaN, which would undo mean_pool's own MASK_SUM_FLOOR one operator later: a fully
    padded row pools to zeros and must stay zeros rather than turning into NaN here. The
    reference gets this from F.normalize's eps.

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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-token vectors to one embedding per text, the TTNN form of reference/postprocessing.py.

    last_hidden_state (B, 1, S, H)
      -> mean_pool           -> (B, 1, 1, H)   fp32, padding excluded
      -> matryoshka_truncate -> (B, 1, 1, dim) optional, feature axis
      -> l2_normalize        -> (B, 1, 1, dim) unit norm

embed() takes a packed form where the chunk's sequences are whole tiles (packed_sum_pool, tt_config.packed_pooling):

    last_hidden_state (B, 1, S, H), S a multiple of 32, in L1
      -> packed_sum_pool     -> (1, 1, R, H)   fp32, row b = text b's masked token sum, R = B rounded up to 32
      -> matryoshka_truncate -> (1, 1, R, dim)
      -> l2_normalize        -> (1, 1, R, dim) unit norm in rows < B, zeros below

Functions rather than a class: there are no weights here. The sequence axis disappears at
mean_pool, which is why this stage needs the batch-separated (B, 1, S, H) layout rather than the
flat token axis the MoE uses.

The pooled embedding is fp32 from mean_pool on. In bfloat16, l2_normalize left the norm anywhere
in 0.9952..1.0047, which the tests' 1 - dot against the unit reference counts as error; in fp32 it
is 1.000000.
"""

from __future__ import annotations

from typing import Optional

import torch

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
    shape = embeddings.shape
    return ttnn.slice(embeddings, [0, 0, 0, 0], [shape[0], shape[1], shape[2], dim])


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


def packed_pooling_weights(attention_mask: torch.Tensor, device) -> ttnn.Tensor:
    """The (B, S) keep-mask as packed_sum_pool's weights: (1, 1, R, B*S) bf16, R = B rounded up to a tile.

    Row b holds text b's mask over its own S columns of the flattened token axis, zero elsewhere, and rows past B
    are zero. 0 and 1 rather than mean_pool's 1/count: they are exact in bfloat16, and the L2 normalize after the
    pool removes each row's scale, so a sum embeds exactly as a mean does. A row that keeps nothing sums to zeros.
    """
    batch, seqlen = attention_mask.shape
    rows = -(-batch // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    weights = torch.zeros(rows, batch * seqlen, dtype=torch.bfloat16)
    weights[:batch] = torch.block_diag(*attention_mask.to(torch.bfloat16).unbind(0))
    return ttnn.from_torch(
        weights.reshape(1, 1, rows, batch * seqlen),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _packed_program_config(k_tiles: int, m_tiles: int, n_tiles: int, grid: ttnn.CoreCoord):
    """One output tile column per core (24 cores for H = 768), the weights multicast along: M is one tile block, so
    N is the only axis to spread. The largest K block up to 32 tiles that divides K (32 measured fastest at 8x512,
    13.7 us against 15.7 at 16 and 16.1 at 64)."""
    cores_x = max(x for x in range(1, grid.x + 1) if n_tiles % x == 0 and n_tiles // x <= grid.y)
    k_block = max(block for block in range(1, 33) if k_tiles % block == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cores_x, n_tiles // cores_x),
        in0_block_w=k_block,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=m_tiles,
        out_block_w=1,
        per_core_M=m_tiles,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def packed_sum_pool(hidden_states: ttnn.Tensor, weights: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
    """Each text's masked token sum, the rows packed into one tile block: one matmul, (R x B*S) against (B*S x H).

    mean_pool runs B separate (1 x S) by (S x H) products, each a tile row of which 31 rows are padding, and reads
    the tokens from DRAM on 33 cores: 61 us at 8x512. Here the B texts share one product over the flattened token
    axis, which is a view of the tile layout when S is a whole number of tiles, on 24 cores with the weights
    multicast: 13.7 us (nomic-opt c09). The 0/1 weights are exact where mean_pool's fp32 1/count is not: against
    an fp64 reference the embeddings are 1.1e-5 off at 8x512, mean_pool's 4.3e-5.

    Args:
        hidden_states: (B, 1, S, H) bf16, S a multiple of 32, in L1. Any sharded layout (the generated norm's ND
            layout) is first copied to L1 interleaved, which the matmul reads; the copy is freed here.
        weights: packed_pooling_weights of the chunk's mask, on the same device.
        compute_kernel_config: The port's REDUCE config: HiFi4 with fp32 accumulation. LoFi drops low bits of
            the tokens (3e-4 in the embedding).

    Returns:
        ttnn.Tensor: (1, 1, R, H) fp32. Not unit norm; rows past B are zeros.
    """
    batch, _, seqlen, width = hidden_states.shape
    if seqlen % ttnn.TILE_SIZE:
        raise ValueError(f"packed_sum_pool needs whole tiles of tokens, got S = {seqlen}")
    interleaved = hidden_states
    if hidden_states.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        interleaved = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
    tokens = ttnn.reshape(interleaved, (1, 1, batch * seqlen, width))
    rows = weights.shape[-2]
    grid = hidden_states.device().compute_with_storage_grid_size()
    pooled = ttnn.matmul(
        weights,
        tokens,
        dtype=ttnn.float32,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        program_config=_packed_program_config(
            batch * seqlen // ttnn.TILE_SIZE, rows // ttnn.TILE_SIZE, width // ttnn.TILE_SIZE, grid
        ),
    )
    if interleaved is not hidden_states:
        ttnn.deallocate(interleaved)
    return pooled

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Receiver-contiguous DRAM layout: the single source of truth shared by the
Tensor Prefetcher (``tensor_prefetcher.py``) and its tests
(``tests/ttnn/unit_tests/operations/prefetcher_common.py``).

The strided receiver placement (``bank_receivers_strided``) and the round-robin
weight layout (``recv_contig_mem_config``) are a matched pair: shard ``r`` of the
``ROUND_ROBIN_1D`` ``NdShardSpec`` lands on bank ``r % num_dram_banks`` slab
``r // num_dram_banks``, and the strided placement maps that same shard to ring
position ``r``. If the two ever disagree, weights are delivered to the wrong
receiver with no error — so both definitions live here, not copied per consumer.
"""

import ttnn


def ring_pos_coord(ring_pos: int, ring_cols: int) -> ttnn.CoreCoord:
    """Map a ring position to its receiver-rectangle ``(col, row)`` in row-major order."""
    return ttnn.CoreCoord(ring_pos % ring_cols, ring_pos // ring_cols)


def bank_receivers_strided(bank_idx: int, recv_per_bank: int, num_dram_banks: int, ring_cols: int) -> ttnn.CoreRangeSet:
    """Receiver-contiguous (strided) receivers for bank ``bank_idx``.

    Bank ``b`` feeds ring positions ``[b, b + num_dram_banks, b + 2*num_dram_banks, ...]``.
    Pairs with the round-robin ``NdShardSpec`` weight layout (shard ``s`` lands on bank
    ``s % num_dram_banks`` slab ``s // num_dram_banks``) so that ring position ``r`` receives
    shard ``r`` without any host permutation. With ``ring_cols == num_dram_banks`` this is
    simply column ``b`` of the rectangle.
    """
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx + s * num_dram_banks
        cores.append(ttnn.CoreRange(ring_pos_coord(ring_pos, ring_cols), ring_pos_coord(ring_pos, ring_cols)))
    return ttnn.CoreRangeSet(cores)


def bank_receivers_contiguous(bank_idx: int, recv_per_bank: int, ring_cols: int) -> ttnn.CoreRangeSet:
    """Contiguous receiver arc matching ``CONTIGUOUS_1D`` shard distribution."""
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + s
        cores.append(ttnn.CoreRange(ring_pos_coord(ring_pos, ring_cols), ring_pos_coord(ring_pos, ring_cols)))
    return ttnn.CoreRangeSet(cores)


def recv_contig_mem_config(
    k: int,
    n: int,
    ring_size: int,
    num_dram_banks: int,
    distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
) -> ttnn.MemoryConfig:
    """Receiver-contiguous DRAM memory config for a prefetched ``(K, N)`` weight.

    Allocates the weight as an ``NdShardSpec`` with ``num_shards = ring_size`` (over-subscribed
    relative to the ``num_dram_banks`` DRAM banks) distributed round-robin, each shard
    ``(K, N // ring_size)``. Paired with the strided GCB topology, shard ``r`` (columns
    ``[r*n_per_recv, (r+1)*n_per_recv)``) is delivered to ring position ``r`` — exactly the
    weight slice the gather_in0 matmul's ring core ``r`` consumes.
    """
    assert n % ring_size == 0, f"N={n} must divide ring_size={ring_size} for receiver-contiguous layout"
    n_per_recv = n // ring_size
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            ttnn.Shape([k, n_per_recv]),
            dram_core_range_set,
            ttnn.ShardOrientation.ROW_MAJOR,
            distribution_strategy,
        ),
    )

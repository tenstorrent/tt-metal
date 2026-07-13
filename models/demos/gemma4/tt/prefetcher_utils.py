# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Prefetcher worker-sub-device constraints on the Gemma4 decode path.

After Prefetcher.init(DECODE), sender cores and worker cores are separate
sub-devices. Any program whose kernels intersect both fails with
"Programs must be executed on a single sub-device". MLP-only prefetch still
carves senders globally, so *all* decode ops must stay on the worker sub-device
(via sub_device_id, sub_core_grids, or a worker-only MemoryConfig).
"""

from __future__ import annotations

from typing import Optional

import ttnn


def pf_sub_device_id(prefetcher) -> Optional[ttnn.SubDeviceId]:
    return prefetcher.worker_sub_device_id if prefetcher is not None else None


def pf_sub_core_grids(prefetcher) -> Optional[ttnn.CoreRangeSet]:
    return prefetcher.all_worker_cores_range_set if prefetcher is not None else None


def pf_residual_mem_config(prefetcher, hidden_size: int) -> Optional[ttnn.MemoryConfig]:
    """Width-sharded L1 residual config on Prefetcher worker cores.

    Mirrors tt_transformers ``get_residual_mem_config(DECODE, prefetcher)`` but
    picks a core count that divides Gemma4's tile-width (5376/32 = 168 tiles).
    ``dynamic_worker_core_grid`` requires a multiple of 8.
    """
    if prefetcher is None:
        return None
    tiles = hidden_size // ttnn.TILE_SIZE
    # Prefer 16/8: under receivers-only worker SD, dynamic_worker_core_grid is
    # overridden to solid rectangles of those sizes (24 is not GCB-safe).
    num_cores = None
    for n in (16, 8, 24):
        if tiles % n == 0:
            num_cores = n
            break
    if num_cores is None:
        for n in range(16, 0, -8):
            if tiles % n == 0:
                num_cores = n
                break
    if num_cores is None:
        return None
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, hidden_size // num_cores),
        core_grid=prefetcher.dynamic_worker_core_grid(num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def pf_kwargs(prefetcher, *, use_sub_device: bool = True, use_sub_core_grids: bool = False) -> dict:
    """Keyword args for ttnn ops that accept sub_device_id and/or sub_core_grids.

    Binary ops accept either ``sub_device_id`` or ``sub_core_grids`` (not both).
    Prefer ``sub_device_id`` when the op supports it; use ``sub_core_grids`` for
    untilize / topk / SDPA-adjacent helpers.
    """
    if prefetcher is None:
        return {}
    if use_sub_core_grids:
        grids = pf_sub_core_grids(prefetcher)
        return {"sub_core_grids": grids} if grids is not None else {}
    if use_sub_device:
        sd = pf_sub_device_id(prefetcher)
        return {"sub_device_id": sd} if sd is not None else {}
    return {}


def pf_rect_worker_core_grid(prefetcher, num_cores: int = 32) -> Optional[ttnn.CoreGrid]:
    """Rectangular CoreGrid inside Prefetcher workers (cols 1.., rows 0..7).

    Plain ``matmul_multicore_reuse_mcast_1d`` rejects non-rectangular sub-device
    worker grids when ``sub_device_id`` is set. Attention / lm_head linears that
    are *not* ring-prefetched should omit ``sub_device_id`` and instead pass
    ``core_grid=`` this rectangle so kernels stay on the worker sub-device.
    ``num_cores`` must be a multiple of 8 (dynamic_worker_core_grid contract).
    """
    if prefetcher is None:
        return None
    if num_cores % 8 != 0 or num_cores < 8:
        num_cores = 32
    cols = num_cores // 8
    # dynamic_worker_core_grid(32) → {[1-0 - 4-7]} = CoreGrid(x=4, y=8) offset.
    # ttnn.linear ``core_grid`` is a size, not an offset — CoreGrid(x=cols, y=8)
    # places work at (0,0).. which includes sender (0,*) and (7,*). Prefer
    # returning None and using program_config with allowed_worker_cores instead.
    return ttnn.CoreGrid(x=min(cols, 6), y=8)


def pf_rect_worker_range_set(prefetcher, num_cores: int = 32) -> Optional[ttnn.CoreRangeSet]:
    """Worker CoreRangeSet for non-ring ops under Prefetcher.

    Prefer the full receiver set (64 cores, matches GCB / worker SD). Fall back
    to the solid (1,0)-(4,3) rectangle when a small grid is explicitly needed.
    """
    if prefetcher is None:
        return None
    if num_cores >= 64 and getattr(prefetcher, "all_worker_cores_range_set", None) is not None:
        return prefetcher.all_worker_cores_range_set
    if num_cores % 8 != 0 or num_cores < 8:
        num_cores = 16
    num_cores = min(num_cores, 16)
    return prefetcher.dynamic_worker_core_grid(num_cores)


def pf_plain_1d_program_config(m: int, k: int, n: int, prefetcher, num_cores: int = 16):
    """1D mcast program config on a rectangular Prefetcher worker grid.

    Used for non-ring linears (attention QKV/O, lm_head chunks) under Prefetcher.
    Caps at 16 cores — plain 1D mcast requires a rectangular grid, and the
    receivers-only worker SD is non-rectangular. Large-N callers (lm_head) must
    split columns so per-core L1 fits (see ``pf_lm_head_linear``).
    """
    import math

    if prefetcher is None:
        return None
    if num_cores % 8 != 0 or num_cores < 8:
        num_cores = 16
    num_cores = min(num_cores, 16)
    workers = pf_rect_worker_range_set(prefetcher, num_cores)
    if workers is None:
        return None
    grid = (4, 4) if num_cores == 16 else (2, 4)

    tile = ttnn.TILE_SIZE
    in0_block_w = max(1, k // (tile * num_cores))
    while in0_block_w > 1 and (k // tile) % in0_block_w != 0:
        in0_block_w -= 1
    per_core_M = max(1, math.ceil(m / tile))
    per_core_N = max(1, math.ceil(n / (tile * num_cores)))
    out_subblock_w = min(8, per_core_N)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet(set()),
        num_global_cb_receivers=1,
        untilize_out=False,
        allowed_worker_cores=workers,
    )


def pf_lm_head_linear(hidden_states, weight, prefetcher, max_columns: int = 4096):
    """Prefetcher-safe lm_head: column-split interleaved matmuls on 16 worker cores.

    Full vocab width (65536) OOMs L1 on the 16-core rectangle that plain 1D
    mcast requires. Split N into chunks ≤ ``max_columns`` and concat.
    """

    k = hidden_states.shape[-1]
    n = weight.shape[-1]
    worker_grids = prefetcher.dynamic_worker_core_grid(16)
    chunks = []
    offset = 0
    while offset < n:
        split_n = min(max_columns, n - offset)
        # Align to tile.
        if split_n % ttnn.TILE_SIZE != 0 and offset + split_n < n:
            split_n = (split_n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        w_slice = (
            weight
            if offset == 0 and split_n == n
            else ttnn.slice(weight, [0, 0, 0, offset], [1, 1, k, offset + split_n], sub_core_grids=worker_grids)
        )
        pc = pf_plain_1d_program_config(hidden_states.shape[-2], k, split_n, prefetcher, num_cores=16)
        chunk = (
            ttnn.linear(hidden_states, w_slice, program_config=pc)
            if pc is not None
            else ttnn.linear(hidden_states, w_slice)
        )
        chunks.append(chunk)
        if w_slice is not weight:
            w_slice.deallocate(True)
        offset += split_n
    if len(chunks) == 1:
        return chunks[0]
    try:
        out = ttnn.concat(chunks, dim=-1, sub_core_grids=worker_grids)
    except TypeError:
        out = ttnn.concat(chunks, dim=-1)
    for c in chunks:
        c.deallocate(True)
    return out

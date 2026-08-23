# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""paged_update_cache with page-ALIASED rows (the spec batch-alias verify shape).

The update kernel read-modify-writes the WHOLE destination cache tile
(untilize -> splice one row -> retilize -> full-tile write), so several rows of
one batched call targeting the same tile race and lose all but the last row —
the 53894 two-family divergence. The fix runs one masked call per candidate
index (-1 skips a row), so no call writes two rows of one tile; op barriers
serialize successive calls. This test gates the fix bitwise and reports the
batched hazard (informational: the race outcome is scheduling-dependent).
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]

USERS, W = 2, 4  # 2 real users x 4 aliased candidate rows
R = USERS * W
NKV, BS, HD = 1, 64, 128
BLOCKS_PU = 2


def _fresh_cache(device):
    return ttnn.from_torch(
        torch.zeros(USERS * BLOCKS_PU, NKV, BS, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def _inputs(device):
    torch.manual_seed(30)
    rows = torch.randn(1, R, NKV, HD, dtype=torch.float32) * 0.5
    k_p = torch.nn.functional.pad(rows, (0, 0, 0, 32 - NKV))  # head dim padded to a tile
    shard = ttnn.create_sharded_memory_config(
        shape=(32, HD),
        core_grid=ttnn.CoreGrid(x=R, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    k_d = ttnn.from_torch(k_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_sh = ttnn.to_memory_config(k_d, shard)
    ttnn.deallocate(k_d)
    # Pseudo-user rows: user u's W rows share ITS page-table row (block aliasing).
    pt_rows = torch.stack(
        [torch.arange(u * BLOCKS_PU, (u + 1) * BLOCKS_PU, dtype=torch.int32) for u in range(USERS) for _ in range(W)]
    )
    pt = ttnn.from_torch(pt_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Positions 10..13 for every user: same block, same destination tile.
    poss = [10 + t for _u in range(USERS) for t in range(W)]
    return rows.to(torch.bfloat16).float(), k_sh, pt, poss


def _pos_tensor(device, vals):
    return ttnn.from_torch(
        torch.tensor(vals, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _read_rows(cache, poss):
    ct = ttnn.to_torch(cache).float()  # [blocks, NKV, BS, HD]
    got = []
    for u in range(USERS):
        for t in range(W):
            p = poss[u * W + t]
            got.append(ct[u * BLOCKS_PU + p // BS, 0, p % BS])
    return torch.stack(got)


def test_paged_update_masked_sequential_exact(device):
    """The fix path: W masked sequential updates land every aliased row exactly."""
    rows_ref, k_sh, pt, poss = _inputs(device)
    cache = _fresh_cache(device)
    for t in range(W):
        masked = [p if (i % W) == t else -1 for i, p in enumerate(poss)]
        upd = _pos_tensor(device, masked)
        ttnn.experimental.paged_update_cache(cache, k_sh, update_idxs_tensor=upd, page_table=pt)
        ttnn.deallocate(upd)
    got = _read_rows(cache, poss)
    ref = rows_ref.reshape(R, NKV, HD)[:, 0]
    assert torch.equal(got, ref), f"masked-sequential aliased update wrong: max abs diff {(got - ref).abs().max()}"


def test_paged_update_batched_aliased_hazard(device):
    """Informational: ONE batched call over aliased rows loses same-tile rows
    (whole-tile RMW). Logged, not asserted — the race outcome is scheduling-
    dependent; if this ever lands all rows, the op changed and the masked
    workaround can be revisited."""
    rows_ref, k_sh, pt, poss = _inputs(device)
    cache = _fresh_cache(device)
    upd = _pos_tensor(device, poss)
    ttnn.experimental.paged_update_cache(cache, k_sh, update_idxs_tensor=upd, page_table=pt)
    ttnn.deallocate(upd)
    got = _read_rows(cache, poss)
    ref = rows_ref.reshape(R, NKV, HD)[:, 0]
    bad = [i for i in range(R) if not torch.equal(got[i], ref[i])]
    logger.info(f"batched aliased update: {len(bad)}/{R} rows lost/stale (expected nonzero): rows {bad}")

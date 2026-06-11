# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validation: per (user, last_valid_token), print the ACTUAL per-chip local memory indices
the device op zeroed, and check them against the expected block-cyclic window.

Cache: global seq 10240 (2 slabs of chunk=5120), 61 layers, 2 users, filled with ones.
'memory index' = per-chip local row index (e.g. global token 5121 -> (sp0, local 641))."""
import math

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

CSG = 5120
SEQ = 10240  # global cache length (2 slabs); seq_local = 1280
NUM_LAYERS = 61
NUM_USERS = 2
LAYER = 0  # all cases use layer 0

# (user, last_valid_token == valid_global)
CASES = [(0, 50), (0, 639), (0, 850), (0, 5000), (0, 6000), (0, 9000), (1, 5000), (1, 10000)]


def _ranges(idxs):
    """sorted list of ints -> list of (start,end) inclusive contiguous ranges."""
    out = []
    for i in sorted(idxs):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.timeout(0)
def test_zero_op_mem_validate(mesh_device):
    sp_axis = 0
    sp = 8
    tp = 4
    kvpe = 64
    seq_local = SEQ // sp  # 1280

    cache = init_kvpe_cache(
        kvpe, mesh_device, SEQ, [8, 4], sp_axis, num_kvpe_cache_layers=NUM_LAYERS, num_users=NUM_USERS
    )
    ones = torch.ones(1, 1, seq_local, kvpe, dtype=torch.bfloat16)
    tt_ones = ttnn.from_torch(
        ones,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    for u, v in CASES:
        target_batch = u * NUM_LAYERS + LAYER
        # fresh ones in the target slot
        ttnn.fill_cache(cache, tt_ones, target_batch, update_idx=0)
        ttnn.synchronize_device(mesh_device)
        # run the op
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(cache, u, LAYER, NUM_LAYERS, v, CSG, sp_axis, 128)
        ttnn.synchronize_device(mesh_device)

        # read back the target slot; collect (chip, local) cells that became zero
        zeroed = []
        for di, dt in enumerate(ttnn.get_device_tensors(cache)):
            if di % tp != 0:
                continue
            chip = di // tp
            rows = ttnn.to_torch(dt).float()[target_batch, 0, :, :].mean(dim=-1)
            for lr in range(seq_local):
                if rows[lr].item() < 0.5:
                    zeroed.append((chip, lr))

        ceil_v = math.ceil(v / 128) * 128
        # expected: every global token in [v, ceil_v) mapped to (chip, local)
        expected = set()
        for g in range(v, ceil_v):
            chip = (g % CSG) // (CSG // sp)
            local = (g // CSG) * (CSG // sp) + (g % CSG) % (CSG // sp)
            expected.add((chip, local))

        logger.info(f"===== user{u}  last_valid={v}  -> window global [{v}, {ceil_v})  ({ceil_v - v} tokens) =====")
        by_chip = {}
        for c, lr in zeroed:
            by_chip.setdefault(c, []).append(lr)
        for c in sorted(by_chip):
            rngs = ", ".join(f"{a}..{b}" if a != b else f"{a}" for a, b in _ranges(by_chip[c]))
            logger.info(f"    zeroed -> sp{c} local[{rngs}]   ({len(by_chip[c])} cells)")
        match = set(zeroed) == expected
        logger.info(f"    matches expected block-cyclic window: {match}")
        assert match, f"user{u} v{v}: device-zeroed cells != expected (got {len(zeroed)}, exp {len(expected)})"

    logger.success("all cases validated")

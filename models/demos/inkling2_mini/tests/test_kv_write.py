# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Decode ring write: one paged_fused_update_cache (k on core (0,0), v on (0,1)) vs two paged_update_cache
calls, as tt/attention_fast.py did: caches bit-equal after a run of slot writes (BFP8 and bf16 rings), plus
traced replay time of 48 writes each.
"""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.attention import HEAD_DIM, N_KV, WINDOW

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


def shard(core):
    return ttnn.create_sharded_memory_config(
        shape=(32, HEAD_DIM),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def replay_us(dev, tid, reps=50):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e6


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_kv_write(device, cache_dtype):
    dev = device
    g = torch.Generator().manual_seed(0)
    s0, s1 = shard((0, 0)), shard((0, 1))
    init = [torch.randn(1, N_KV, WINDOW, HEAD_DIM, generator=g) for _ in range(2)]
    caches = {
        m: [ttnn.from_torch(t, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=dev) for t in init]
        for m in ("old", "new")
    }

    def row(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def slot(p):
        return ttnn.from_torch(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32, device=dev)

    def old(c, k, v, idx):
        ttnn.experimental.paged_update_cache(c[0], ttnn.to_memory_config(k, s0), update_idxs_tensor=idx)
        ttnn.experimental.paged_update_cache(c[1], ttnn.to_memory_config(v, s0), update_idxs_tensor=idx)

    def new(c, k, v, idx):
        ttnn.experimental.paged_fused_update_cache(
            c[0], ttnn.to_memory_config(k, s0), c[1], ttnn.to_memory_config(v, s1), update_idxs_tensor=idx
        )

    for p in (0, 1, 15, 16, 31, 32, 517, 1000, 1023, 1):
        k = row(torch.randn(1, 1, N_KV, HEAD_DIM, generator=g) * 3)
        v = row(torch.randn(1, 1, N_KV, HEAD_DIM, generator=g))
        idx = slot(p)
        old(caches["old"], k, v, idx)
        new(caches["new"], k, v, idx)
    for a, b in zip(caches["old"], caches["new"]):
        ta, tb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        assert torch.equal(ta, tb), int((ta != tb).sum())

    k = row(torch.randn(1, 1, N_KV, HEAD_DIM, generator=g))
    v = row(torch.randn(1, 1, N_KV, HEAD_DIM, generator=g))
    idx = slot(7)
    times = {}
    for label, fn in (("old", old), ("new", new)):
        c = caches[label]
        fn(c, k, v, idx)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(48):
            fn(c, k, v, idx)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        times[label] = replay_us(dev, tid) / 48
        ttnn.release_trace(dev, tid)
    print(f"KV_WRITE_US {cache_dtype} per layer old {times['old']:.1f} new {times['new']:.1f}")

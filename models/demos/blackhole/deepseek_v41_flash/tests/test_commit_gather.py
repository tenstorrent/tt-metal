# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SpecCompressedAttention.commit with DSV41_SPEC_COMMIT_GATHER=1: ``ttnn.gather`` of block index m[u] of ``cs_block`` [U, n, 1024] fp32 row-major equals the one-hot multiply + sum
of the default path (bit for bit), timed (traced, 20 calls) against it."""

import time

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 20_000_000}, id="plain")],
    indirect=True,
)
@pytest.mark.parametrize("U,n", [(1, 6), (2, 4), (2, 6), (4, 6)])
@torch.no_grad()
def test_commit_gather(mesh_device, U, n):
    md = mesh_device
    torch.manual_seed(U * 10 + n)
    rep = ttnn.ReplicateTensorToMesh(md)
    cs = torch.randn(U, n, 1024).float()
    m = torch.randint(0, n, (U,))
    cs_d = ttnn.from_torch(
        cs,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    oh = torch.nn.functional.one_hot(m, n).float().reshape(U, n, 1)
    oh_d = ttnn.from_torch(
        oh,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    mi_d = ttnn.from_torch(
        m.reshape(U, 1).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    def old():
        prod = ttnn.multiply(ttnn.to_layout(cs_d, ttnn.TILE_LAYOUT), ttnn.to_layout(oh_d, ttnn.TILE_LAYOUT))
        sel = ttnn.to_layout(ttnn.sum(prod, dim=1, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(sel, [1, 1, U, 1024]), ttnn.TILE_LAYOUT)

    def new():
        idx3 = ttnn.repeat(ttnn.reshape(mi_d, [U, 1, 1]), [1, 1, 1024])
        sel = ttnn.gather(cs_d, 1, idx3)
        return ttnn.to_layout(ttnn.reshape(sel, [1, 1, U, 1024]), ttnn.TILE_LAYOUT)

    want = cs[torch.arange(U), m].reshape(U, 1024)
    outs = {}
    for name, fn in (("old", old), ("new", new)):
        o = fn()
        ttnn.synchronize_device(md)
        outs[name] = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(-1, 1024)[:U].float()
        assert torch.equal(outs[name], want), name
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(20):
            fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        print(f"COMMIT_GATHER U={U} n={n} {name} {(time.perf_counter() - t0) / 100 * 1e6:.1f} us per call", flush=True)
        ttnn.release_trace(md, tid)
    assert torch.equal(outs["old"], outs["new"])

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-host microbench of the per-round upload / read-back calls (shapes of a B=4 k=3 round: T=16 block rows, rows [T,1,1,12288] bf16, ctrl [T,32] int32, pack ~700 int32).
Prints HO lines (ms per call, mean of 200; DSV41_HOSTOPT_BENCH_SLEEP_MS between calls mimics the host sleeping in the device wait, default 50)."""

import os
import time

import pytest
import torch

import ttnn


def _bench(fn, reps=200):
    slp = float(os.environ.get("DSV41_HOSTOPT_BENCH_SLEEP_MS", "50")) / 1e3
    ts = []
    for _ in range(reps + 10):
        time.sleep(slp)
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = sorted(ts[10:])
    return sum(ts) / len(ts), ts[int(0.95 * len(ts))]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_hostopt_dev(mesh_device):
    md = mesh_device
    rows, cols = 4, 8
    T = int(os.environ.get("DSV41_HOSTOPT_BENCH_T", "16"))
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    ctrl = torch.zeros(T, 32, dtype=torch.int32)
    cat = torch.randn(T, 1, 1, 12288).to(torch.bfloat16)
    force = torch.full((rows * 4, 1), -1.0)
    d_ctrl = ttnn.from_torch(ctrl, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp, device=md)
    d_rows = ttnn.from_torch(cat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp, device=md)
    d_force = ttnn.from_torch(force, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp, device=md)
    h_ctrl = ttnn.from_torch(ctrl, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
    h_rows = ttnn.from_torch(cat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
    h_force = ttnn.from_torch(force, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp)
    N = 704
    host = torch.arange(rows * cols * N, dtype=torch.int32).reshape(rows * cols, 1, 1, N)
    pk = ttnn.from_torch(
        host,
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
    )
    ttnn.synchronize_device(md)
    out = {}

    def run(name, fn):
        out[name] = _bench(fn)
        print(f"HO {name}: mean {out[name][0]:.3f} p95 {out[name][1]:.3f} ms", flush=True)

    run(
        "from_torch ctrl", lambda: ttnn.from_torch(ctrl, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
    )
    run(
        "from_torch rows",
        lambda: ttnn.from_torch(cat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp),
    )
    run("from_torch force", lambda: ttnn.from_torch(force, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp))
    run("copy ctrl", lambda: ttnn.copy_host_to_device_tensor(h_ctrl, d_ctrl))
    run("copy rows", lambda: ttnn.copy_host_to_device_tensor(h_rows, d_rows))
    run("copy force", lambda: ttnn.copy_host_to_device_tensor(h_force, d_force))

    def both():
        ttnn.copy_host_to_device_tensor(h_ctrl, d_ctrl)
        ttnn.copy_host_to_device_tensor(h_rows, d_rows)

    run("copy ctrl+rows", both)

    def issue():
        return [ttnn.from_device(ttnn.get_device_tensors(pk)[r * cols], blocking=False) for r in range(rows)]

    run("rb issue x4", issue)

    def issue_sync():
        hs = issue()
        ttnn.synchronize_device(md)
        return hs

    run("rb issue+sync", issue_sync)
    hs = issue_sync()
    run("rb ttnn.to_torch x4 + cat", lambda: torch.cat([ttnn.to_torch(h).reshape(1, -1) for h in hs]).long())
    alts = {}
    if hasattr(hs[0], "to_torch"):
        alts["rb h.to_torch x4 + cat"] = lambda: torch.cat([h.to_torch().reshape(1, -1) for h in hs]).long()
    for name, fn in alts.items():
        try:
            ref = torch.cat([ttnn.to_torch(h).reshape(1, -1) for h in hs]).long()
            assert torch.equal(ref, fn())
            run(name, fn)
        except Exception as e:  # noqa: BLE001
            print(f"HO {name}: ERR {str(e)[:200]}", flush=True)
    run("sync only (idle queue)", lambda: ttnn.synchronize_device(md))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_host_tensor_alias(mesh_device):
    """Does a host tensor built by ``from_torch`` (mesh mapper) see later in-place writes to its source torch tensor? (if yes a round could refill one persistent torch buffer and re-copy the same host tensor)"""
    md = mesh_device
    rows, cols = 4, 8
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    for name, shape, dt, tdt, lay in (
        ("ctrl int32", (16, 32), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT),
        ("rows bf16", (16, 1, 1, 12288), ttnn.bfloat16, torch.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ):
        src = torch.zeros(*shape, dtype=tdt)
        dev = ttnn.from_torch(src, dtype=dt, layout=lay, mesh_mapper=mp, device=md)
        host = ttnn.from_torch(src, dtype=dt, layout=lay, mesh_mapper=mp)
        src.fill_(3)
        ttnn.copy_host_to_device_tensor(host, dev)
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(ttnn.get_device_tensors(dev)[0]).reshape(-1)[:4].tolist()
        print(f"HO alias {name}: device sees {got} after the in-place write of 3 (3 = aliased, 0 = copied)", flush=True)

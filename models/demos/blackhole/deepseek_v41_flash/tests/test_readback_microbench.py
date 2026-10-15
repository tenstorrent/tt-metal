# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host cost of reading the small per-round ``pack`` tensor (one row-leader shard per mesh row, ~700 int32): the current loop of per-device ``ttnn.to_torch`` calls
against a single whole-mesh host copy. Prints RB lines (ms per read, mean of 200)."""

import time

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_readback(mesh_device):
    md = mesh_device
    rows, cols = 4, 8
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

    def cur():
        return torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(pk)[r * cols]).reshape(1, -1) for r in range(rows)]
        ).long()

    def whole():
        t = ttnn.to_torch(pk, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(rows * cols, -1)
        return t[::cols].long()

    def host_copy():
        h = ttnn.from_device(pk)  # one blocking copy of every shard
        return torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(h)[r * cols]).reshape(1, -1) for r in range(rows)]
        ).long()

    def nonblock():
        hs = [ttnn.from_device(ttnn.get_device_tensors(pk)[r * cols], blocking=False) for r in range(rows)]
        ttnn.synchronize_device(md)
        return torch.cat([ttnn.to_torch(h).reshape(1, -1) for h in hs]).long()

    ref = cur()
    for name, fn in (
        ("current_loop", cur),
        ("whole_mesh_composer", whole),
        ("from_device_whole", host_copy),
        ("nonblocking_rows", nonblock),
    ):
        try:
            out = fn()
            ok = bool(torch.equal(out, ref))
            for _ in range(10):
                fn()
            t = time.perf_counter()
            for _ in range(200):
                fn()
            dt = (time.perf_counter() - t) / 200 * 1e3
            print(f"RB {name}: {dt:.3f} ms  identical={ok}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RB {name}: ERR {str(e)[:200]}", flush=True)

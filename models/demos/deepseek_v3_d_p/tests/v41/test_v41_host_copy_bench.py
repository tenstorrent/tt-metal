# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The host-to-device cost of the decode step's small inputs. ``test_v41_host_copy_bench``:
for each dtype / layout / shape variant the median and p90 of 60 ``copy_host_to_device_tensor`` into a fixed
mesh
buffer (one shard per rank, replicated over TP, as the step stages them), of the host tensor's construction
(``from_torch`` with the mesh mapper, or ``ttnn.Tensor`` per rank + ``from_host_shards``) and of a device
sync.
``test_v41_first_copy_after_step``: the step's sequence (a captured trace replayed non-blocking, a device
sync, a
read of a device tensor, then the copies) and the cost of the first copy after it against the second, with the
replay, the sync and the read left out one at a time (the served step's first copy costs 1.7 to 2.0 ms, the
next
ones 0.2 to 0.3, the isolated copy 0.09). No weights; ~2 minutes on the Galaxy. ``V41_COPY_BENCH_OUT``: the
JSON."""

import json
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes

VARIANTS = [
    # name, torch dtype, ttnn dtype, layout, per-rank shape
    ("ints rm u32 [14,32]", torch.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 14, 32)),
    ("one rm u32 [1,12800]", torch.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 1, 12800)),
    ("floats rm bf16 [2,32]", torch.float32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 2, 32)),
    ("rope tile bf16 [32,384]", torch.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, 32, 384)),
    ("one tile u32 [64,384]", torch.int32, ttnn.uint32, ttnn.TILE_LAYOUT, (1, 1, 64, 384)),
    ("one tile u32 [32,416]", torch.int32, ttnn.uint32, ttnn.TILE_LAYOUT, (1, 1, 32, 416)),
    ("ints rm u32 [1,448]", torch.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 1, 448)),
    ("ints rm u16 [14,32]", torch.int16, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 14, 32)),
    ("ints tile u32 [32,32]", torch.int32, ttnn.uint32, ttnn.TILE_LAYOUT, (1, 1, 32, 32)),
    ("engram rm u16 [192,160]", torch.int16, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 192, 160)),
]


def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {"median_us": 1e6 * statistics.median(s), "p90_us": 1e6 * s[int(0.9 * (len(s) - 1))], "n": len(s)}


def _dump(name: str, out: dict) -> None:
    path = os.environ.get("V41_COPY_BENCH_OUT")
    if path:
        base, ext = os.path.splitext(path)
        json.dump(out, open(f"{base}.{name}{ext}", "w"), indent=1)


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_host_copy_bench(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, None))
    out = {}
    for name, tdtype, dtype, layout, shape in VARIANTS:
        full = (sp,) + shape[1:]
        host_t = torch.zeros(full, dtype=tdtype)

        def via_from_torch():
            return ttnn.from_torch(host_t, dtype=dtype, layout=layout, mesh_mapper=mapper)

        def via_shards():
            parts = []
            for r in range(sp):
                t = ttnn.Tensor(host_t[r : r + 1], dtype)
                if layout == ttnn.TILE_LAYOUT:
                    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
                parts.extend([t] * tp)
            return ttnn.from_host_shards(parts, ttnn.MeshShape(sp, tp))

        h = via_from_torch()
        buf = ttnn.to_device(h, mesh_device)
        rec = {"shape": list(shape), "dtype": str(dtype), "layout": str(layout)}
        for label, build in (("from_torch", via_from_torch), ("shards", via_shards)):
            builds, copies = [], []
            for i in range(70):
                t0 = time.perf_counter()
                h = build()
                t1 = time.perf_counter()
                ttnn.copy_host_to_device_tensor(h, buf)
                t2 = time.perf_counter()
                if i >= 10:
                    builds.append(t1 - t0)
                    copies.append(t2 - t1)
            ttnn.synchronize_device(mesh_device)
            rec[label] = {"build": _stats(builds), "copy": _stats(copies)}
        syncs = []
        for _ in range(20):
            t0 = time.perf_counter()
            ttnn.synchronize_device(mesh_device)
            syncs.append(time.perf_counter() - t0)
        rec["sync"] = _stats(syncs)
        out[name] = rec
        print(
            f"{name:26s} from_torch build {rec['from_torch']['build']['median_us']:7.0f} us copy "
            f"{rec['from_torch']['copy']['median_us']:7.0f} us | shards build {rec['shards']['build']['median_us']:7.0f}"
            f" us copy {rec['shards']['copy']['median_us']:7.0f} us | sync {rec['sync']['median_us']:6.0f} us"
        )
    _dump("variants", out)


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_first_copy_after_step(mesh_device, device_params):
    """A captured trace of ``V41_COPY_BENCH_OPS`` (default 200) small ops replayed non-blocking, a device
    sync, a
    read of a device tensor (as the step reads its ids), then four copies of the step's shapes: the first
    copy's
    cost against the others, with each of the replay, the sync and the read left out in turn."""
    sp, tp = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, None))
    n_ops = int(os.environ.get("V41_COPY_BENCH_OPS", "200"))
    x = ttnn.from_torch(
        torch.randn(sp, 1, 32, 1024),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
    )

    def graph(t):
        for _ in range(n_ops):
            t = ttnn.add(t, 1.0)
        return t

    graph(x)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y = graph(x)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ids = ttnn.from_torch(
        torch.zeros(sp, 1, 256, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
    )
    shapes = [
        ("ints", torch.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 14, 32)),
        ("floats", torch.float32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 2, 32)),
        ("rope", torch.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, 32, 384)),
        ("one", torch.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 1, 12800)),
    ]
    hosts = {
        name: ttnn.from_torch(
            torch.zeros((sp,) + shape[1:], dtype=tdtype), dtype=dtype, layout=layout, mesh_mapper=mapper
        )
        for name, tdtype, dtype, layout, shape in shapes
    }
    bufs = {name: ttnn.to_device(h, mesh_device) for name, h in hosts.items()}
    out = {}
    for case in (
        "replay+sync+read",
        "replay+sync",
        "sync+read",
        "read",
        "nothing",
        "replay+sync+read, sync before copies",
    ):
        times = {name: [] for name in hosts}
        replays = []
        for i in range(40):
            t0 = time.perf_counter()
            if "replay" in case:
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            if "sync" in case.split(",")[0]:
                ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            if "read" in case:
                ttnn.to_torch(ttnn.get_device_tensors(ids)[0])
            if "sync before copies" in case:
                ttnn.synchronize_device(mesh_device)
            for name, h in hosts.items():
                t2 = time.perf_counter()
                ttnn.copy_host_to_device_tensor(h, bufs[name])
                if i >= 5:
                    times[name].append(time.perf_counter() - t2)
            if i >= 5:
                replays.append(t1 - t0)
        ttnn.synchronize_device(mesh_device)
        rec = {name: _stats(v) for name, v in times.items()}
        rec["replay_and_sync"] = _stats(replays)
        out[case] = rec
        print(
            f"{case:40s} replay+sync {rec['replay_and_sync']['median_us']:7.0f} us | "
            + " | ".join(f"{name} {rec[name]['median_us']:6.0f} (p90 {rec[name]['p90_us']:6.0f})" for name in hosts)
        )
    ttnn.release_trace(mesh_device, tid)
    _dump("firstcopy", out)

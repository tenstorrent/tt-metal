# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import torch

import ttnn


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        # A zero denominator means one side is CONSTANT after centering. The reference here is
        # `randn @ w`, which is never constant, so this is the DEVICE output being degenerate --
        # exactly what a garbled bfloat4_b/LoFi matmul, a deallocated buffer or a silently failed
        # op produces. Returning 1.0 called that a perfect match, and matmul_sweep.pick_best then
        # selected the fastest BROKEN config as its PCC-gated recommendation.
        return 0.0
    return float((a @ b).item() / denom)


def _mesh_axes(mesh_device) -> list:
    """The mesh's (rows, cols), or [] when the shape cannot be read."""
    try:
        shape = mesh_device.shape
        return [int(shape[0]), int(shape[1])]
    except Exception:  # noqa: BLE001
        try:
            return [1, int(mesh_device.get_num_devices())]
        except Exception:  # noqa: BLE001
            return []


def all_gather_full(mesh_device, t, dim: int = -1):
    """all_gather across EVERY device of the mesh, whatever its shape.

    A bare ``ttnn.all_gather(t, dim)`` only works on a 1-D mesh. With 1-D fabric on a 2-D mesh ttnn
    cannot infer which axis to gather along and raises ("1D fabric on a 2D mesh_device requires
    cluster_axis to be set"), so this module worked on a (1,4) mesh and failed on the (2,2) one --
    including in a real run, since the mesh shape comes from the model, not from us.

    A degenerate mesh gathers along its one real axis. A true 2-D mesh gathers along each axis in
    turn: within a row first, then across rows. Shards are placed in row-major device order, so the
    two stages concatenate in shard order and the result matches the dense reference.
    """
    axes = _mesh_axes(mesh_device)
    if not axes:
        return ttnn.all_gather(t, dim=dim)
    rows, cols = axes[0], axes[1]
    if rows == 1 and cols == 1:
        return t
    if rows == 1:
        return ttnn.all_gather(t, dim=dim, cluster_axis=1)
    if cols == 1:
        return ttnn.all_gather(t, dim=dim, cluster_axis=0)
    inner = ttnn.all_gather(t, dim=dim, cluster_axis=1)
    return ttnn.all_gather(inner, dim=dim, cluster_axis=0)


def dense_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w


def column_fracture_all_gather(mesh_device, x: torch.Tensor, w: torch.Tensor, tp: int) -> torch.Tensor:
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    y_local = ttnn.matmul(x_tt, w_tt)
    y_full = all_gather_full(mesh_device, y_local, dim=-1)
    out = ttnn.to_torch(y_full, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[: x.shape[0]]


def verify_fracture(mesh_device, m: int, k: int, n: int, tp: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n)
    ref = dense_reference(x, w)
    got = column_fracture_all_gather(mesh_device, x, w, tp)
    got = got[:, : ref.shape[1]]
    pcc = _pcc(ref, got)
    return {"pcc": pcc, "ref_shape": list(ref.shape), "got_shape": list(got.shape), "tp": tp}


def _dense_on_one_chip(mesh_device, x, w):
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return x_tt, w_tt


def _time_ms(fn, iters: int = 5) -> float:
    fn()
    t0 = time.monotonic()
    for _ in range(iters):
        fn()
    return (time.monotonic() - t0) * 1000.0 / iters


def bench_fracture(mesh_device, m: int, k: int, n: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n)
    xd, wd = _dense_on_one_chip(mesh_device, x, w)

    def dense():
        y = ttnn.matmul(xd, wd)
        ttnn.synchronize_device(mesh_device)
        return y

    xf = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    wf = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    def frac():
        y = ttnn.matmul(xf, wf)
        g = all_gather_full(mesh_device, y, dim=-1)
        ttnn.synchronize_device(mesh_device)
        return g

    d_ms = _time_ms(dense)
    f_ms = _time_ms(frac)
    return {"dense_ms": d_ms, "frac_ms": f_ms, "speedup": d_ms / f_ms if f_ms else 0.0, "m": m, "k": k, "n": n}


def _time_degree(mesh_device, x, w, degree: int, num_devices: int) -> float:
    xr = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    if degree <= 1:
        wr = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        def run():
            y = ttnn.matmul(xr, wr)
            ttnn.synchronize_device(mesh_device)
            return y

        return _time_ms(run)
    ws = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    def run():
        y = ttnn.matmul(xr, ws)
        g = all_gather_full(mesh_device, y, dim=-1)
        ttnn.synchronize_device(mesh_device)
        return g

    return _time_ms(run)


def sweep_degrees(mesh_device, m: int, k: int, n: int, candidates=None) -> dict:
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    num = rows * cols
    if candidates is None:
        candidates = [1, num]
    candidates = [d for d in dict.fromkeys(candidates) if d == 1 or d == num]
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n)
    timings = {}
    for d in candidates:
        timings[d] = _time_degree(mesh_device, x, w, d, num)
    best_tp = min(timings, key=timings.get)
    return {"best_tp": best_tp, "timings_ms": timings, "m": m, "k": k, "n": n}

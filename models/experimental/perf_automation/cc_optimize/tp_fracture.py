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
        return 1.0
    return float((a @ b).item() / denom)


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
    y_full = ttnn.all_gather(y_local, dim=-1)
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


def _time_ms(fn, iters: int = 20) -> float:
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
        g = ttnn.all_gather(y, dim=-1)
        ttnn.synchronize_device(mesh_device)
        return g

    d_ms = _time_ms(dense)
    f_ms = _time_ms(frac)
    return {"dense_ms": d_ms, "frac_ms": f_ms, "speedup": d_ms / f_ms if f_ms else 0.0, "m": m, "k": k, "n": n}

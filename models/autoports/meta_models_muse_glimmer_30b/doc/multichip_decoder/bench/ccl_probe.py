# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure the collectives this stage can choose between, at this model's shapes.

The multichip decoder has to pick a residual/collective topology *before* the
final path is coded, and the choice is a byte/op-count trade-off (a 4-device
ring all-reduce moves 1.5x the tensor; a reduce-scatter or all-gather moves
0.75x, but two of them are two dispatches).  On a 4-chip Blackhole mesh at
decode shapes both terms are small, so the ranking has to be measured rather
than derived.

Usage::

    python .../bench/ccl_probe.py --mesh 1x4 --out logs/ccl_probe.log
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn

HIDDEN = 6656
ATTN = 4096
INTERMEDIATE = 19968


def _mesh_shape(text: str) -> tuple[int, int]:
    a, b = text.lower().split("x")
    return int(a), int(b)


def _bench(fn, iters: int, warmup: int = 3) -> float:
    """Min wall time per call over ``iters`` rounds, warmed."""
    for _ in range(warmup):
        out = fn()
        if out is not None:
            _dealloc(out)
    ttnn.synchronize_device(fn.mesh)
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(fn.mesh)
        best = min(best, time.perf_counter() - start)
        if out is not None:
            _dealloc(out)
    return best * 1e6  # microseconds


def _dealloc(out) -> None:
    if isinstance(out, (list, tuple)):
        for tensor in out:
            ttnn.deallocate(tensor)
    else:
        ttnn.deallocate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="1x4")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--topology", default="ring", choices=("ring", "linear", "none"))
    parser.add_argument("--fabric", default="1d_ring", choices=("1d_ring", "1d", "2d", "2d_torus_xy", "none"))
    args = parser.parse_args()

    rows_shape = _mesh_shape(args.mesh)
    fabric = {
        "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
        "1d": ttnn.FabricConfig.FABRIC_1D,
        "2d": ttnn.FabricConfig.FABRIC_2D,
        "2d_torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "none": None,
    }[args.fabric]
    if fabric is not None:
        ttnn.set_fabric_config(fabric)
    print(f"FABRIC {args.fabric}")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*rows_shape))
    devices = mesh.get_num_devices()
    topology = {
        "ring": ttnn.Topology.Ring,
        "linear": ttnn.Topology.Linear,
        "none": None,
    }[args.topology]
    print(f"MESH {args.mesh} devices={devices} topology={args.topology}")

    try:
        for rows in (32, 128, 1024, 8192):
            for width, label in ((HIDDEN, "hidden"), (ATTN, "attn"), (INTERMEDIATE, "inter")):
                torch_full = torch.randn(1, 1, rows, width, dtype=torch.bfloat16)
                # A "fractured" tensor: each device owns width/devices of the row.
                fractured = ttnn.from_torch(
                    torch_full,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                # A "partial" tensor: every device holds a full-width partial sum.
                partial = ttnn.from_torch(
                    torch_full,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                bytes_full = rows * width * 2

                def ag():
                    kwargs = {} if topology is None else {}
                    return ttnn.all_gather(fractured, dim=3, **kwargs)

                def rs():
                    kwargs = {} if topology is None else {"topology": topology}
                    return ttnn.reduce_scatter(partial, dim=3, **kwargs)

                def ar():
                    kwargs = {} if topology is None else {"topology": topology}
                    return ttnn.all_reduce(partial, **kwargs)

                def bcast():
                    kwargs = {} if topology is None else {"topology": topology}
                    return ttnn.all_broadcast(fractured, **kwargs)

                for name, fn, moved in (
                    ("all_gather[fractured->full]", ag, bytes_full * (devices - 1) / devices),
                    ("reduce_scatter[partial->fractured]", rs, bytes_full * (devices - 1) / devices),
                    ("all_reduce[partial->full]", ar, 2 * bytes_full * (devices - 1) / devices),
                    ("all_broadcast[fractured]", bcast, bytes_full * (devices - 1) / devices),
                ):
                    fn.mesh = mesh
                    try:
                        us = _bench(fn, args.iters)
                        gbs = moved / us / 1e3 if us else 0.0
                        print(
                            f"CCL rows={rows:5d} {label:6s} w={width:6d} {name:36s} "
                            f"{us:9.2f} us  moved={moved/1024:9.1f} KiB  {gbs:7.2f} GB/s"
                        )
                    except Exception as exc:  # noqa: BLE001 - probe records failures
                        print(
                            f"CCL rows={rows:5d} {label:6s} w={width:6d} {name:36s} FAIL {type(exc).__name__}: {str(exc)[:200]}"
                        )
                ttnn.deallocate(fractured)
                ttnn.deallocate(partial)
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

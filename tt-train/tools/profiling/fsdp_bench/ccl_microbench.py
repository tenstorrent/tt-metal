# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark of the exact CCL primitives ttml.fsdp uses.

For every FSDP-managed parameter shape of a model we time, on a line mesh
``[N, 1]`` with axis ``fsdp``:

  * ``ttml.core.distributed.all_gather(shard, shard_dim, axis)``  (pre_forward / backward_pre)
  * ``ttml.core.distributed.reduce_scatter(full, shard_dim, axis)``  (backward_post)
  * ``ttnn.multiply(shard, 1/N)``  (mean scaling in backward_post)

Three numbers per (op, shape):
  * ``dev_us``  : device time of one op, measured with a device sync around a
                  batch of ``--iters`` back-to-back launches (steady state).
  * ``lat_us``  : latency of a single op incl. launch, sync before & after.
  * ``host_us`` : host-side dispatch cost (no sync; the host runs ahead).

Effective algorithmic bandwidth uses the standard formula
``bytes_moved_per_device = full_bytes * (N-1)/N``.

Usage (one process per mesh size, fabric config is process global):
  TT_MESH_GRAPH_DESC_PATH=.../bh_galaxy_8_1_ring_ring.textproto \
  python ccl_microbench.py --mesh 8 --model tinyllama --out results/ccl_8.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import ttnn
import ttml

# [1,1,O,I] weight shapes per model. TP shard for the 8B TP=4 case is applied below.
MODEL_SHAPES = {
    "tinyllama": {  # E=2048, I=5632, heads=32, groups=4 -> kv rows = 2*4*64
        "q_linear": (1, 1, 2048, 2048),
        "kv_linear": (1, 1, 512, 2048),
        "out_linear": (1, 1, 2048, 2048),
        "w1": (1, 1, 5632, 2048),
        "w2": (1, 1, 2048, 5632),
        "fc": (1, 1, 32000, 2048),
    },
    "llama8b": {  # E=4096, I=14336, heads=32, groups=8
        "q_linear": (1, 1, 4096, 4096),
        "kv_linear": (1, 1, 2048, 4096),
        "out_linear": (1, 1, 4096, 4096),
        "w1": (1, 1, 14336, 4096),
        "w2": (1, 1, 4096, 14336),
        "fc": (1, 1, 128256, 4096),
    },
    "llama8b_tp4": {  # per-TP-rank shapes; column-parallel shards dim2, row-parallel dim3
        "q_linear": (1, 1, 1024, 4096),
        "kv_linear": (1, 1, 512, 4096),
        "out_linear": (1, 1, 4096, 1024),
        "w1": (1, 1, 3584, 4096),
        "w2": (1, 1, 4096, 3584),
        "fc": (1, 1, 32064, 4096),
    },
}


def sync(device):
    try:
        ttnn.synchronize_device(device)
    except TypeError:
        ttnn.synchronize_device(device, None)


def on_queue(fn, cq_id):
    """Issue ``fn`` on hardware queue ``cq_id`` (the collectives go to the CCL queue in sub-device mode)."""
    if cq_id is None:
        return fn

    def wrapped():
        with ttnn.command_queue(ttnn.QueueId(cq_id)):
            return fn()

    return wrapped


def timed(fn, device, iters, warmup=2):
    """Return (dev_us_per_op, lat_us, host_us_per_op)."""
    outs = []
    for _ in range(warmup):
        outs.append(fn())
    sync(device)
    for o in outs:
        ttnn.deallocate(o)
    outs.clear()

    # single-op latency (sync before and after)
    sync(device)
    t0 = time.perf_counter()
    o = fn()
    sync(device)
    lat_us = (time.perf_counter() - t0) * 1e6
    ttnn.deallocate(o)

    # steady-state throughput: back-to-back launches then one sync
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        outs.append(fn())
    host_us = (time.perf_counter() - t0) * 1e6 / iters
    sync(device)
    dev_us = (time.perf_counter() - t0) * 1e6 / iters
    for o in outs:
        ttnn.deallocate(o)
    return dev_us, lat_us, host_us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, required=True, help="FSDP axis size (line mesh [N,1])")
    ap.add_argument("--model", default="tinyllama", choices=sorted(MODEL_SHAPES))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dims", default="2,3", help="shard dims to test (comma separated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument(
        "--ccl-subdevice",
        default=None,
        help="run the collectives on a CCL sub-device from the second command queue: rows=N or columns=N",
    )
    args = ap.parse_args()

    n = args.mesh
    mesh = ttml.Mesh((n, 1), ("fsdp", "_1"))
    ttml.open_device_mesh(mesh, num_command_queues=2 if args.ccl_subdevice else 1)
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    if args.ccl_subdevice:
        kind, count = args.ccl_subdevice.split("=")
        ctx.enable_ccl_sub_device(int(count) if kind == "columns" else 0, int(count) if kind == "rows" else 0)
        grid = device.compute_with_storage_grid_size()
        print(f"CCL sub-device on: compute grid {grid.x}x{grid.y}", flush=True)
    axis = mesh.axis_index("fsdp")
    dtype = ttnn.DataType.BFLOAT16 if args.dtype == "bf16" else ttnn.DataType.FLOAT32
    elt = 2 if args.dtype == "bf16" else 4

    fabric = str(ttnn.get_fabric_config()) if hasattr(ttnn, "get_fabric_config") else "?"
    rows = []
    shapes = MODEL_SHAPES[args.model]
    dims = [int(d) for d in args.dims.split(",")]
    print(f"mesh={mesh.shape} axis={axis} fabric={fabric} mgd={os.environ.get('TT_MESH_GRAPH_DESC_PATH')}", flush=True)

    for name, shape in shapes.items():
        for dim in dims:
            if shape[dim] % n != 0:  # same rule as fsdp.py (no tile-alignment check)
                print(f"skip {name} {shape} dim={dim}: not divisible by {n}")
                continue
            aligned = (shape[dim] // n) % 32 == 0
            full_bytes = int(np.prod(shape)) * elt
            shard_shape = list(shape)
            shard_shape[dim] //= n

            # Full replicated tensor -> shard per device via mapper (as materialize does under FSDP).
            full_np = np.random.uniform(-1, 1, size=shape).astype(np.float32)
            shard_at = ttml.autograd.Tensor.from_numpy(full_np, ttnn.Layout.TILE, dtype, mesh.axis_mapper("fsdp", dim))
            shard = shard_at.get_value()
            full_at = ttml.autograd.Tensor.from_numpy(full_np, ttnn.Layout.TILE, dtype, None)
            full = full_at.get_value()

            row = {
                "model": args.model,
                "param": name,
                "shape": list(shape),
                "shard_shape": shard_shape,
                "dim": dim,
                "N": n,
                "shard_tile_aligned": aligned,
                "full_MB": full_bytes / 2**20,
                "shard_MB": full_bytes / n / 2**20,
            }
            moved = full_bytes * (n - 1) / n

            cq = 1 if args.ccl_subdevice else None
            dev, lat, host = timed(
                on_queue(lambda: ttml.core.distributed.all_gather(shard, dim, axis), cq), device, args.iters
            )
            row.update(ag_dev_us=dev, ag_lat_us=lat, ag_host_us=host, ag_GBps=moved / dev / 1e3)

            dev, lat, host = timed(
                on_queue(lambda: ttml.core.distributed.reduce_scatter(full, dim, axis), cq), device, args.iters
            )
            row.update(rs_dev_us=dev, rs_lat_us=lat, rs_host_us=host, rs_GBps=moved / dev / 1e3)

            dev, lat, host = timed(lambda: ttnn.multiply(shard, 1.0 / n), device, args.iters)
            row.update(mul_dev_us=dev, mul_lat_us=lat, mul_host_us=host)

            print(
                f"{name:10s} {str(shape):22s} dim={dim} N={n:2d} {'aligned  ' if aligned else 'MISALIGNED'} full={row['full_MB']:8.1f}MB | "
                f"AG dev={row['ag_dev_us']:9.0f}us lat={row['ag_lat_us']:9.0f}us host={row['ag_host_us']:7.0f}us "
                f"{row['ag_GBps']:6.1f}GB/s | RS dev={row['rs_dev_us']:9.0f}us lat={row['rs_lat_us']:9.0f}us "
                f"host={row['rs_host_us']:7.0f}us {row['rs_GBps']:6.1f}GB/s | mul dev={row['mul_dev_us']:7.0f}us",
                flush=True,
            )
            rows.append(row)
            ttnn.deallocate(shard)
            ttnn.deallocate(full)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"mesh": list(mesh.shape), "fabric": fabric, "rows": rows}, f, indent=1)
    print(f"wrote {args.out}")
    ttml.close_device_mesh()


if __name__ == "__main__":
    main()

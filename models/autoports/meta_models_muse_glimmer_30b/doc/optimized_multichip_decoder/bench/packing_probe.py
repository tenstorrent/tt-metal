# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed vs separate same-input decode projections, at the *per-device* shapes.

Both packings were measured and rejected on one chip -- the doubled output width
capped ``in0_block_w`` at 2 and the packed matmul lost before the split was even
counted (``doc/optimized_decoder/logs/decode_matmul_geometry_packed.log``).  That
rejection turns on a number tensor parallelism changes by 4x: the per-device
output width.  So both families are re-measured here at the real TP=4 shapes,
with the full cost of getting the halves apart again included, not just the
matmul row ($optimize OPT-001, OPT-010).

Two groups:

* ``attn_in``  -- ``wqkv`` (6656 x 1280) and the attention gate (6656 x 1024),
  which both consume the post-norm activation.  The split path's ``wqkv`` output
  goes to ``sharded_to_interleaved`` regardless (the head-creation op needs L1
  interleaved), so the packed path's slice can ride on that same conversion; the
  gate half then needs an ``interleaved_to_sharded`` back onto the boundary grid.
* ``gate_up``  -- the MLP gate and up projections (6656 x 5120 each), whose
  outputs are consumed by one ``ttnn.mul``.  The packed path must split a
  10240-wide sharded output before that multiply.

Everything is traced and replayed, and each arm is measured at 1, 2, 4 and 8
copies per capture so the per-call cost is the *slope* and the replay floor is
the intercept -- the same floor-calibrated protocol
``bench/fractured_decode_probe.py`` established for the multichip stage.

    python .../bench/packing_probe.py
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_L1_SMALL_SIZE,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    decode_matmul_program_config,
    dram_sharded_weight_memcfg,
    width_sharded_l1,
)

TILE = 32
ROWS = 32
HIDDEN = 6656
BOUNDARY_CORES = 16
BFP8, BFP4, BF16 = ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat16
COPIES = (1, 2, 4, 8)


def weight(mesh, k, n, dtype):
    return ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=dram_sharded_weight_memcfg(k, n, mesh),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def activation(mesh, width, cores, grid):
    return ttnn.from_torch(
        torch.randn(1, 1, ROWS, width),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=width_sharded_l1(ROWS, width, cores, grid),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def slope_us(mesh, body, rounds=3):
    """Per-call microseconds with the trace-replay floor removed.

    ``body(n)`` runs the arm ``n`` times inside one capture.  Timing 1/2/4/8
    copies and taking the least-squares slope removes the fixed replay cost,
    which is several microseconds here -- larger than some of the effects.
    """
    times = []
    for n in COPIES:
        outs = body(n)
        ttnn.synchronize_device(mesh)
        for t in outs:
            ttnn.deallocate(t)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = body(n)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(rounds):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - t0) / 32 * 1e6)
        times.append(best)
        ttnn.release_trace(mesh, trace_id)
        del traced
    xs = torch.tensor(COPIES, dtype=torch.float64)
    ys = torch.tensor(times, dtype=torch.float64)
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    floor = float(ys.mean() - slope * xs.mean())
    return slope, floor, times


def report(name, slope, floor, times):
    print(
        f"PACK {name:34s} per_call={slope:8.2f} us  floor={floor:7.2f} us  "
        f"raw={'/'.join(f'{t:.1f}' for t in times)}",
        flush=True,
    )
    return slope


def attn_in_group(mesh, grid):
    """``wqkv`` + attention gate: two 6656-K projections of the same activation."""
    x = activation(mesh, HIDDEN, BOUNDARY_CORES, grid)
    w_qkv = weight(mesh, HIDDEN, 1280, BFP8)
    w_gate = weight(mesh, HIDDEN, 1024, BFP8)
    w_pack = weight(mesh, HIDDEN, 2304, BFP8)
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, packer_l1_acc=True
    )
    results = {}

    def split(n):
        out = []
        for _ in range(n):
            qkv = ttnn.linear(
                x,
                w_qkv,
                dtype=BF16,
                memory_config=width_sharded_l1(ROWS, 1280, BOUNDARY_CORES, grid),
                program_config=decode_matmul_program_config(ROWS, 1280, BOUNDARY_CORES, 13),
                compute_kernel_config=ck,
            )
            qkv_l1 = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(qkv)
            gate = ttnn.linear(
                x,
                w_gate,
                dtype=BF16,
                memory_config=width_sharded_l1(ROWS, 1024, BOUNDARY_CORES, grid),
                program_config=decode_matmul_program_config(ROWS, 1024, BOUNDARY_CORES, 13),
                compute_kernel_config=ck,
            )
            out += [qkv_l1, gate]
        return out

    def packed(n):
        out = []
        for _ in range(n):
            both = ttnn.linear(
                x,
                w_pack,
                dtype=BF16,
                memory_config=width_sharded_l1(ROWS, 2304, BOUNDARY_CORES, grid),
                program_config=decode_matmul_program_config(ROWS, 2304, BOUNDARY_CORES, 13),
                compute_kernel_config=ck,
            )
            both_l1 = ttnn.sharded_to_interleaved(both, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(both)
            qkv_l1 = ttnn.slice(both_l1, [0, 0, 0, 0], [1, 1, ROWS, 1280])
            gate_l1 = ttnn.slice(both_l1, [0, 0, 0, 1280], [1, 1, ROWS, 2304])
            ttnn.deallocate(both_l1)
            gate = ttnn.interleaved_to_sharded(gate_l1, width_sharded_l1(ROWS, 1024, BOUNDARY_CORES, grid))
            ttnn.deallocate(gate_l1)
            out += [qkv_l1, gate]
        return out

    for name, fn in (("attn_in split (shipped)", split), ("attn_in packed", packed)):
        try:
            results[name] = report(name, *slope_us(mesh, fn))
        except Exception as exc:  # noqa: BLE001
            print(f"PACK-FAILED {name}: {str(exc).splitlines()[0][:220]}", flush=True)
    for t in (x, w_qkv, w_gate, w_pack):
        ttnn.deallocate(t)
    return results


def gate_up_group(mesh, grid):
    """MLP gate + up: two 5120-wide BFP4 projections feeding one ``ttnn.mul``."""
    x = activation(mesh, HIDDEN, BOUNDARY_CORES, grid)
    w_gate = weight(mesh, HIDDEN, 5120, BFP4)
    w_up = weight(mesh, HIDDEN, 5120, BFP4)
    w_pack = weight(mesh, HIDDEN, 10240, BFP4)
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, packer_l1_acc=True
    )
    mem5120 = width_sharded_l1(ROWS, 5120, BOUNDARY_CORES, grid)
    results = {}

    def split(n):
        out = []
        for _ in range(n):
            g = ttnn.linear(
                x,
                w_gate,
                dtype=BF16,
                memory_config=mem5120,
                program_config=decode_matmul_program_config(ROWS, 5120, BOUNDARY_CORES, 13),
                compute_kernel_config=ck,
            )
            u = ttnn.linear(
                x,
                w_up,
                dtype=BF16,
                memory_config=mem5120,
                program_config=decode_matmul_program_config(ROWS, 5120, BOUNDARY_CORES, 13),
                compute_kernel_config=ck,
            )
            h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=BF16, memory_config=mem5120)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            out.append(h)
        return out

    def packed(n, block_w):
        out = []
        for _ in range(n):
            both = ttnn.linear(
                x,
                w_pack,
                dtype=BF16,
                memory_config=width_sharded_l1(ROWS, 10240, BOUNDARY_CORES, grid),
                program_config=decode_matmul_program_config(ROWS, 10240, BOUNDARY_CORES, block_w),
                compute_kernel_config=ck,
            )
            both_i = ttnn.sharded_to_interleaved(both, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(both)
            g = ttnn.interleaved_to_sharded(ttnn.slice(both_i, [0, 0, 0, 0], [1, 1, ROWS, 5120]), mem5120)
            u = ttnn.interleaved_to_sharded(ttnn.slice(both_i, [0, 0, 0, 5120], [1, 1, ROWS, 10240]), mem5120)
            ttnn.deallocate(both_i)
            h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=BF16, memory_config=mem5120)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            out.append(h)
        return out

    try:
        results["gate_up split (shipped)"] = report("gate_up split (shipped)", *slope_us(mesh, split))
    except Exception as exc:  # noqa: BLE001
        print(f"PACK-FAILED gate_up split: {str(exc).splitlines()[0][:220]}", flush=True)
    for block_w in (13, 4, 2, 1):
        name = f"gate_up packed in0_block_w={block_w}"
        try:
            results[name] = report(name, *slope_us(mesh, lambda n, b=block_w: packed(n, b)))
        except Exception as exc:  # noqa: BLE001
            print(f"PACK-FAILED {name}: {str(exc).splitlines()[0][:220]}", flush=True)
    for t in (x, w_gate, w_up, w_pack):
        ttnn.deallocate(t)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="attn_in,gate_up")
    args = ap.parse_args()
    mesh = open_multichip_mesh((1, 4), trace_region_size=90112 * 12, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        groups = args.groups.split(",")
        if "attn_in" in groups:
            attn_in_group(mesh, grid)
        if "gate_up" in groups:
            gate_up_group(mesh, grid)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()

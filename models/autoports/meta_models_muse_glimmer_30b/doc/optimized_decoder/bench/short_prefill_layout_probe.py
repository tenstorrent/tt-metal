# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The two open items from the 128-row prefill ``tt-perf-report``.

The short-prefill window
(``tracy/*/prefill_128_perf_report.txt``) leaves exactly two things unsettled,
and this probe measures both rather than arguing about them.

**1. "If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)".**
`tt-perf-report` repeats this for all six 2D-multicast rows.  It is legal here:
``validate_matmul_mcast2d_config`` accepts a ``BLOCK_SHARDED`` in0 with
``ROW_MAJOR`` orientation as long as ``per_core_M`` matches the shard height and
``in0_block_w`` divides the shard's K-tile width, and at 128 rows the activation
is only 1.7 MB.  So the question is whether it *helps*, and the a-priori answer is
"barely", because the activation is ~2 % of the bytes this matmul moves (1.7 MB of
activation against 74.8 MB of BFP4 gate/up weight).  Measured either way.

**2. The core-starved prefill RMSNorm.**  In the 128-row window each of the four
hidden-size ``LayerNormDeviceOperation`` rows costs ~134 us on **4 cores** -- 21 %
of the whole window -- because ``ttnn.rms_norm`` on a DRAM-interleaved input
parallelises over tile *rows*, and 128 rows is 4 tile rows.  The decode path
already uses a width-sharded L1 norm on 16-22 cores at 8-9 us, so the question is
what the same treatment is worth in short prefill, counting the
``interleaved_to_sharded``/``sharded_to_interleaved`` pair it costs.

    python .../bench/short_prefill_layout_probe.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import _norm_subblock_w, norm_compute_kernel_config
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    DEFAULT_PRECISION,
    dram_sharded_weight_memcfg,
    prefill_mcast2d_program_config,
    prefill_mcast2d_spec,
    width_sharded_l1,
)

TILE = 32
HIDDEN = 6656
SHAPES = {"wqkv": (6656, 4608), "mlp_gate": (6656, 19968), "mlp_down": (19968, 6656)}


def timed(fn, mesh, reps: int, rounds: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / reps * 1e3)
    return best


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    if not bool(torch.isfinite(a).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def norm_probe(mesh, rows_list, reps, rounds) -> None:
    """Interleaved vs width-sharded L1 RMSNorm at short prefill row counts."""
    grid = mesh.compute_with_storage_grid_size()
    ck = norm_compute_kernel_config(mesh.arch())
    torch.manual_seed(7)
    weight_t = (1.0 + torch.randn(HIDDEN) * 0.02).to(torch.bfloat16)
    weight = ttnn.from_torch(
        weight_t.reshape(1, 1, 1, HIDDEN), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    for rows in rows_list:
        x_t = torch.randn(1, 1, rows, HIDDEN)
        ref = (
            x_t.to(torch.float64)
            / torch.sqrt((x_t.to(torch.float64) ** 2).mean(-1, keepdim=True) + 1e-5)
            * weight_t.to(torch.float64)
        ).float()
        x = ttnn.from_torch(
            x_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def interleaved():
            return ttnn.rms_norm(
                x, weight=weight, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck
            )

        ms = timed(interleaved, mesh, reps, rounds)
        out = interleaved()
        print(
            f"NORM rows={rows:5d} interleaved            {ms * 1e3:9.1f} us  pcc={pcc(ttnn.to_torch(out), ref):.6f}",
            flush=True,
        )
        ttnn.deallocate(out)

        for cores in (8, 16, 26, 52):
            if (HIDDEN // TILE) % cores:
                continue
            memcfg = width_sharded_l1(rows, HIDDEN, cores, grid)
            block_w = HIDDEN // cores // TILE
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[min(cores, grid.x), math.ceil(cores / grid.x)],
                subblock_w=_norm_subblock_w(block_w),
                block_h=rows // TILE,
                block_w=block_w,
                inplace=False,
            )

            def sharded():
                # The whole cost, not just the norm: prefill activations are DRAM
                # interleaved on both sides of this, so the conversions are part of
                # the candidate.
                xs = ttnn.interleaved_to_sharded(x, memcfg)
                ys = ttnn.rms_norm(
                    xs,
                    weight=weight,
                    epsilon=1e-5,
                    memory_config=memcfg,
                    program_config=program_config,
                    compute_kernel_config=ck,
                )
                ttnn.deallocate(xs)
                y = ttnn.sharded_to_interleaved(ys, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(ys)
                return y

            try:
                ms_s = timed(sharded, mesh, reps, rounds)
                out = sharded()
                print(
                    f"NORM rows={rows:5d} sharded_l1_{cores:2d}cores+i2s+s2i {ms_s * 1e3:9.1f} us  "
                    f"pcc={pcc(ttnn.to_torch(out), ref):.6f}  speedup={ms / ms_s:5.2f}x",
                    flush=True,
                )
                ttnn.deallocate(out)
            except Exception as exc:  # noqa: BLE001
                msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                print(f"NORM rows={rows} sharded_l1_{cores}cores BLOCKED: {msg[:220]}", flush=True)
        ttnn.deallocate(x)
    ttnn.deallocate(weight)


def in0_l1_probe(mesh, rows, reps, rounds) -> None:
    """2D-multicast matmul with in0 DRAM-interleaved vs BLOCK_SHARDED in L1."""
    banks = mesh.dram_grid_size().x
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=DEFAULT_PRECISION.prefill_math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    for name, (k, n) in SHAPES.items():
        dtype = DEFAULT_PRECISION.weight_dtype(name)
        spec = prefill_mcast2d_spec(name, rows, dtype)
        if spec is None:
            print(f"IN0L1 {name} rows={rows}: no 2D-multicast band, skipped", flush=True)
            continue
        grid_y, in0_block_w = spec
        torch.manual_seed(abs(hash(name)) % 9973)
        w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
        w = ttnn.from_torch(
            w_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=dram_sharded_weight_memcfg(k, n, mesh)
        )
        x_t = torch.randn(1, 1, rows, k)
        ref = (x_t.to(torch.float64) @ w_t.to(torch.float64)).float()
        program_config = prefill_mcast2d_program_config(rows, n, grid_y, in0_block_w, banks)
        x = ttnn.from_torch(
            x_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def dram_in0():
            return ttnn.linear(
                x,
                w,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ck,
                program_config=program_config,
            )

        ms = timed(dram_in0, mesh, reps, rounds)
        out = dram_in0()
        print(
            f"IN0L1 {name:10s} rows={rows:5d} in0=DRAM_interleaved      {ms * 1e3:9.1f} us  "
            f"pcc={pcc(ttnn.to_torch(out), ref):.6f}",
            flush=True,
        )
        ttnn.deallocate(out)

        # BLOCK_SHARDED in0 over the same (banks, grid_y) program grid: the
        # validator requires per_core_M == shard_height / 32 and in0_block_w to
        # divide the shard's K-tile width, which the program grid already gives.
        shard = (rows // grid_y, k // banks)
        memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, grid_y - 1))]),
                shard,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        def l1_in0():
            xs = ttnn.interleaved_to_sharded(x, memcfg)
            y = ttnn.linear(
                xs,
                w,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ck,
                program_config=program_config,
            )
            ttnn.deallocate(xs)
            return y

        try:
            ms_l1 = timed(l1_in0, mesh, reps, rounds)
            out = l1_in0()
            print(
                f"IN0L1 {name:10s} rows={rows:5d} in0=L1_block_sharded+i2s  {ms_l1 * 1e3:9.1f} us  "
                f"pcc={pcc(ttnn.to_torch(out), ref):.6f}  speedup={ms / ms_l1:5.2f}x  shard={shard}",
                flush=True,
            )
            ttnn.deallocate(out)
        except Exception as exc:  # noqa: BLE001
            msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
            print(f"IN0L1 {name} rows={rows} in0=L1_block_sharded BLOCKED: {msg[:260]}", flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,512,1024,2048")
    ap.add_argument("--in0-rows", type=int, default=128)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        norm_probe(mesh, [int(r) for r in args.rows.split(",")], args.reps, args.rounds)
        in0_l1_probe(mesh, args.in0_rows, args.reps, args.rounds)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

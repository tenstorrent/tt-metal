# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill dense-projection sweep for the optimized weight layout.

``weight_layout_probe.py`` established that one DRAM **width-sharded** weight
serves both the decode ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``
matmul and prefill's ``ttnn.experimental.minimal_matmul``, while ``ttnn.linear``
refuses a width-sharded ``input_tensor_b`` outright
(``matmul_device_operation.cpp:1233``).  The fused decoder's ``_dense`` dispatched
to ``ttnn.linear`` below 3072 rows, so that rule cannot survive unchanged: this
sweep measures, per projection and per row count,

* ``minimal_matmul`` with the width-sharded BFP8 weight,
* ``minimal_matmul`` with the width-sharded BFP4 weight,
* the fused decoder's shipped baseline -- ``ttnn.linear`` with a DRAM-interleaved
  BF16 weight -- so the "never slower than the previous stage at any row count"
  property can be re-checked rather than assumed,
* ``minimal_matmul`` with a DRAM-interleaved BF16 weight, i.e. the fused stage's
  own >=3072-row branch, as the dtype-only comparison.

    python .../bench/prefill_matmul_sweep.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32

SHAPES = {
    "wqkv": (6656, 4608),
    "attn_gate": (6656, 4096),
    "o_proj": (4096, 6656),
    "mlp_gate_up": (6656, 19968),
    "mlp_down": (19968, 6656),
}

ROWS = (32, 128, 512, 1024, 2048, 3072, 4096, 6144, 8192)


def dram_width_sharded(k: int, n: int, mesh) -> ttnn.MemoryConfig:
    cores = mesh.dram_grid_size().x
    padded = math.ceil(n / (TILE * cores)) * (TILE * cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def timed(fn, mesh, reps: int, rounds: int) -> tuple[float, list[float]]:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    per_round = []
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        per_round.append((time.perf_counter() - t0) / reps * 1e3)
    return min(per_round), per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--fidelity", default="lofi", choices=("lofi", "hifi2", "hifi4"))
    args = ap.parse_args()

    names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    fidelity = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[
        args.fidelity
    ]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # The fused stage's shipped policy for the BF16 baseline column.
        ck_hifi2 = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for name in names:
            k, n = SHAPES[name]
            torch.manual_seed(abs(hash(name)) % 10_000)
            w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
            weights = {
                "mm_bfp8_shard": (
                    ttnn.from_torch(
                        w_t,
                        device=mesh,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat8_b,
                        memory_config=dram_width_sharded(k, n, mesh),
                    ),
                    "minimal",
                    ck,
                ),
                "mm_bfp4_shard": (
                    ttnn.from_torch(
                        w_t,
                        device=mesh,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat4_b,
                        memory_config=dram_width_sharded(k, n, mesh),
                    ),
                    "minimal",
                    ck,
                ),
                "mm_bf16_inter": (
                    ttnn.from_torch(
                        w_t,
                        device=mesh,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    "minimal",
                    ck_hifi2,
                ),
            }
            weights["linear_bf16_inter"] = (weights["mm_bf16_inter"][0], "linear", ck_hifi2)
            for rows in [int(r) for r in args.rows.split(",")]:
                x_t = torch.randn(1, 1, rows, k)
                ref = (x_t.to(torch.float64) @ w_t.to(torch.float64)).float()
                x = ttnn.from_torch(
                    x_t,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for label, (w, kind, cfg) in weights.items():

                    def run(w=w, kind=kind, cfg=cfg):
                        op = ttnn.experimental.minimal_matmul if kind == "minimal" else ttnn.linear
                        return op(
                            x,
                            w,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=cfg,
                        )

                    print(f"GROUP {args.reps} {name} prefill_{label}_r{rows}", flush=True)
                    try:
                        ms, rounds = timed(run, mesh, args.reps, args.rounds)
                        out = run()
                        p = pcc(ttnn.to_torch(out), ref)
                        ttnn.deallocate(out)
                        print(
                            f"RESULT {name:12s} rows={rows:5d} {label:18s} {ms:9.4f} ms  pcc={p:.6f}  "
                            f"({'/'.join(f'{r:.4f}' for r in rounds)})",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                        print(f"BLOCKED {name} prefill_{label}_r{rows}: {msg[:400]}", flush=True)
                ttnn.deallocate(x)
            for w, _, _ in list(weights.values())[:3]:
                ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

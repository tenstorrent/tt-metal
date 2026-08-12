# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode projection matmul sweep: dtype x fidelity x program-config geometry.

93 % of the fused decoder's decode step is six matmuls streaming 968 MB of BF16
weights (``doc/fused_decoder/README.md`` limitation 1), so this probe is the
whole optimized-decoder search in isolation: for each of the five projection
shapes the layer runs at 32 rows, measure

* the shipped form -- ``ttnn.linear`` with DRAM-interleaved weights and a
  DRAM-interleaved activation, the op picking its own program config;
* the same with BFP8 / BFP4 weights;
* an explicit DRAM-sharded matmul
  (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``) over a sweep of
  logical core counts and ``in0_block_w`` values, with the weight
  width-sharded in DRAM and activation/output width-sharded in L1;

each at LoFi and HiFi2 math fidelity, with PCC against a float32 torch
reference on the *same* random weights so an op-contract failure cannot be
mistaken for a precision result.

Wall-clock is measured over ``--reps`` back-to-back dispatches with one
synchronize at the end, ``--rounds`` times, reported as the minimum.  At these
shapes one dispatch is 0.15-1.1 ms, so a 200-rep window is 30-200 ms and the
wall-clock resolves the 2x dtype effects this probe is looking for; the
1-3 % geometry decisions are re-confirmed on device time by running this same
script under Tracy (each candidate emits a ``GROUP`` line for
``bench/summarize_device_probe.py``).

    python .../bench/decode_matmul_sweep.py --shapes all --stage dtype
    python .../bench/decode_matmul_sweep.py --shapes wqkv --stage geometry
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32

#: ``(label, K, N)`` for every dense projection a decode step runs, in graph
#: order.  ``mlp_gate`` and ``mlp_up`` are the same shape, dispatched twice.
SHAPES = {
    "wqkv": (6656, 4608),
    "attn_gate": (6656, 4096),
    "o_proj": (4096, 6656),
    "mlp_gate_up": (6656, 19968),
    "mlp_down": (19968, 6656),
    # Shared-LHS packing candidates (OPT-001, OPT-010).  ``wqkv`` and the
    # attention output gate both consume the input_layernorm output; ``gate`` and
    # ``up`` both consume the pre_feedforward_layernorm output.  Packed here as
    # one matmul of the summed output width, so the packed *matmul* row can be
    # compared against the sum of the split rows before the on-device slice cost
    # is added back in.
    "qkv_gate_packed": (6656, 8704),
    "gate_up_packed": (6656, 39936),
}

DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}

FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def compute_kernel(arch, fidelity: str):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=FIDELITIES[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def core_range_set(num_cores: int, grid: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    """Row-major prefix of ``grid`` holding ``num_cores`` cores."""
    ranges = []
    full_rows = num_cores // grid.x
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    rest = num_cores % grid.x
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rest - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def dram_weight_memcfg(k: int, n: int, mesh) -> ttnn.MemoryConfig:
    dram_grid = mesh.dram_grid_size()
    dram_cores = dram_grid.x
    assert dram_grid.y == 1, f"unexpected dram grid {dram_grid}"
    padded = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def l1_width_sharded(rows: int, width: int, num_cores: int, grid: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    per_core = math.ceil(width / (TILE * num_cores)) * TILE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set(num_cores, grid), (rows, per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )


def measure(fn, reps: int, rounds: int, mesh) -> tuple[float, list[float]]:
    """Min-of-``rounds`` wall clock per dispatch over ``reps`` back-to-back calls.

    Each output is deallocated as soon as it is issued.  Holding ``reps``
    L1-sharded outputs alive instead makes the op fail with *"Statically
    allocated circular buffers ... clash with L1 buffers"* (``program.cpp:1779``)
    at every wide shape -- a harness artefact, not an op contract.  Dispatch
    stays in order on one command queue, so freeing the address immediately is
    safe and is the same pattern ``bench/ab_latency.py`` uses.
    """
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


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--stage", default="dtype", choices=("dtype", "geometry", "both"))
    ap.add_argument("--dtypes", default="bf16,bfp8,bfp4")
    ap.add_argument("--fidelities", default="lofi,hifi2")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--geometry-dtype", default="bfp8")
    ap.add_argument("--geometry-fidelity", default="lofi")
    ap.add_argument("--cores", default="", help="comma-separated logical core counts; default = all legal")
    args = ap.parse_args()

    names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    grid = mesh.compute_with_storage_grid_size()
    print(f"grid={grid} dram_grid={mesh.dram_grid_size()}", flush=True)
    try:
        for name in names:
            k, n = SHAPES[name]
            torch.manual_seed(hash(name) % 10_000)
            w_t = torch.randn(1, 1, k, n, dtype=torch.float32) * (1.0 / math.sqrt(k))
            x_t = torch.randn(1, 1, args.rows, k, dtype=torch.float32)
            ref = (x_t.to(torch.float64) @ w_t.to(torch.float64)).to(torch.float32)
            x_dram = ttnn.from_torch(
                x_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            if args.stage in ("dtype", "both"):
                for dt in args.dtypes.split(","):
                    w = ttnn.from_torch(
                        w_t,
                        device=mesh,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=DTYPES[dt],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for fid in args.fidelities.split(","):
                        ck = compute_kernel(mesh.arch(), fid)

                        def run():
                            return ttnn.linear(
                                x_dram,
                                w,
                                dtype=ttnn.bfloat16,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                compute_kernel_config=ck,
                            )

                        label = f"interleaved_{name}_{dt}_{fid}"
                        print(f"GROUP {args.reps} {name} {label}", flush=True)
                        ms, rounds = measure(run, args.reps, args.rounds, mesh)
                        out = run()
                        p = pcc(ttnn.to_torch(out), ref)
                        ttnn.deallocate(out)
                        print(
                            f"RESULT interleaved {name:12s} K={k:5d} N={n:5d} dtype={dt:5s} fid={fid:5s} "
                            f"cores=auto in0bw=auto  {ms:8.4f} ms  pcc={p:.6f}  "
                            f"({'/'.join(f'{r:.4f}' for r in rounds)})",
                            flush=True,
                        )
                    ttnn.deallocate(w)

            if args.stage in ("geometry", "both"):
                dt = args.geometry_dtype
                fid = args.geometry_fidelity
                ck = compute_kernel(mesh.arch(), fid)
                w_sharded = ttnn.from_torch(
                    w_t,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=DTYPES[dt],
                    memory_config=dram_weight_memcfg(k, n, mesh),
                )
                k_tiles = k // TILE
                if args.cores:
                    core_list = [int(c) for c in args.cores.split(",")]
                else:
                    core_list = [c for c in divisors(k_tiles) if c <= grid.x * grid.y]
                for cores in core_list:
                    if k_tiles % cores:
                        print(f"SKIP {name} cores={cores}: {k_tiles} tiles not divisible", flush=True)
                        continue
                    in_memcfg = l1_width_sharded(args.rows, k, cores, grid)
                    out_memcfg = l1_width_sharded(args.rows, n, cores, grid)
                    per_core_n = math.ceil(n / (TILE * cores))
                    for in0_bw in divisors(k_tiles // cores):
                        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=in0_bw,
                            per_core_M=math.ceil(args.rows / TILE),
                            per_core_N=per_core_n,
                            fused_activation=None,
                        )
                        label = f"dramshard_{name}_{dt}_{fid}_c{cores}_bw{in0_bw}"
                        print(f"GROUP {args.reps} {name} {label}", flush=True)
                        try:
                            x_sharded = ttnn.to_memory_config(x_dram, in_memcfg)

                            def run():
                                return ttnn.linear(
                                    x_sharded,
                                    w_sharded,
                                    dtype=ttnn.bfloat16,
                                    memory_config=out_memcfg,
                                    program_config=cfg,
                                    compute_kernel_config=ck,
                                )

                            ms, rounds = measure(run, args.reps, args.rounds, mesh)
                            out = run()
                            got = ttnn.to_torch(out)[..., :n]
                            p = pcc(got, ref)
                            ttnn.deallocate(out)
                            ttnn.deallocate(x_sharded)
                            print(
                                f"RESULT dramshard  {name:12s} K={k:5d} N={n:5d} dtype={dt:5s} fid={fid:5s} "
                                f"cores={cores:4d} in0bw={in0_bw:3d} percoreN={per_core_n:3d}  {ms:8.4f} ms  "
                                f"pcc={p:.6f}  ({'/'.join(f'{r:.4f}' for r in rounds)})",
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - the point is to record the blocker
                            msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                            print(
                                f"BLOCKED {name} {label}: cores={cores} in0bw={in0_bw} {msg[:600]}",
                                flush=True,
                            )
                ttnn.deallocate(w_sharded)
            ttnn.deallocate(x_dram)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

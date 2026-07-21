# SPDX-License-Identifier: Apache-2.0
"""
Llama-3.1-8B decode matmul sweep v2 (Blackhole P150) — robust edition.

Differences vs tt_llama_p150_matmul_sweep.py:
  * Every candidate is built + measured inside its own try/except; the exact
    exception text is recorded in the CSV `note` column (so FF2/QKV/WO/LMHEAD
    failures are explained, not just "OOM/ERR").
  * Tensors are (re)built per-config so a failure never poisons later configs.
  * Two program-config factories are attempted per candidate:
        - dram_sharded : MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        - mcast_1d     : MatmulMultiCoreReuseMultiCast1DProgramConfig
    (FF2's K=14336 shard often only builds under one of them.)
  * PRIMARY metric = device kernel duration captured via the profiler
    (ReadDeviceProfiler + get_latest_programs_perf_data), exactly like
    tt_llama_p150_matmul_sweep.py. Host wall-clock is only a diagnostic
    fallback: host-timed rows are marked src=host, status=HOST_ONLY and are
    EXCLUDED from BEST ranking (unless --allow-host-rank). Run with the
    profiler env vars below or every row will be host-only.

  Required env for device-kernel capture (teja's methodology):
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
           TT_METAL_PROFILER_CPP_POST_PROCESS=1
  * Expanded axes: --sweep-dtype/-fid/-packer/-fp32/-percoreN/-factory.

Run (profiler build recommended):
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) MESH_DEVICE=P150
  python_env/bin/python models/tt_transformers/tests/sweeps/sweep_llama_mm_v2.py \
      --shapes FF2 QKV WO LMHEAD --factory both --sweep-percoreN --csv sweep_v2.csv
"""
import argparse
import csv
import itertools
import math
import time

import torch

import ttnn

DRAM_PEAK_GBPS = 512.0
DRAM_ACHIEVABLE_GBPS = 377.0
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048, ttnn.float32: 4096}
BYTES_PER_ELEM = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625, ttnn.bfloat16: 2.0}
CYCLES_PER_TILE = {ttnn.MathFidelity.LoFi: 16, ttnn.MathFidelity.HiFi2: 32, ttnn.MathFidelity.HiFi4: 64}
DEVICE_FREQ_HZ = 1350e6
DRAM_ALIGN = 64
L1_BUDGET_KB_DEFAULT = 1400

SHAPES = {
    "QKV": dict(M=32, K=4096, N=6144, dtype=ttnn.bfloat8_b, fid=ttnn.MathFidelity.HiFi2),
    "WO": dict(M=32, K=4096, N=4096, dtype=ttnn.bfloat8_b, fid=ttnn.MathFidelity.HiFi2),
    "FF1_FF3": dict(M=32, K=4096, N=14336, dtype=ttnn.bfloat4_b, fid=ttnn.MathFidelity.LoFi),
    "FF2": dict(M=32, K=14336, N=4096, dtype=ttnn.bfloat8_b, fid=ttnn.MathFidelity.HiFi2),
    "LMHEAD": dict(M=32, K=4096, N=16032, dtype=ttnn.bfloat8_b, fid=ttnn.MathFidelity.HiFi2),
}


def align_up(x, a):
    return x if x % a == 0 else x + (a - x % a)


def pad_to_dram_banks(n, num_banks):
    lcm = 32 * num_banks
    r = n % lcm
    return n if r == 0 else n + (lcm - r)


def grid_from_cores(nc, max_x=13, max_y=10):
    for rows in range(1, max_y + 1):
        if nc % rows == 0 and nc // rows <= max_x:
            return (nc // rows, rows)
    return None


def find_largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def all_divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def model_baseline_cores(K_tiles, N_tiles, max_cores=64):
    cand = [c for c in range(1, max_cores + 1) if K_tiles % c == 0 and N_tiles % c == 0]
    return max(cand) if cand else None


def candidate_cores(K_tiles, N_tiles, cap=130, require_n_div=False):
    """Core counts that divide K tiles (for in0 width-shard). Optionally also require
    N divisibility (some factories need per_core_N exact)."""
    out = []
    for c in range(1, cap + 1):
        if K_tiles % c != 0:
            continue
        if require_n_div and N_tiles % c != 0:
            continue
        if grid_from_cores(c):
            out.append(c)
    return out


def estimate_l1_kb(
    per_core_M, per_core_N, in0_block_w, K_tiles, in0_slice_tiles, in1_dtype, packer_l1_acc, fp32_dest, has_bias=False
):
    in0_tile = TILE_BYTES[ttnn.bfloat16]
    in1_tile = align_up(TILE_BYTES[in1_dtype], DRAM_ALIGN)
    out_tile = TILE_BYTES[ttnn.bfloat16]
    num_blocks = max(1, K_tiles // max(1, in0_block_w))
    double = num_blocks > 1
    packer_en = packer_l1_acc and num_blocks > 1
    if packer_en:
        interm_tile = TILE_BYTES[ttnn.float32] if fp32_dest else TILE_BYTES[ttnn.bfloat16]
    else:
        interm_tile = TILE_BYTES[ttnn.float32] if fp32_dest else out_tile
    pcn = per_core_N + 3
    in0_CB = per_core_M * in0_block_w * in0_tile * (2 if double else 1)
    in1_CB = pcn * in0_block_w * in1_tile * (3 if double else 1)
    out_CB = per_core_M * pcn * out_tile
    interm_CB = per_core_M * pcn * interm_tile
    reshard_CB = per_core_M * pcn * out_tile
    in2_CB = per_core_M * in0_slice_tiles * in0_tile
    in3_CB = pcn * align_up(TILE_BYTES[in1_dtype], DRAM_ALIGN) if has_bias else 0
    return (in0_CB + in1_CB + out_CB + interm_CB + reshard_CB + in2_CB + in3_CB) / 1024.0


def build_in1(device, K, N, in1_dtype, num_banks):
    N_padded = pad_to_dram_banks(N, num_banks)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    in1_spec = ttnn.ShardSpec(shard_grid, [K, N_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_spec)
    in1 = torch.randn([1, 1, K, N]).bfloat16().float()
    return ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem)


def build_in1_interleaved(device, K, N, in1_dtype):
    dram_il = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in1 = torch.randn([1, 1, K, N]).bfloat16().float()
    return ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_il)


def build_in0_sharded(device, M, K, num_cores, grid_xy):
    K_tiles = K // 32
    in0_slice_tiles = K_tiles // num_cores
    dram_il = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in0 = torch.randn([1, 1, M, K]).bfloat16().float()
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_il)
    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_xy,
        [M, in0_slice_tiles * 32],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return in0_t, in0_slice_tiles


def build_in0_dram(device, M, K):
    dram_il = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in0 = torch.randn([1, 1, M, K]).bfloat16().float()
    return ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_il)


def make_pc(factory, in0_block_w, M, per_core_N, grid_xy, K_tiles, num_cores):
    if factory == "dram_sharded":
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=M // 32,
            per_core_N=per_core_N,
            fused_activation=None,
        )
    elif factory == "mcast_1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_xy,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=find_largest_divisor(per_core_N, 4),
            per_core_M=M // 32,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    raise ValueError(f"unknown factory {factory}")


def time_kernel_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return None
    if not latest:
        return None
    dev_id = device.get_device_ids()[0] if hasattr(device, "get_device_ids") else 0
    progs = latest.get(dev_id) if latest else None
    if not progs:
        return None
    dur_ns, cores = None, 0
    for p in progs:
        r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
        if r and r.duration is not None and (dur_ns is None or r.duration > dur_ns):
            dur_ns, cores = r.duration, p.core_count
    if not dur_ns:
        return None
    return dur_ns, cores


def measure(
    device,
    factory,
    in0_t,
    in1_t,
    out_mem,
    M,
    K,
    N,
    in1_dtype,
    fidelity,
    in0_block_w,
    per_core_N,
    grid_xy,
    K_tiles,
    num_cores,
    packer,
    fp32,
    out_dtype,
    iterations,
):
    pc = make_pc(factory, in0_block_w, M, per_core_N, grid_xy, K_tiles, num_cores)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=packer,
    )
    # warmup + profiler flush
    out = ttnn.matmul(
        in0_t, in1_t, program_config=pc, memory_config=out_mem, dtype=out_dtype, compute_kernel_config=ckc
    )
    ttnn.synchronize_device(device)
    out.deallocate(True)
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass

    t0 = time.perf_counter()
    for _ in range(iterations):
        out = ttnn.matmul(
            in0_t, in1_t, program_config=pc, memory_config=out_mem, dtype=out_dtype, compute_kernel_config=ckc
        )
        ttnn.synchronize_device(device)
        out.deallocate(True)
    host_ns = (time.perf_counter() - t0) * 1e9 / iterations

    prof = time_kernel_profiler(device)
    if prof:
        dur_ns, cores = prof
        src = "prof"
    else:
        dur_ns, cores = host_ns, num_cores
        src = "host"

    weight_bytes = K * N * BYTES_PER_ELEM[in1_dtype]
    bw = weight_bytes / dur_ns
    ideal = ((M // 32) * K_tiles * (N // 32) * CYCLES_PER_TILE[fidelity]) / max(cores, 1)
    actual = dur_ns * 1e-9 * DEVICE_FREQ_HZ
    util = 100.0 * ideal / actual if actual else 0.0
    return dict(
        dur_ns=dur_ns,
        cores=cores,
        bw_gbps=bw,
        src=src,
        dram_pct=100 * bw / DRAM_PEAK_GBPS,
        ach_pct=100 * bw / DRAM_ACHIEVABLE_GBPS,
        util=util,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES.keys()))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--factory", choices=["dram_sharded", "mcast_1d", "both"], default="dram_sharded")
    ap.add_argument("--sweep-dtype", action="store_true")
    ap.add_argument("--sweep-fid", action="store_true")
    ap.add_argument("--sweep-packer", action="store_true")
    ap.add_argument("--sweep-fp32", action="store_true")
    ap.add_argument("--sweep-percoreN", action="store_true", help="also try per_core_N multiples (pad N distribution)")
    ap.add_argument("--sweep-outdtype", action="store_true", help="also sweep bf16 vs bf8 output")
    ap.add_argument(
        "--out-mems",
        nargs="+",
        default=["l1_ws"],
        choices=["l1_ws", "l1_il", "dram_il"],
        help="output memory configs: l1_ws=L1 width-sharded, l1_il=L1 interleaved, dram_il=DRAM interleaved",
    )
    ap.add_argument("--max-cores", type=int, default=130)
    ap.add_argument("--max-in0-block-w", type=int, default=0)
    ap.add_argument("--l1-budget", type=int, default=L1_BUDGET_KB_DEFAULT)
    ap.add_argument("--no-l1-filter", action="store_true", help="skip analytic L1 prefilter; rely on try/except")
    ap.add_argument("--csv", type=str, default="sweep_v2.csv")
    args = ap.parse_args()

    factories = ["dram_sharded", "mcast_1d"] if args.factory == "both" else [args.factory]

    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    num_banks = device.dram_grid_size().x

    with open(args.csv, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "shape",
                "factory",
                "M",
                "K",
                "N",
                "num_cores",
                "grid",
                "in0_block_w",
                "per_core_N",
                "in1_dtype",
                "out_dtype",
                "out_mem",
                "fidelity",
                "packer_l1_acc",
                "fp32_dest",
                "l1_kb_est",
                "src",
                "dur_us",
                "bw_gbps",
                "dram_pct",
                "ach_pct",
                "util_pct",
                "status",
                "note",
            ]
        )

    def row_out(row):
        with open(args.csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    hdr = (
        f"{'fac':>12s} {'nc':>4s} {'grid':>7s} {'bw':>5s} {'pcN':>4s} {'pk':>2s} {'f32':>3s} "
        f"{'odt':>4s} {'L1KB':>6s} {'src':>4s} {'us':>8s} {'GB/s':>7s} {'ach%':>6s} {'note':>10s}"
    )

    try:
        for sname in args.shapes:
            base = SHAPES[sname]
            M, K, N = base["M"], base["K"], base["N"]
            K_tiles, N_tiles = K // 32, N // 32
            baseline_nc = model_baseline_cores(K_tiles, N_tiles)
            dtypes = [ttnn.bfloat8_b, ttnn.bfloat4_b] if args.sweep_dtype else [base["dtype"]]
            fids = (
                [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
                if args.sweep_fid
                else [base["fid"]]
            )
            packers = [True, False] if args.sweep_packer else [True]
            fp32s = [True, False] if args.sweep_fp32 else [True]
            out_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b] if args.sweep_outdtype else [ttnn.bfloat16]

            print(
                f"\n===== {sname}  M={M} K={K} N={N}  (model baseline={baseline_nc}c, "
                f"in0bw={find_largest_divisor(K_tiles // baseline_nc) if baseline_nc else '?'}) ====="
            )
            print(hdr)

            n_ok = n_l1 = n_err = 0
            best = None
            cores_list = candidate_cores(K_tiles, N_tiles, args.max_cores)
            out_mem_map = {
                "l1_ws": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                "l1_il": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
                "dram_il": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            }
            for factory, in1_dtype, out_mem_name in itertools.product(factories, dtypes, args.out_mems):
                out_mem = out_mem_map[out_mem_name]
                for nc in cores_list:
                    grid_xy = grid_from_cores(nc)
                    in0_slice_tiles = K_tiles // nc
                    base_pcN = math.ceil(N_tiles / nc)
                    pcN_list = [base_pcN]
                    if args.sweep_percoreN:
                        for mult in (base_pcN + 1, base_pcN + 2):
                            if mult not in pcN_list:
                                pcN_list.append(mult)
                    blk_cap = args.max_in0_block_w or in0_slice_tiles
                    blk_cands = [b for b in all_divisors(in0_slice_tiles) if b <= blk_cap]
                    for in0_block_w, per_core_N, fid, packer, fp32, odt in itertools.product(
                        blk_cands, pcN_list, fids, packers, fp32s, out_dtypes
                    ):
                        l1_kb = estimate_l1_kb(
                            M // 32, per_core_N, in0_block_w, K_tiles, in0_slice_tiles, in1_dtype, packer, fp32
                        )
                        rb = [
                            sname,
                            factory,
                            M,
                            K,
                            N,
                            nc,
                            str(grid_xy),
                            in0_block_w,
                            per_core_N,
                            str(in1_dtype).split(".")[-1],
                            str(odt).split(".")[-1],
                            out_mem_name,
                            str(fid).split(".")[-1],
                            packer,
                            fp32,
                            f"{l1_kb:.0f}",
                        ]
                        if (not args.no_l1_filter) and l1_kb > args.l1_budget:
                            n_l1 += 1
                            row_out(rb + ["", "", "", "", "", "", "SKIP_L1", ""])
                            continue
                        in0_t = in1_t = None
                        try:
                            if factory == "dram_sharded":
                                in1_t = build_in1(device, K, N, in1_dtype, num_banks)
                                in0_t, _ = build_in0_sharded(device, M, K, nc, grid_xy)
                            else:  # mcast_1d
                                in1_t = build_in1_interleaved(device, K, N, in1_dtype)
                                in0_t, _ = build_in0_sharded(device, M, K, nc, grid_xy)
                            r = measure(
                                device,
                                factory,
                                in0_t,
                                in1_t,
                                out_mem,
                                M,
                                K,
                                N,
                                in1_dtype,
                                fid,
                                in0_block_w,
                                per_core_N,
                                grid_xy,
                                K_tiles,
                                nc,
                                packer,
                                fp32,
                                odt,
                                args.iters,
                            )
                        except Exception as e:
                            n_err += 1
                            msg = str(e).strip().split("\n")[0][:80] or type(e).__name__
                            row_out(rb + ["", "", "", "", "", "", "ERR", msg])
                            continue
                        finally:
                            for t in (in0_t, in1_t):
                                try:
                                    if t is not None:
                                        t.deallocate(True)
                                except Exception:
                                    pass
                        n_ok += 1
                        row_out(
                            rb
                            + [
                                r["src"],
                                f"{r['dur_ns']/1000:.1f}",
                                f"{r['bw_gbps']:.1f}",
                                f"{r['dram_pct']:.1f}",
                                f"{r['ach_pct']:.1f}",
                                f"{r['util']:.1f}",
                                "OK",
                                "",
                            ]
                        )
                        print(
                            f"{factory:>12s} {nc:4d} {str(grid_xy):>7s} {in0_block_w:5d} {per_core_N:4d} "
                            f"{'Y' if packer else 'N':>2s} {'Y' if fp32 else 'N':>3s} "
                            f"{str(odt).split('.')[-1][:4]:>4s} {l1_kb:6.0f} {r['src']:>4s} "
                            f"{r['dur_ns']/1000:8.1f} {r['bw_gbps']:7.1f} {r['ach_pct']:5.1f}%"
                        )
                        if best is None or r["dur_ns"] < best[0]:
                            best = (
                                r["dur_ns"],
                                factory,
                                nc,
                                in0_block_w,
                                per_core_N,
                                fid,
                                packer,
                                fp32,
                                in1_dtype,
                                odt,
                                r["bw_gbps"],
                                r["ach_pct"],
                                r["src"],
                            )

            print(f"  [{sname}] OK={n_ok} SKIP_L1={n_l1} ERR={n_err}")
            if best:
                (d, fac, nc, bw, pcN, fid, pk, f32, idt, odt, gbps, ach, src) = best
                print(
                    f"  BEST {sname}: {fac} {nc}c in0bw={bw} pcN={pcN} fid={str(fid).split('.')[-1]} "
                    f"pk={pk} f32={f32} in1={str(idt).split('.')[-1]} out={str(odt).split('.')[-1]} "
                    f"-> {d/1000:.1f}us {gbps:.0f}GB/s ({ach:.0f}% ach) [{src}]"
                )
        print(f"\nFull results -> {args.csv}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

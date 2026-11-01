# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What actually bounds the bf2 matmul? DRAM-sharded in1, the config not yet tried.

The grid and block sweeps left bfp2 at 149.8 us against bfp4's 161.4 (1.08x) and
showed the two formats tracking each other within ~10% at every 1D-mcast config.
That is not what half the bytes should look like, so something other than the
weight stream is setting the time. Two candidates:

  the in1 read handshake   every K-block costs a round trip that is independent of
                           how few bytes are fetched, so halving the bytes saves
                           nothing until the block is large enough for the read to
                           dominate. DRAM-sharding in1 across banks is the config
                           that removes that coupling, and it is the config the plan
                           originally chose for weight streaming. I have not
                           measured it for bfp2.

  compute                  the MVMUL itself, in which case no format helps and the
                           ternary roofline is optimistic for this shape.

This measures the first. The DRAM-sharded factory wants a width-sharded in1 whose
shards are contiguous per bank, so the weight is rebuilt that way; the per-core
column count follows from n divided across the DRAM banks (8 on Blackhole).

If DRAM-sharded bfp2 lands near its byte-scaled expectation (~half of bfp4) then
the handshake was the bound and the format does pay. If it lands at bfp4's time
again, the bound is compute and the recommendation is to stop chasing bfp2 on this
shape.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_shard
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_calib import device_kernel_us
from .placement import act_mem, describe, out_mem
from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer


def weight_mb(n: int, k: int, fmt: str) -> float:
    if fmt == "bfp8":
        return (n * k + n * k // 16) / 1e6
    if fmt == "bfp4":
        return (n * k // 2 + n * k // 16) / 1e6
    if fmt == "bfp2":
        return (n * k // 4 + n * k // 16) / 1e6
    raise ValueError(fmt)


def all_kernel_us(ttnn, device) -> list[float]:
    """DEVICE KERNEL DURATION of every program since the last readback, in us."""
    ttnn.ReadDeviceProfiler(device)
    key = "DEVICE KERNEL DURATION [ns]"
    return [int(p.program_analyses_results[key].duration) / 1e3
            for progs in ttnn.get_latest_programs_perf_data().values() for p in progs
            if key in p.program_analyses_results]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_down")
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--fmts", default="bfp2,bfp4,bfp8")
    ap.add_argument("--workers", default="1", help="num_workers_per_dram_bank values (1-3)")
    ap.add_argument("--tile-h", type=int, default=32, help="in0 tile height (32 = full tile, <32 = tiny tile)")
    ap.add_argument("--cores", default="8,16,32", help="activation shard cores (must divide K tiles)")
    ap.add_argument("--k-blocks", default="1,2,4,8,17,34,68",
                    help="in0_block_w values for the sharded config")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--prof-calls", type=int, default=9, help="profiled calls; device time is their median")
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_bfp2_shard.jsonl"))
    args = ap.parse_args()

    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"),
                     ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn

    k, n = args.k, args.n
    k_tiles, n_tiles = k // 32, n // 32
    name = args.shape
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
    codes = np.random.default_rng(0).integers(-1, 2, (n, k)).astype(np.float32)
    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}

    # DRAM-sharded in1, built the way tt_transformers' dram_matmul_config does it:
    # weight width-sharded over the 8 DRAM banks (P150), activation width-sharded in
    # L1 over `cores` cores, per_core_N = ceil(n_tiles / cores). The DRAM-sharded
    # factory requires a sharded in0 (the 10:40 run died on that).
    # BFLOAT2_B cannot go torch->device straight into a sharded config (py_to_tt_tensor
    # picks a BFLOAT2_B host buffer, which pytensor cannot build), so tilize on host
    # first and then move the packed tiles to the device.
    num_banks = 8
    print(f"{name} K={k} N={n} ({k_tiles} K-tiles, {n_tiles} N-tiles), {describe(name)}")
    print(f"{'fmt':5s} {'cores':>5} {'kblk':>4} {'MB':>8} {'dev us':>8} {'GB/s':>7} {'%pk':>5}")
    rc = 0
    try:
        x_torch = torch.randn((1, 1, 32, k), dtype=torch.bfloat16)
        in0_tile = ttnn.Tile((args.tile_h, 32))
        ref = x_torch[0, 0, :args.tile_h].double() @ torch.from_numpy(codes.T).double()
        sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                             ttnn.CoreCoord(num_banks - 1, 0))}),
                           (k, n // num_banks), ttnn.ShardOrientation.ROW_MAJOR))
        res = {}
        for fmt in args.fmts.split(","):
            mb = weight_mb(n, k, fmt)
            wt = ttnn.to_device(
                ttnn.from_torch(torch.from_numpy(codes.T.copy()), dtype=dtypes[fmt], layout=ttnn.TILE_LAYOUT),
                device, memory_config=sharded_mem)
            for cores in [int(c) for c in args.cores.split(",")]:
                if k_tiles % cores:
                    continue
                cols = min(8, cores)
                act_mem_sh = ttnn.create_sharded_memory_config(
                    shape=(args.tile_h, k // cores), core_grid=ttnn.CoreGrid(x=cols, y=cores // cols),
                    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True)
                x = ttnn.from_torch(x_torch[:, :, :args.tile_h], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    tile=in0_tile, device=device, memory_config=act_mem_sh)
                for kb, wk in [(int(b), int(w)) for b in args.k_blocks.split(",") for w in args.workers.split(",")]:
                    per_core_k = k_tiles // cores
                    if k_tiles % kb or (per_core_k % kb and kb % per_core_k):
                        continue
                    progcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=kb, per_core_M=1, per_core_N=-(-n_tiles // cores),
                        fused_activation=None, num_workers_per_dram_bank=wk)
                    def once(wt=wt, x=x, progcfg=progcfg):
                        y = ttnn.linear(x, wt, compute_kernel_config=ckc,
                                        program_config=progcfg,
                                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(y)

                    try:
                        med, samples = timer.best(once)
                        device_kernel_us(ttnn, device)  # drain: a read returns the FIRST program since the last one
                        for _ in range(args.prof_calls):
                            once()
                        devs = all_kernel_us(ttnn, device)
                        dev = float(np.median(devs)) if devs else med * 1e6
                        y = ttnn.to_torch(ttnn.linear(x, wt, compute_kernel_config=ckc, program_config=progcfg,
                                                      memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)).double()
                        y = y.reshape(-1, y.shape[-1])[:args.tile_h, :n]
                        pcc = float(np.corrcoef(y.flatten().numpy(), ref.flatten().numpy())[0, 1])
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} {cores:5d} {kb:4d} w{wk} FAILED "
                              f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                        rc = 1
                        continue
                    gbs = mb * 1e3 / dev  # MB/us -> GB/s
                    print(f"{fmt:5s} {cores:5d} {kb:4d} w{wk} {mb:8.2f} {dev:8.1f} "
                          f"{gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:4.1f}%  pcc {pcc:.7f}  n={len(devs)} [{min(devs or [0]):.1f}-{max(devs or [0]):.1f}]", flush=True)
                    res[(fmt, cores, kb, wk)] = dev
                    rec = new_record(f"shard_{name}", K=k, N=n, fmt=fmt, in0_block_w=kb, cores=cores, workers=wk, tile_h=args.tile_h, pcc=pcc,
                                     cfg="dram-sharded", placement=describe(name))
                    rec.samples = samples
                    rec.meta["device_us"] = devs
                    append_jsonl(args.out, rec, int(mb * 1e6))
                ttnn.deallocate(x)
            ttnn.deallocate(wt)
        print()
        for (fmt, cores, kb, wk), a in sorted(res.items()):
            b = res.get(("bfp4", cores, kb, wk))
            if fmt == "bfp2" and b:
                print(f"dram-sharded cores={cores:2d} kblk={kb:3d} w{wk}: bfp2 {a:6.1f} bfp4 {b:6.1f} -> {b / a:.2f}x")
        print()
        print("Byte-scaled expectation is ~2.0x (bfp2 moves half of bfp4's bytes).")
        print("Near 2.0x -> the in1 handshake was the bound and the format pays.")
        print("Near 1.0x -> the bound is compute; stop chasing bfp2 on this shape.")
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

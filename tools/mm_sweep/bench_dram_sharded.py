"""
Best-case benchmark of ttnn's DRAM-sharded matmul (MatmulMultiCoreReuseMultiCastDRAMSharded)
on the skinny / DRAM-bound shapes, to see what in1-read bandwidth the *proven* DRAM-sharded
reader achieves vs our minimal_matmul interleaved path (~290 GB/s on 32x6144x1536).

in1 (weights) is WIDTH_SHARDED into DRAM banks (one contiguous shard per bank); in0 (activation)
is WIDTH_SHARDED into L1 and mcast. This is the recipe from tech_reports/Saturating_DRAM_bandwidth.

bf16 weights (per project constraint: no bfp8). One bf8 reference row optional via env BENCH_BF8=1.

Usage:
  MM_CLOCK_HZ=1.35e9 TT_METAL_DEVICE_PROFILER=1 python bench_dram_sharded.py
  (optional) BENCH_SHAPES="32x6144x1536,32x9216x1536" BENCH_BF8=1
"""
import os, statistics, torch, ttnn

CLOCK = float(os.environ.get("MM_CLOCK_HZ", 1.35e9))
RAW = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
REPS = 5  # measured replays (median); 1 fresh-compile call discarded first


def durs():
    L = open(RAW).read().splitlines()
    h = [x.strip() for x in L[1].split(",")]
    ix = {k: i for i, k in enumerate(h)}
    zi, ti, ri, ty = ix["zone name"], ix["time[cycles since reset]"], ix["run host ID"], ix["type"]
    st, en = {}, {}
    for ln in L[2:]:
        f = ln.split(",")
        if len(f) <= zi or not f[zi].strip().endswith("-FW"):
            continue
        rid, t = f[ri].strip(), f[ti].strip()
        if not rid or not t:
            continue
        t = int(t)
        if f[ty].strip() == "ZONE_START":
            st[rid] = min(st.get(rid, t), t)
        elif f[ty].strip() == "ZONE_END":
            en[rid] = max(en.get(rid, t), t)
    return [en[r] - st[r] for r in sorted(st, key=int) if r in en]


def find_max_subblock(h, w):
    best = (1, 1, 0)
    for sh in range(1, h + 1):
        if h % sh:
            continue
        for sw in range(1, w + 1):
            if w % sw == 0 and sh * sw <= 8 and sh * sw > best[2]:
                best = (sh, sw, sh * sw)
    return best[0], best[1], best[2]


def pad_to_dram_banks(num, num_banks):
    lcm = 32 * num_banks
    r = num % lcm
    return num if r == 0 else num + (lcm - r)


def bench_one(d, M, K, N, in1_dtype, in0_blk_div):
    num_banks = d.dram_grid_size().x
    N_padded = pad_to_dram_banks(N, num_banks)
    num_cores = num_banks
    grid_size = (num_banks, 1)

    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N_padded // num_cores // 32
    sbh, sbw, _ = find_max_subblock(out_block_h, out_block_w)
    if in0_block_w % in0_blk_div:
        return None  # divisor must divide the per-core K width

    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    in1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT, memory_config=interleaved_dram)
    in0_t = ttnn.interleaved_to_sharded(
        in0_t, grid_size, [M, in0_block_w * 32], ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )

    dg = ttnn.CoreCoord(d.dram_grid_size().x - 1, d.dram_grid_size().y - 1)
    in1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), dg)})
    in1_spec = ttnn.ShardSpec(in1_grid, [K, N_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR)
    in1_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_spec)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, device=d, layout=ttnn.TILE_LAYOUT, memory_config=in1_cfg)

    pcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // in0_blk_div, per_core_M=out_block_h, per_core_N=out_block_w, fused_activation=None
    )
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    ref = (in0.float() @ in1.float()).flatten()
    ref = ref - ref.mean()
    refn = ref.norm()

    def call():
        return ttnn.matmul(
            in0_t, in1_t, program_config=pcfg, memory_config=out_l1, dtype=ttnn.bfloat16, compute_kernel_config=cc
        )

    pcc = 0.0
    try:
        o = call()
        ot = ttnn.to_torch(o)[..., :N].flatten().float()[: ref.numel()]
        ot = ot - ot.mean()
        pcc = float(torch.dot(ot, ref) / (ot.norm() * refn + 1e-12))
        o.deallocate()
        ttnn.synchronize_device(d)
        for _ in range(REPS):
            o = call()
            o.deallocate()
        ttnn.synchronize_device(d)
        ttnn.ReadDeviceProfiler(d)
    except Exception as e:
        in0_t.deallocate()
        in1_t.deallocate()
        return {"err": str(e)[:80]}
    in0_t.deallocate()
    in1_t.deallocate()
    return {
        "pcc": pcc,
        "in0_block_w": in0_block_w // in0_blk_div,
        "grid": grid_size,
        "num_banks": num_banks,
        "N_padded": N_padded,
    }


def main():
    shapes = os.environ.get("BENCH_SHAPES", "32x6144x1536")
    shapes = [tuple(int(x) for x in s.split("x")) for s in shapes.split(",")]
    dtypes = [("bf16", ttnn.bfloat16)]
    if os.environ.get("BENCH_BF8") == "1":
        dtypes.append(("bf8", ttnn.bfloat8_b))
    divisors = [1, 2, 4, 8]  # in0_block_w param = (K/cores/32)//div ; smaller param = more pipelining

    for M, K, N in shapes:
        in1_bytes = K * N * 2  # bf16 weights read once
        print(f"\n===== {M}x{K}x{N}  (in1={in1_bytes/1e6:.2f} MB bf16) =====")
        print(f"{'dtype':>5} {'in0_blk_w':>9} {'us':>8} {'GB/s':>8} {'pcc':>8}   note")
        for dn, dt in dtypes:
            best = None
            for div in divisors:
                if os.path.exists(RAW):
                    os.remove(RAW)
                d = ttnn.open_device(device_id=0)
                d.enable_program_cache()
                r = bench_one(d, M, K, N, dt, div)
                ttnn.close_device(d)  # CSV only flushes on device close
                ds = durs() if os.path.exists(RAW) else []
                if r is None:
                    continue
                if "err" in r:
                    print(f"{dn:>5} {'div'+str(div):>9} {'':>8} {'':>8} {'':>8}   ERR {r['err']}")
                    continue
                # last REPS durations are the measured replays
                meas = ds[-REPS:] if len(ds) >= REPS else ds
                if not meas:
                    continue
                cyc = statistics.median(meas)
                us = cyc / CLOCK * 1e6
                gbs = in1_bytes / (cyc / CLOCK) / 1e9
                tag = "PCC FAIL" if r["pcc"] < 0.97 else ""
                print(f"{dn:>5} {r['in0_block_w']:>9} {us:>8.2f} {gbs:>8.1f} {r['pcc']:>8.4f}   {tag}")
                if r["pcc"] >= 0.97 and (best is None or gbs > best):
                    best = gbs
            if best:
                print(f"{dn:>5} {'BEST':>9} {'':>8} {best:>8.1f}")


if __name__ == "__main__":
    main()

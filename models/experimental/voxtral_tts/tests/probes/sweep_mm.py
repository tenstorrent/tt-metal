"""Program-config sweep for EVERY remaining matmul in Blocks 1 and 2. 6.43 did this for Wo alone
and found the tuned config was a LOSS; this asks the same question of the other four shapes.

Both blocks share dims (DIM=FM_DIM=3072, HIDDEN=FM_HIDDEN=9216, 32 heads x 128), and Block 1's
1 row and Block 2's 6 rows are both ONE tile, so four shapes cover all eight matmul sites:

    wqkv   K=3072 N=6144    Kt= 96  Nt=192
    wo     K=4096 N=3072    Kt=128  Nt= 96      (6.43: default beat the tuned config)
    w1/w3  K=3072 N=9216    Kt= 96  Nt=288
    w2     K=9216 N=3072    Kt=288  Nt= 96      <-- the suspect

WHY w2 IS THE SUSPECT. bw_vs_rows.py measured achieved bandwidth against reduction depth:
K=3072 reached 239 GB/s, K=8192 only 144. w2 has the deepest reduction in the model (Kt=288) and
is read 26x in Block 1 plus 21x in Block 2. If in0_block_w is the reason deep reductions stall,
w2 is where it shows.

ARMS
  * default        -- no program_config, the ttnn heuristic. This is what ships everywhere now.
  * 1D mcast_in0   -- MatmulMultiCoreReuseMultiCast1DProgramConfig over a grid x in0_block_w grid.
                      Splits N across cores and broadcasts in0; the natural batch-1 shape.
  * DRAM-sharded   -- MatmulMultiCoreReuseMultiCastDRAMSharded, weights width-sharded across the
                      8 DRAM banks. 6.28 rejected it on the N150, but it is precisely the config
                      built for decode matmuls, and ceilings.py put us at 45% of DRAM -- so it is
                      the one arm with a mechanism for the gap rather than a constant to retune.

fp32_dest_acc_en=True halves the dest register file, so out_subblock_h*out_subblock_w <= 4.
Every arm is checked for PCC against a float64 reference built from the DEVICE'S OWN quantised
weights, so a config that is merely fast and wrong cannot win.
"""
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
SHAPES = [("wqkv ", 3072, 6144), ("wo   ", 4096, 3072),
          ("w1/w3", 3072, 9216), ("w2   ", 9216, 3072)]
GRIDS = [(8, 2), (8, 3), (8, 4), (8, 6), (8, 8), (8, 9), (8, 12), (12, 8), (12, 6), (13, 10)]
REPS = 200


def bench(dev, fn, reps=REPS):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    dev = open_device()
    nbanks = dev.dram_grid_size().x
    print(f"DRAM banks: {nbanks}\n")
    try:
        for lbl, K, N in SHAPES:
            Kt, Nt = K // 32, N // 32
            torch.manual_seed(0)
            wt = torch.randn(K, N) * 0.02
            xt = torch.randn(1, 1, K) * 0.02
            w = ttnn.from_torch(wt.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                device=dev)
            x = ttnn.from_torch(xt.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev)
            ref = (ttnn.to_torch(x).double().reshape(1, K) @ ttnn.to_torch(w).double())
            wbytes = K * N * 1.0625

            rows = []

            def record(name, fn):
                try:
                    out = fn()
                    got = ttnn.to_torch(out).double().reshape(-1)[:N].reshape(1, N)
                    p = pcc(got, ref)
                    if p < 0.99:
                        rows.append((name, float("inf"), p, 0.0)); return
                    us = bench(dev, fn)
                    rows.append((name, us, p, wbytes / (us / 1e6) / 1e9))
                except Exception:
                    pass

            record("default", lambda: ttnn.linear(x, w, compute_kernel_config=COMPUTE))

            # ---- 1D multicast: split N across cores, broadcast in0 ----
            for gx, gy in GRIDS:
                nc = gx * gy
                pcn = (Nt + nc - 1) // nc
                osw = next((s for s in (4, 2, 1) if pcn % s == 0), 1)
                for ibw in (1, 2, 4, 8, 16, 32):
                    if Kt % ibw:
                        continue
                    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw,
                        out_subblock_h=1, out_subblock_w=osw, per_core_M=1, per_core_N=pcn,
                        fuse_batch=True, fused_activation=None, mcast_in0=True)
                    record(f"1D {gx}x{gy:<2} ibw={ibw:<2} pcn={pcn}",
                           lambda c=cfg: ttnn.linear(x, w, program_config=c,
                                                     compute_kernel_config=COMPUTE))

            # ---- DRAM-sharded: weights striped across the DRAM banks ----
            try:
                wd = ttnn.from_torch(
                    wt.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.create_sharded_memory_config(
                        (K, N // nbanks), ttnn.CoreGrid(y=1, x=nbanks),
                        ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR))
                for gx, gy in ((8, 8), (8, 12), (12, 8), (13, 10)):
                    nc = gx * gy
                    xs = ttnn.to_memory_config(x, ttnn.create_sharded_memory_config(
                        (32, K // nc), ttnn.CoreGrid(y=gy, x=gx), ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR))
                    for ibw in (1, 2, 4):
                        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=ibw, per_core_M=1,
                            per_core_N=(Nt + nc - 1) // nc, fused_activation=None)
                        record(f"DRAMsh {gx}x{gy:<2} ibw={ibw}",
                               lambda c=cfg, xx=xs: ttnn.linear(
                                   xx, wd, program_config=c, compute_kernel_config=COMPUTE,
                                   memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG))
            except Exception as e:
                print(f"  (DRAM-sharded setup failed for {lbl}: {type(e).__name__})")

            rows.sort(key=lambda r: r[1])
            base = next((r[1] for r in rows if r[0] == "default"), float("nan"))
            print(f"=== {lbl}  K={K} N={N}  (Kt={Kt} Nt={Nt})   "
                  f"{sum(1 for r in rows if r[1] < float('inf'))} configs built ===")
            print(f"  {'config':<24} {'us':>8} {'GB/s':>6} {'vs default':>11} {'PCC':>10}")
            for nm, us, p, gbs in rows[:8]:
                if us == float("inf"):
                    continue
                print(f"  {nm:<24} {us:>8.1f} {gbs:>6.0f} {base-us:>+10.1f}u {p:>10.7f}"
                      f"{'   <-- ships' if nm == 'default' else ''}")
            if not any(r[0] == "default" for r in rows[:8]):
                d = next(r for r in rows if r[0] == "default")
                print(f"  {'default':<24} {d[1]:>8.1f} {d[3]:>6.0f} {0.0:>+10.1f}u {d[2]:>10.7f}"
                      f"   <-- ships")
            print()
            del w, x
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

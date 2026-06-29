"""
Sweep (grid, S, Pk, blocking) for ONE shape to find the best config, exploiting sub-grids to unlock
arbitrary K-partitioning (Pk constrained only by the SUB-grid's grid.y, not the full device grid.y).
Realizes "step 1" (round-down to a sub-grid) via the config's compute_with_storage_grid_size -- no
kernel change. For each grid.y in a range, enumerate feasible (S,Pk) (divisors of that grid.y, Pk|K,
K/Pk>=2) x pruned blocks. Reports best by absolute us / effective DRAM BW (the real goal: approach
peak memory BW). util% is vs the FULL device peak for comparability with the 11x10 sweep.

  MM_CLOCK_HZ=1.35e9 TT_METAL_DEVICE_PROFILER=1 python tools/mm_sweep/grid_sweep_one.py M K N
"""
import os, sys, statistics, json, torch, ttnn, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("js", os.path.join(HERE, "joint_sweep.py"))
js = importlib.util.module_from_spec(spec)
spec.loader.exec_module(js)

M, K, N = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (32, 6144, 1536)
Mt, Nt, Kt = M // 32, N // 32, K // 32
WARMUP, REPS, CHUNK, CLOCK, RAW = js.WARMUP, js.REPS, js.CHUNK, js.CLOCK, js.RAW
MIN_BYTES = (M * K + K * N + M * N) * 2  # bf16, each operand once
PEAK_BW = float(os.environ.get("MM_PEAK_BW_GBPS", 500)) * 1e9
GY_RANGE = [int(x) for x in os.environ.get("MM_GY_RANGE", "4,5,6,7,8,9,10").split(",")]

if os.path.exists(RAW):
    os.remove(RAW)
d = ttnn.open_device(device_id=0)
d.enable_program_cache()
gs = d.compute_with_storage_grid_size()
GX, GY_DEV = gs.x, gs.y
PEAK_FULL = GX * GY_DEV * 2048 * CLOCK
cc = ttnn.init_device_compute_kernel_config(
    d.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
ta = torch.randn(M, K, dtype=torch.bfloat16)
tb = torch.randn(K, N, dtype=torch.bfloat16)
ref = (ta.float() @ tb.float()).flatten()
rv = ref - ref.mean()
rvn = rv.norm()
a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"


def pcc(t):
    ov = t.flatten().float()[: rv.numel()]
    ov = ov - ov.mean()
    return float(torch.dot(ov, rv) / (ov.norm() * rvn + 1e-12))


man = []  # (tag, ok, n_exec); tag=(gy,S,Pk,mb,kb,nb,sbh,sbw)
transpose = M > N
for gy in GY_RANGE:
    if gy > GY_DEV:
        continue
    for S, Pk in js.feasible_spk(Kt, gy):
        js.clear_spk()
        os.environ["TT_MM_NUM_SLICES"] = str(S)
        if Pk > 1:
            os.environ["TT_MM_K_SLICES"] = str(Pk)
            os.environ["TT_MM_K_FUSED"] = "1"
        pcM, pcN = js.percore(Mt, Nt, S, Pk, GX, gy, transpose)
        for mb, kb, nb, sbh, sbw in js.gen_blocks(pcM, pcN, Kt // Pk):
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, gy),
            )
            d.clear_program_cache()
            ok = True
            fresh = 0.0
            n_exec = 0
            try:
                for j in range(1 + WARMUP):
                    o = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    n_exec += 1
                    if j == 0:
                        fresh = pcc(ttnn.to_torch(o))
                    o.deallocate()
                ttnn.synchronize_device(d)
                for _ in range(REPS):
                    o = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    n_exec += 1
                    o.deallocate()
                ttnn.synchronize_device(d)
                ttnn.ReadDeviceProfiler(d)
                ok = fresh >= 0.99
            except Exception:
                ok = False
            man.append(((gy, S, Pk, mb, kb, nb, sbh, sbw), ok, n_exec))
a.deallocate()
b.deallocate()
ttnn.close_device(d)

ds = js.durs()
i = 0
recs = []
for tag, ok, n_exec in man:
    seg = ds[i : i + n_exec]
    i += n_exec
    if ok and n_exec == CHUNK and len(seg) == CHUNK:
        us = statistics.median(seg[-REPS:]) / 1000
        recs.append((tag, us))
recs.sort(key=lambda r: r[1])

print(
    f"\n=== {M}x{K}x{N}  (device {GX}x{GY_DEV}, full peak {PEAK_FULL/1e12:.0f} TFLOP/s, in1+={MIN_BYTES/1e6:.1f}MB) ==="
)
print(f"{'rank':>4} {'grid':>7} {'S,Pk':>6} {'mb/kb/nb':>10} {'us':>7} {'GB/s':>6} {'BW%':>5} {'util(full)%':>11}")


def line(rank, tag, us):
    gy, S, Pk, mb, kb, nb, sbh, sbw = tag
    gbps = MIN_BYTES / (us * 1e-6) / 1e9
    util = 100 * 2 * M * K * N / (PEAK_FULL * us * 1e-6)
    print(
        f"{rank:>4} {f'{GX}x{gy}':>7} {f'{S},{Pk}':>6} {f'{mb}/{kb}/{nb}':>10} {us:>7.1f} {gbps:>6.0f} {100*gbps*1e9/PEAK_BW:>5.0f} {util:>11.1f}"
    )


for r, (tag, us) in enumerate(recs[:15]):
    line(r + 1, tag, us)
# best per grid.y
print("\nbest per grid.y:")
seen = {}
for tag, us in recs:
    gy = tag[0]
    if gy not in seen:
        seen[gy] = (tag, us)
for gy in sorted(seen):
    line("", *seen[gy])

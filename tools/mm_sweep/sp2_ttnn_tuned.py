import sys, time, torch, ttnn

M, K, N = (int(x) for x in sys.argv[1].split("x"))
GX, GY = (int(x) for x in sys.argv[2].split("x"))
pm, pn, bw, sh, sw = (int(x) for x in sys.argv[3].split(","))  # per_core_M,N, in0_block_w, subblock h,w
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 8

device = ttnn.open_device(device_id=0)
device.enable_program_cache()
cc = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
    in0_block_w=bw,
    out_subblock_h=sh,
    out_subblock_w=sw,
    per_core_M=pm,
    per_core_N=pn,
    transpose_mcast=False,
)
print(f"START {M}x{K}x{N} grid{GX}x{GY} pcM{pm} pcN{pn} bw{bw} sb{sh}x{sw}", flush=True)
a = ttnn.from_torch(
    torch.randn(M, K, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
b = ttnn.from_torch(
    torch.randn(K, N, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
for _ in range(2):
    ttnn.matmul(a, b, program_config=pcfg, compute_kernel_config=cc).deallocate()
ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(reps):
    ttnn.matmul(a, b, program_config=pcfg, compute_kernel_config=cc).deallocate()
ttnn.synchronize_device(device)
dt = (time.perf_counter() - t0) / reps
tf = 2.0 * M * K * N / dt / 1e12
peak = 304.0 * (GX * GY) / 110.0
print(
    f"RESULT grid{GX}x{GY} pcM{pm}pcN{pn}bw{bw}sb{sh}x{sw} ms={dt*1e3:.3f} TFLOPs={tf:.1f} pct304={tf/304*100:.1f} pctGridPeak={tf/peak*100:.1f}",
    flush=True,
)
ttnn.close_device(device)

import sys, time, torch, ttnn

M, K, N = (int(x) for x in sys.argv[1].split("x"))
mb, kb, nb, sbh, sbw = (int(x) for x in sys.argv[2].split(","))
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GX, GY = 11, 10

device = ttnn.open_device(device_id=0)
device.enable_program_cache()
cc = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
cfg = ttnn.MinimalMatmulConfig(
    M_block_size=mb,
    K_block_size=kb,
    N_block_size=nb,
    subblock_h=sbh,
    subblock_w=sbw,
    compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
)
print(f"START {M}x{K}x{N} blk mb{mb}kb{kb}nb{nb}sb{sbh}x{sbw} grid{GX}x{GY}", flush=True)
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
print("warmup...", flush=True)
for _ in range(2):
    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
ttnn.synchronize_device(device)
print("timing...", flush=True)
t0 = time.perf_counter()
for _ in range(reps):
    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
ttnn.synchronize_device(device)
dt = (time.perf_counter() - t0) / reps
tf = 2.0 * M * K * N / dt / 1e12
print(f"RESULT blk mb{mb}kb{kb}nb{nb}sb{sbh}x{sbw} ms={dt*1e3:.3f} TFLOPs={tf:.1f} pct304={tf/304*100:.1f}", flush=True)
ttnn.close_device(device)

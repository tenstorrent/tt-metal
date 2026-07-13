import sys, time, torch, ttnn

M, K, N = (int(x) for x in sys.argv[1].split("x"))
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

device = ttnn.open_device(device_id=0)
device.enable_program_cache()
cc = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
print(f"START {M}x{K}x{N} reps={reps}", flush=True)
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
print("compiling+warmup...", flush=True)
for _ in range(2):
    ttnn.matmul(a, b, compute_kernel_config=cc).deallocate()
ttnn.synchronize_device(device)
print("timing...", flush=True)
t0 = time.perf_counter()
for _ in range(reps):
    ttnn.matmul(a, b, compute_kernel_config=cc).deallocate()
ttnn.synchronize_device(device)
dt = (time.perf_counter() - t0) / reps
flop = 2.0 * M * K * N
ai = flop / (2.0 * (M * K + K * N + M * N))
tf = flop / dt / 1e12
print(
    f"RESULT {M}x{K}x{N} AI={ai:.0f} ms={dt*1e3:.3f} TFLOPs={tf:.1f} pct304={tf/304*100:.1f} percore_GFLOPs={tf*1000/110:.1f}",
    flush=True,
)
ttnn.close_device(device)

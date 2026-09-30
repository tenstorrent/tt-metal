"""Loop one projection matmul (model config) for --seconds while the caller samples telemetry."""
import argparse, math, time, torch, ttnn
from models.demos.gemma4_d_p.tt.matmul_config import prefill_matmul_program_config, prefill_1d_matmul_program_config
ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=1024); ap.add_argument("--seconds", type=float, default=30)
a = ap.parse_args()
dev = ttnn.open_device(device_id=0)
K = N = 5376
x = ttnn.from_torch(torch.randn(1, 1, a.m, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
w = ttnn.from_torch(torch.randn(1, 1, K, N) / math.sqrt(K), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
g = dev.compute_with_storage_grid_size()
class W: padded_shape = (1, 1, K, N)
pc = prefill_1d_matmul_program_config(x, W, g) or prefill_matmul_program_config(x, W, 12, g.y, fp32_dest_acc=True)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
out = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
n, t0 = 0, time.time()
print("LOOP_START", flush=True)
while time.time() - t0 < a.seconds:
    for _ in range(200):
        out.deallocate(True); out = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc)
    ttnn.synchronize_device(dev); n += 200
dt = time.time() - t0
print(f"LOOP_DONE m={a.m} iters={n} per_iter_us={1e6*dt/n:.1f} tflops={2*a.m*K*N*n/dt/1e12:.0f}", flush=True)
ttnn.close_device(dev)

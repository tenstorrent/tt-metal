import time, statistics, torch, ttnn
from ttnn.operations.onorm import onorm

HV, V = 32, 128
FLAT = HV * V


def cfg(fid=ttnn.MathFidelity.HiFi4, approx=False, fp32=True, fullsync=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
        dst_full_sync_en=fullsync,
    )


CONFIGS = [
    ("DEFAULT HiFi4/exact/fp32dst", cfg()),
    ("math_approx=True", cfg(approx=True)),
    ("LoFi", cfg(fid=ttnn.MathFidelity.LoFi)),
    ("fp32_dest_acc=False", cfg(fp32=False)),
    ("approx+LoFi+no-fp32dst", cfg(fid=ttnn.MathFidelity.LoFi, approx=True, fp32=False)),
    ("dst_full_sync=True", cfg(fullsync=True)),
]
device = ttnn.open_device(device_id=0)
try:
    b, t = 1, 640
    o = ttnn.from_torch(
        torch.randn(b, t, HV, V, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.randn(b, t, FLAT, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 1, V, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for name, c in CONFIGS:
        onorm(o, g, w, compute_kernel_config=c)
    ttnn.synchronize_device(device)
    res = {n: [] for n, _ in CONFIGS}
    for trial in range(5):  # trial-major interleave
        for name, c in CONFIGS:
            N = 10
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(N):
                out = onorm(o, g, w, compute_kernel_config=c)
            ttnn.synchronize_device(device)
            res[name].append((time.perf_counter() - t0) / N * 1e6)
    base = statistics.median(res["DEFAULT HiFi4/exact/fp32dst"])
    for n, _ in CONFIGS:
        m = statistics.median(res[n])
        print(f"{n:<32} median={m:8.1f} us  speedup_vs_default={base/m:5.3f}  spread={max(res[n])-min(res[n]):5.1f}")
finally:
    ttnn.close_device(device)

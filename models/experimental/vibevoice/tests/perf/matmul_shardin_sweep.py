"""Sharded-INPUT matmul sweep for prefill M=256 shapes: width / height / block shard
the activation in L1, feed to ttnn.linear (HiFi4), vs DRAM-interleaved auto baseline.

  python models/experimental/vibevoice/tests/perf/matmul_shardin_sweep.py 2>&1 | tee \
    models/experimental/vibevoice/lm/matmul_shardin_out.txt
"""
from __future__ import annotations
import time, torch, ttnn
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
G = dev.compute_with_storage_grid_size()
GX, GY = G.x, G.y
print(f"grid {GX}x{GY}={GX*GY}", flush=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
H4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
M = 256
SHAPES = [
    ("qkv", 1536, 2048),
    ("o_proj", 1536, 1536),
    ("gate/up", 1536, 8960),
    ("down", 8960, 1536),
    ("lm_head", 1536, 151936),
]


def tt(x, dt, mem=DRAM):
    return ttnn.as_tensor(x, device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=mem)


def timed(fn, iters=30):
    o = fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6, ttnn.to_torch(o)


def grids(n):
    out = []
    for cy in range(1, GY + 1):
        for cx in range(1, GX + 1):
            if cx * cy == n:
                out.append((cx, cy))
    return out


for name, K, N in SHAPES:
    a_t = torch.randn(1, 1, M, K)
    w_t = torch.randn(K, N) * 0.02
    w = tt(w_t.unsqueeze(0).unsqueeze(0).numpy(), ttnn.bfloat16)
    ref = a_t.to(torch.float32) @ w_t.to(torch.float32)
    a_dram = tt(a_t.numpy(), ttnn.bfloat16)
    print(f"\n===== {name}  M={M} K={K} N={N} =====", flush=True)
    res = []
    us, o = timed(lambda: ttnn.linear(a_dram, w, compute_kernel_config=H4, memory_config=DRAM))
    res.append((us, "auto DRAM-in", comp_pcc(ref, o)[1]))
    print(f"  {us:8.1f}us  auto DRAM-in  pcc={res[-1][2]:.5f}", flush=True)
    # try sharding the activation across several core counts
    for strat, tag in [
        (ttnn.ShardStrategy.WIDTH, "WIDTH"),
        (ttnn.ShardStrategy.HEIGHT, "HEIGHT"),
        (ttnn.ShardStrategy.BLOCK, "BLOCK"),
    ]:
        for ncores in [GX * GY, 64, 32, 16, 8]:
            for cx, cy in grids(ncores)[:1]:
                try:
                    cg = ttnn.CoreGrid(y=cy, x=cx)
                    smem = ttnn.create_sharded_memory_config((M, K), cg, strat, ttnn.ShardOrientation.ROW_MAJOR)
                    a_s = tt(a_t.numpy(), ttnn.bfloat16, smem)
                    for omem, ot in [(DRAM, "outDRAM"), (L1, "outL1")]:
                        us, o = timed(lambda: ttnn.linear(a_s, w, compute_kernel_config=H4, memory_config=omem))
                        p = comp_pcc(ref, o)[1]
                        res.append((us, f"{tag} {cx}x{cy} {ot}", p))
                except Exception as e:
                    pass
    res.sort(key=lambda r: r[0])
    print(f"  --- top 5 (of {len(res)} valid) ---", flush=True)
    for us, tag, p in res[:5]:
        print(f"  {us:8.1f}us  {tag}  pcc={p:.5f}", flush=True)

ttnn.close_device(dev)

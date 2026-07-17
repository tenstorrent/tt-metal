"""Prefill matmul config sweep (M=256, current shapes incl. fused QKV).

For each (K,N) tries: auto config, tuned 2D-mcast configs (grid/in0_block_w/subblock),
HiFi4 vs fp32_dest_acc_en unlock, and DRAM vs L1 output — warm device timing + PCC gate.

  python models/experimental/vibevoice/tests/perf/matmul_prefill_sweep.py 2>&1 | tee \
    models/experimental/vibevoice/lm/matmul_prefill_sweep_out.txt
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
H4nf = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)

M = 256
SHAPES = [
    ("qkv", 1536, 2048),
    ("o_proj", 1536, 1536),
    ("gate/up", 1536, 8960),
    ("down", 8960, 1536),
    ("lm_head", 1536, 151936),
]


def mm2d(cx, cy, bw, pcm, pcn, osh, osw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=bw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def time_mm(a, w, pc, ckc, omem, iters=30):
    try:
        o = ttnn.linear(a, w, program_config=pc, compute_kernel_config=ckc, memory_config=omem)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            o = ttnn.linear(a, w, program_config=pc, compute_kernel_config=ckc, memory_config=omem)
        ttnn.synchronize_device(dev)
        us = (time.perf_counter() - t0) / iters * 1e6
        return us, ttnn.to_torch(o)
    except Exception as e:
        return None, str(e)[:60]


for name, K, N in SHAPES:
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    a_t = torch.randn(1, 1, M, K)
    w_t = torch.randn(K, N) * 0.02
    a = ttnn.as_tensor(a_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
    w = ttnn.as_tensor(
        w_t.unsqueeze(0).unsqueeze(0), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
    )
    ref = a_t.to(torch.float32) @ w_t.to(torch.float32)
    print(f"\n===== {name}  M={M} K={K} N={N}  (Mt={Mt} Kt={Kt} Nt={Nt}) =====", flush=True)
    results = []
    # auto (baseline), HiFi4, DRAM and L1
    for omem, otag in [(DRAM, "DRAM"), (L1, "L1")]:
        us, out = time_mm(a, w, None, H4, omem)
        if us:
            pcc = comp_pcc(ref, out)[1]
            results.append((us, f"auto H4 {otag}", pcc))
            print(f"  {us:8.1f}us  auto H4 {otag}   pcc={pcc:.5f}", flush=True)
    # tuned 2D configs: pick grids where cy|Mt and cx|Nt
    cand = []
    for cy in divisors(Mt):
        if cy > GY:
            continue
        for cx in range(1, GX + 1):
            if Nt % cx:
                continue
            pcm, pcn = Mt // cy, Nt // cx
            for bw in [b for b in divisors(Kt) if b <= 8][:4]:
                # subblock: osh*osw<=4 (fp32_dest True). pick osh dividing pcm, osw dividing pcn
                for osh, osw in [(1, 2), (2, 2), (1, 4), (4, 1), (1, 1)]:
                    if pcm % osh or pcn % osw or osh * osw > 4:
                        continue
                    cand.append((cx, cy, bw, pcm, pcn, osh, osw))
    # de-dup and cap
    seen = set()
    cand2 = []
    for c in cand:
        if c in seen:
            continue
        seen.add(c)
        cand2.append(c)
    cand2 = cand2[:40]
    for cx, cy, bw, pcm, pcn, osh, osw in cand2:
        pc = mm2d(cx, cy, bw, pcm, pcn, osh, osw)
        us, out = time_mm(a, w, pc, H4, DRAM)
        if us:
            pcc = comp_pcc(ref, out)[1]
            results.append((us, f"2D {cx}x{cy} bw{bw} pcm{pcm} pcn{pcn} sb{osh}x{osw} H4 DRAM", pcc))
    results.sort(key=lambda r: r[0])
    print(f"  --- top 5 for {name} ---", flush=True)
    for us, tag, pcc in results[:5]:
        print(f"  {us:8.1f}us  {tag}  pcc={pcc:.5f}", flush=True)

ttnn.close_device(dev)

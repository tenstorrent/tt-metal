# WO / FF2 at the model's config (bench_mm_ablate presets): plain minimal_matmul vs
# dit_minimal_matmul_addcmul_fused(x, w, 1.0, resid, ones) (= resid + x @ w, the residual add in the matmul epilogue),
# traced us per call, and the fused output's PCC vs plain + resid (NEGATIVE_RESULTS 74). The fused path requires the
# residual's tile size to match the weights': W_BF8=1 uses bfp8 weights (bfp4 fails the check). RES=L1 puts the residual
# in L1. Usage: bench_resid_mm.py <ff2|wo> <bs>
import os
import statistics
import sys
import time

sys.path.insert(0, "models/demos/blackhole/pplx_embed_4b/perf_tools")
import torch
from bench_mm_ablate import preset

import ttnn

name, B = sys.argv[1], int(sys.argv[2])
M, K, N, blocks, _, wlayout, outp, *rest = preset(name, B)
in0p = rest[0] if rest else "L1"
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
try:
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=blocks[0],
        K_block_size=blocks[1],
        N_block_size=blocks[2],
        subblock_h=blocks[3],
        subblock_w=blocks[4],
        compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
    )
    ckc = ttnn.init_device_compute_kernel_config(
        D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    pad = -(-N // 256) * 256
    w_mc = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                [K, pad // 8],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        if wlayout == "sharded"
        else ttnn.DRAM_MEMORY_CONFIG
    )
    wdt = ttnn.bfloat4_b if os.getenv("W_BF8") != "1" else ttnn.bfloat8_b
    torch.manual_seed(0)
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N) * 0.02, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc
    )
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(
        torch.randn(1, 1, M, K),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=D,
        memory_config=L1 if in0p == "L1" else DR,
    )
    rmc = DR if os.getenv("RES", "DRAM") == "DRAM" else L1
    res = ttnn.from_torch(
        torch.randn(1, 1, M, N), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=rmc
    )
    ones = ttnn.from_torch(
        torch.ones(1, 1, 1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR
    )
    omc = L1 if outp == "L1" else DR
    plain = lambda: ttnn.experimental.minimal_matmul(
        x, w, compute_kernel_config=ckc, config=cfg, memory_config=omc, dtype=ttnn.bfloat8_b
    )
    fused = lambda: ttnn.experimental.dit_minimal_matmul_addcmul_fused(
        x, w, 1.0, res, ones, config=cfg, memory_config=rmc, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc
    )
    ref = ttnn.to_torch(plain()).float() + ttnn.to_torch(res).float()
    got = ttnn.to_torch(fused()).float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
    for label, fn in (("plain", plain), ("fused resid", fused)):
        for _ in range(2):
            ttnn.deallocate(fn())
        n = 6
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        keep = []
        for _ in range(n):
            o = fn()
            ttnn.deallocate(o) if o.memory_config().buffer_type == ttnn.BufferType.L1 else keep.append(o)
        ttnn.end_trace_capture(D, tid, cq_id=0)
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(D, tid)
        [ttnn.deallocate(o) for o in keep]
        print(
            f"RES {name} bs{B} blocks={blocks} res={os.getenv('RES','DRAM')} {label:12s} {statistics.median(ts):8.1f} us/call",
            flush=True,
        )
    print(f"RES {name} bs{B} fused pcc vs plain+res {pcc:.5f}", flush=True)
finally:
    ttnn.close_device(D)

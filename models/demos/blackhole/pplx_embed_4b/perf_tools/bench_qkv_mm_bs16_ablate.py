# bs16 QKV projection (minimal_matmul, the model's config): compute-bound or data-movement-bound?
# Variants skip the in0 (activation) DRAM/L1 reads, the in1 (weight) reads, and/or the output writes; multicast and
# CB handshakes stay. The kernels' paths are fixed in the program factory, so each variant temporarily patches
# minimal_matmul's matmul_dataflow_common_metal2.hpp (the header the Metal 2.0 kernels it runs include; restored afterwards) and runs in its own process with a fresh JIT cache.
# Usage: bench_qkv_mm_bs16_ablate.py [batch]
import os
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
HDR = os.path.join(
    REPO, "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common_metal2.hpp"
)
B = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--child" else 16
S, K, N = 512, 2560, 6144
VARIANTS = {
    k: v
    for k, v in {  # name: (read in0, read in1, write out)
        "full": (1, 1, 1),
        "no output write": (1, 1, 0),
        "no in0 read": (0, 1, 1),
        "no in1 read": (1, 0, 1),
        "no reads": (0, 0, 1),
        "compute only (no reads, no write)": (0, 0, 0),
    }.items()
    if not os.getenv("ABL_ONLY") or k in os.getenv("ABL_ONLY").split("|")
}


def patch(src, r0, r1, w):
    lines = src.split("\n")
    reads = [i for i, l in enumerate(lines) if l.strip() == "noc.async_read("]
    writes = [i for i, l in enumerate(lines) if l.strip() == "noc.async_write("]
    # read_in0_block_sync: 3 reads (two in3 / main split reads, then the plain one); read_in1_block_sync: 1
    assert len(reads) >= 4 and len(writes) >= 2, (reads, writes)
    in0 = reads[:3]
    in1 = reads[3]
    for i in in0:
        lines[i] = lines[i].replace("noc.async_read(", "if (ABL_R0) noc.async_read(")
    lines[in1] = lines[in1].replace("noc.async_read(", "if (ABL_R1) noc.async_read(")
    for i in writes[:2]:  # write_block_sync, write_block_sync_granular
        lines[i] = lines[i].replace("noc.async_write(", "if (ABL_W) noc.async_write(")
    out = "\n".join(lines)
    return out.replace(
        "#pragma once\n", f"#pragma once\n#define ABL_R0 {r0}\n#define ABL_R1 {r1}\n#define ABL_W {w}\n", 1
    )


def child():
    import torch

    import ttnn

    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        g = D.compute_with_storage_grid_size()
        grid = ttnn.CoreCoord(min(13, g.x), min(10, g.y))
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        ckc = ttnn.init_device_compute_kernel_config(
            D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        wt = torch.randn(1, 1, K, N)
        w = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=D,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for pname, mc in (("L1 in", ttnn.L1_MEMORY_CONFIG), ("DRAM in", ttnn.DRAM_MEMORY_CONFIG)):
            xt = torch.randn(1, 1, B * S, K)
            x = ttnn.from_torch(xt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)

            def fn():
                return ttnn.experimental.minimal_matmul(
                    x,
                    w,
                    compute_kernel_config=ckc,
                    config=cfg,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                )

            # output check: a variant that skips reads or writes must not match (proves the patch was compiled in)
            o = ttnn.to_torch(fn()).float().flatten()[: 64 * N]
            ref = (xt[0, 0, :64] @ wt[0, 0]).flatten()
            pcc = torch.corrcoef(torch.stack([o, ref]))[0, 1].item()
            for _ in range(2):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(D)
            n = 8
            tid = ttnn.begin_trace_capture(D, cq_id=0)
            outs = [fn() for _ in range(n)]
            ttnn.end_trace_capture(D, tid, cq_id=0)
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) / n * 1e6)
            ttnn.release_trace(D, tid)
            [ttnn.deallocate(o) for o in outs]
            ttnn.deallocate(x)
            print(f"CHILD {pname} {statistics.median(ts):.1f} grid={grid.x}x{grid.y} pcc={pcc:.3f}", flush=True)
    finally:
        ttnn.close_device(D)


def main():
    orig = open(HDR).read()
    flop = 2 * B * S * K * N
    try:
        for name, (r0, r1, w) in VARIANTS.items():
            open(HDR, "w").write(patch(orig, r0, r1, w))
            env = dict(os.environ, TT_METAL_CACHE=tempfile.mkdtemp(prefix="mm_abl_cache_"))
            p = subprocess.run(
                [sys.executable, __file__, "--child", str(B)], env=env, capture_output=True, text=True, timeout=900
            )
            res = [l for l in p.stdout.splitlines() if l.startswith("CHILD")]
            if not res:
                tail = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "Error:" in l][:3]
                print(f"RES B{B} {name:36s} FAILED {tail}", flush=True)
            for l in res:
                _, pl, pn, t, grid, pcc = l.split()
                print(
                    f"RES B{B} {pl} {pn:4s} {name:36s} {float(t):7.1f} us/call  {flop / float(t) / 1e6:6.1f} TFLOP/s  {grid}  {pcc}",
                    flush=True,
                )
    finally:
        open(HDR, "w").write(orig)


if __name__ == "__main__":
    if "--child" in sys.argv:
        B = int(sys.argv[sys.argv.index("--child") + 1])
        child()
    else:
        main()

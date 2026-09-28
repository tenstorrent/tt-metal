# minimal_matmul at the model's per-batch configs (captured with capture_qkv_call.py): compute-bound or data-movement-
# bound? Variants skip the in0 (activation) reads, the in1 (weight) reads and / or the output writes; forwarding and CB
# handshakes stay. The kernels' paths are fixed in the program descriptor, so each variant temporarily patches
# matmul_dataflow_common_metal2.hpp (restored afterwards) and runs in its own process with a fresh JIT cache. Each
# variant's output is compared with the full variant's (PCC ~0 shows the patch compiled in).
# Usage: bench_mm_ablate.py <preset> [batch]   presets: ff13 (bs8 / bs16 fused SwiGLU, bs32 plain FF1), ff13plain
#        (the packed w13 shape without the SwiGLU epilogue), qkv, ff2, wo
#        ABL_ONLY="full|no reads" picks variants; MM_BLOCKS=M,K,N,sh,sw overrides the preset's blocks; MM_IN0=L1|DRAM
#        overrides the in0 placement.
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
VARIANTS = {  # name: (read in0, read in1, write out)
    "full": (1, 1, 1),
    "no output write": (1, 1, 0),
    "no in0 read": (0, 1, 1),
    "no in1 read": (1, 0, 1),
    "no reads": (0, 0, 1),
    "compute only (no reads, no write)": (0, 0, 0),
}
VARIANTS = {k: v for k, v in VARIANTS.items() if not os.getenv("ABL_ONLY") or k in os.getenv("ABL_ONLY").split("|")}


def preset(name, B):
    """(M, K, N, blocks, fuse_swiglu, weight layout, out placement[, in0 placement = L1]) as the model runs it at ISL
    512 (capture_qkv_call.py with CAP_N=<N> prints the model's calls)."""
    M = B * 512
    if name == "ff13fused":  # the fused SwiGLU kernel at any batch (bs32 runs unfused in the model)
        return M, 2560, 19456, (8, 8, 8, 1, 8), True, "interleaved", "DRAM"
    if name == "ff13plain":  # the fused kernel's packed shape as a plain matmul (what the SwiGLU epilogue costs)
        return M, 2560, 19456, (8, 8, 8, 1, 8), False, "interleaved", "DRAM"
    if name == "ff13":
        if B in (8, 16):  # fused SwiGLU: packed gate/up weight [2560, 2 * 9728], interleaved
            return M, 2560, 19456, (8, 8, 8, 1, 8) if B == 8 else (4, 20, 8, 1, 4), True, "interleaved", "DRAM"
        return M, 2560, 9728, (8, 8, 8, 1, 8), False, "sharded", "DRAM"  # bs32: FF1 (FF3 identical)
    if name == "qkv":
        return (
            M,
            2560,
            6144,
            (8, 4, 8, 1, 8) if B == 8 else (8, 8, 8, 1, 8),
            False,
            "sharded",
            "L1" if B <= 16 else "DRAM",
        )
    if name == "ff2":  # in0 = the fused SwiGLU output, in DRAM
        return M, 9728, 2560, (16, 8, 8, 1, 8) if B == 8 else (8, 8, 8, 1, 8), False, "sharded", "L1", "DRAM"
    if name == "wo":  # in0 = the concat-free SDPA output, in DRAM
        return M, 4096, 2560, (16, 8, 8, 1, 8) if B == 8 else (8, 8, 8, 1, 8), False, "sharded", "L1", "DRAM"
    raise ValueError(name)


def patch(src, r0, r1, w):
    lines = src.split("\n")
    reads = [i for i, l in enumerate(lines) if l.strip() == "noc.async_read("]
    writes = [i for i, l in enumerate(lines) if l.strip() == "noc.async_write("]
    # read_in0_block_sync: 3 reads (two in3 / main split reads, then the plain one); read_in1_block_sync: 1
    assert len(reads) >= 4 and len(writes) >= 2, (reads, writes)
    for i in reads[:3]:
        lines[i] = lines[i].replace("noc.async_read(", "if (ABL_R0) noc.async_read(")
    lines[reads[3]] = lines[reads[3]].replace("noc.async_read(", "if (ABL_R1) noc.async_read(")
    for i in writes[:2]:  # write_block_sync, write_block_sync_granular
        lines[i] = lines[i].replace("noc.async_write(", "if (ABL_W) noc.async_write(")
    out = "\n".join(lines)
    return out.replace(
        "#pragma once\n", f"#pragma once\n#define ABL_R0 {r0}\n#define ABL_R1 {r1}\n#define ABL_W {w}\n", 1
    )


def child(name, B, out_path):
    import torch

    import ttnn

    M, K, N, blocks, swiglu, wlayout, outp, *rest = preset(name, B)
    in0p = os.getenv("MM_IN0") or (rest[0] if rest else "L1")
    if os.getenv("MM_BLOCKS"):
        blocks = tuple(int(v) for v in os.getenv("MM_BLOCKS").split(","))
    torch.manual_seed(0)
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
        if wlayout == "sharded":  # DRAM width-sharded over the 8 banks, width padded to a multiple of 8 tiles
            pad = -(-N // 256) * 256
            w_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                    [K, pad // 8],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        else:
            w_mc = ttnn.DRAM_MEMORY_CONFIG
        w = ttnn.from_torch(
            torch.randn(1, 1, K, N) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc
        )
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=D,
            memory_config=ttnn.L1_MEMORY_CONFIG if in0p == "L1" else ttnn.DRAM_MEMORY_CONFIG,
        )
        out_mc = ttnn.L1_MEMORY_CONFIG if outp == "L1" else ttnn.DRAM_MEMORY_CONFIG
        kw = dict(compute_kernel_config=ckc, config=cfg, memory_config=out_mc, dtype=ttnn.bfloat8_b)
        if swiglu:
            kw["fuse_swiglu"] = True
        fn = lambda: ttnn.experimental.minimal_matmul(x, w, **kw)
        torch.save(ttnn.to_torch(fn())[..., :64, :].float().flatten().clone(), out_path)
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(D)
        n = 8
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        keep = []
        for _ in range(n):
            o = fn()
            ttnn.deallocate(o) if outp == "L1" else keep.append(o)  # 8 live L1 outputs would not fit
        ttnn.end_trace_capture(D, tid, cq_id=0)
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(D, tid)
        [ttnn.deallocate(o) for o in keep]
        print(f"CHILD {statistics.median(ts):.1f}", flush=True)
    finally:
        ttnn.close_device(D)


def main():
    import torch

    name = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    M, K, N, blocks, swiglu, wlayout, outp, *rest = preset(name, B)
    in0p = os.getenv("MM_IN0") or (rest[0] if rest else "L1")
    flop = 2 * M * K * N
    print(f"RES {name} B{B}: M={M} K={K} N={N} blocks={os.getenv('MM_BLOCKS') or blocks} swiglu={swiglu} "
          f"weights={wlayout} in0={in0p} out={outp}", flush=True)  # fmt: skip
    orig = open(HDR).read()
    ref = None
    try:
        for vname, (r0, r1, w) in VARIANTS.items():
            open(HDR, "w").write(patch(orig, r0, r1, w))
            d = tempfile.mkdtemp(prefix="mm_abl_")
            env = dict(os.environ, TT_METAL_CACHE=os.path.join(d, "cache"))
            out_path = os.path.join(d, "out.pt")
            p = subprocess.run(
                [sys.executable, __file__, "--child", name, str(B), out_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=1200,
            )
            res = [l for l in p.stdout.splitlines() if l.startswith("CHILD")]
            if not res:
                err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "Error:" in l][:3]
                print(f"RES {vname:36s} FAILED {err}", flush=True)
                continue
            t = float(res[0].split()[1])
            o = torch.load(out_path)
            if vname == "full":
                ref = o
            pcc = torch.corrcoef(torch.stack([o, ref]))[0, 1].item() if ref is not None else float("nan")
            print(f"RES {vname:36s} {t:8.1f} us  {flop / t / 1e6:6.1f} TFLOP/s  pcc vs full {pcc:.3f}", flush=True)
    finally:
        open(HDR, "w").write(orig)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()

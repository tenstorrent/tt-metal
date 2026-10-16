# What the fused FF1+FF3 matmul's partial-sum add costs, split into its per-subblock re-inits and its per-tile adds.
# In the last K block every subblock adds the packer-L1-accumulated partial sums of K blocks 0..K-2 back into DST
# (matmul_blocks_swiglu in compute_metal2.cpp): reconfig SrcA to the intermediate, add_reuse_dest_init, one dest-reuse
# add per tile, reconfig back, matmul_block_init. Arms patch that section (compute_metal2.cpp restored afterwards) and
# run bench_mm_ablate.py's ff13fused child under the device profiler, in its own process with a fresh JIT cache.
# Arms: full (as shipped), noadd (re-inits only), noinit (adds only; wrong values; hangs: the adds need their init),
# none (no partials at all; wrong), preload (partials copied into DST before the last K block's matmul accumulates on
# top: no DST move; correct values, different rounding order: rel. RMSE vs float 0.174 -> 0.192 at bs16). The add
# used DST -> SrcB until 2026-10-16; DST -> SrcA (shipped) is bit-identical and 3 / 9 us a call faster at bs16 / 32.
# Usage: bench_ff13_partials.py <batch> [arm ...] [@M,K,N,sh,sw]   (blocks default 4,40,8,1,8)
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR  # noqa: E402
from device_kernel_us import kernel_us  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")

ADD = """                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                            interm_dfb, (M_start + h) * full_N_block_tiles + N_start + w, dst++);
"""
INITS = [
    "                reconfig_data_format_srcb(in0_dfb, interm_dfb);\n",
    "                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(interm_dfb);\n",
    "                reconfig_data_format_srcb(interm_dfb, in0_dfb);\n",
    "                matmul_block_init(in0_dfb, in1_dfb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);\n"
    "            }\n            tile_regs_commit();\n",
]


PRE_OLD = """            tile_regs_acquire();
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_dfb,
                    in1_dfb,
                    in0_index,
                    in1_index,
                    0,"""
PRE_NEW = """            tile_regs_acquire();
            if (add_partials) {  // partial sums -> DST, then the last K block accumulates on top
                reconfig_data_format_srca(in1_dfb, interm_dfb);
                copy_init(interm_dfb);
                uint32_t pdst = 0;
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        copy_tile(interm_dfb, (M_start + h) * full_N_block_tiles + N_start + w, pdst++);
                    }
                }
                reconfig_data_format_srca(interm_dfb, in1_dfb);
                matmul_block_init(in0_dfb, in1_dfb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
            }
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_dfb,
                    in1_dfb,
                    in0_index,
                    in1_index,
                    0,"""
POST_OLD = "            if (add_partials) {\n                // DST -> SrcA, partial-sum tile -> SrcB"
POST_NEW = "            if (false) {\n                // DST -> SrcA, partial-sum tile -> SrcB"


def patch(src, arm):
    if arm == "preload":
        assert src.count(PRE_OLD) == 1 and src.count(POST_OLD) == 1
        return src.replace(PRE_OLD, PRE_NEW).replace(POST_OLD, POST_NEW)
    assert src.count(ADD) == 1
    for s in INITS:
        assert src.count(s) == 1, s
    if arm in ("noadd", "none"):
        src = src.replace(ADD, ADD.replace("add_reuse_dest_tiles", "if (0) add_reuse_dest_tiles"))
    if arm in ("noinit", "none"):
        for s in INITS:
            body = s.split("\n")[0]
            src = src.replace(body + "\n", body.replace(body.strip(), "if (0) " + body.strip()) + "\n", 1)
    return src


def main():
    B = sys.argv[1]
    rest = sys.argv[2:]
    blocks = next((a[1:] for a in rest if a.startswith("@")), "4,40,8,1,8")
    arms = [a for a in rest if not a.startswith("@")] or ["full", "noadd", "none", "preload", "full", "preload"]
    orig = open(COMPUTE).read()
    try:
        for arm in arms:
            open(COMPUTE, "w").write(patch(orig, arm))
            d = tempfile.mkdtemp(prefix="ff13p_")
            env = dict(
                os.environ,
                TT_METAL_CACHE=os.path.join(d, "cache"),
                TT_METAL_DEVICE_PROFILER="1",
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="40000",
                TT_METAL_PROFILER_DIR=os.path.join(d, "prof"),
                MM_BLOCKS=blocks,
            )
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", "ff13fused", B, d + "/o.pt"],
                env=env, capture_output=True, text=True, timeout=1200,
            )  # fmt: skip
            if p.returncode:
                err = [
                    l.split("TT_THROW: ")[-1][:200] for l in (p.stdout + p.stderr).splitlines() if "error" in l.lower()
                ]
                print(f"RES ff13 B{B} {arm:7s} FAILED {err[:3]}", flush=True)
                continue
            us = sorted(kernel_us(os.path.join(d, "prof")).values(), key=len)[-1]
            print(
                f"RES ff13 B{B} blocks {blocks} {arm:7s} {statistics.median(us):8.1f} us/call device  out {d}/o.pt",
                flush=True,
            )
    finally:
        open(COMPUTE, "w").write(orig)


if __name__ == "__main__":
    main()

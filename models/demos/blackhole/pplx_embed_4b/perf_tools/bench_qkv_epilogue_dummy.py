# How much epilogue work can the QKV minimal_matmul hide in its last K block? Sizes a fused QK-norm + RoPE epilogue
# (which would retire the heads op) before building it. Dummy work is patched into the plain path's last K block:
# - pack-thread SFPU: per DST tile, SFPU_OPS dependent SFPMADs per row vector, run on the packer's DST half after the
#   math thread's commit (the fused SwiGLU's mechanism), so it can overlap the math thread's next subblock;
# - math thread: per DST tile, MATH_OPS dest-reuse LoFi multiplies by an in1 tile before the commit (what loading
#   cos / sin or an FPU RoPE pass would cost there).
# Values are garbage (no PCC); device kernel time per call is the result. "skel" arms drop the matmul math, so the
# SFPU cost per tile can be calibrated against the skeleton. Patches compute_metal2.cpp (restored afterwards) and runs
# bench_mm_ablate.py's child per arm under the device profiler, in its own process with a fresh JIT cache.
# Usage: bench_qkv_epilogue_dummy.py <batch> [arm ...]   arm = <sfpu ops>,<math ops>[,skel][@M,K,N,sh,sw]
#        e.g. 0,0  8,0  0,2  8,0,skel  16,0@4,40,8,1,8
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR  # noqa: E402
from device_kernel_us import kernel_us  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")

DUMMY_FN = """#include "swiglu_sfpu.hpp"

#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
template <int OPS, int ITERATIONS = 8>
inline void calculate_dummy_epi(const uint a, const uint, const uint c) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * 32];
        sfpi::vFloat y = x;
#pragma GCC unroll 128
        for (int i = 0; i < OPS; i++) {
            y = y * 0.5f + x;
        }
        sfpi::dst_reg[c * 32] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
"""

SIG_OLD = "    const uint32_t subblock_w) {\n    uint32_t in0_index_offset = 0;\n"
SIG_NEW = "    const uint32_t subblock_w,\n    const bool dummy_last = false) {\n    uint32_t in0_index_offset = 0;\n"
COMMIT_OLD = (
    "                in1_index += full_N_block_tiles;\n            }\n            tile_regs_commit();\n"
    "            tile_regs_wait();\n            uint32_t write_dst_index = 0;\n"
)
COMMIT_NEW = """                in1_index += full_N_block_tiles;
            }
            if (DUMMY_MATH_OPS && dummy_last) {
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in1_dfb);
                for (uint32_t r = 0; r < DUMMY_MATH_OPS; r++) {
                    for (uint32_t t = 0; t < subblock_h * subblock_w; t++) {
                        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in1_dfb, t, t);
                    }
                }
                matmul_block_init(in0_dfb, in1_dfb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (DUMMY_SFPU_OPS && dummy_last) {
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t t = 0; t < subblock_h * subblock_w; t++) {
                    PACK((SFPU_BINARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_dummy_epi, (DUMMY_SFPU_OPS_T, 8), t, t, t,
                        VectorMode::RC)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            }
            uint32_t write_dst_index = 0;
"""
INIT_OLD = "    matmul_init(dfb::in0, dfb::in1);\n"
INIT_NEW = INIT_OLD + "    if constexpr (DUMMY_SFPU_OPS) {\n        PACK((minimal_matmul_swiglu_tile_init()));\n    }\n"
CALL_OLD = (
    "                        current_subblock_h,\n                        current_subblock_w);\n                }"
)
CALL_NEW = (
    "                        current_subblock_h,\n                        current_subblock_w,\n"
    "                        k_block == K_num_blocks - 1);\n                }"
)
MM_OLD = "            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {\n                matmul_block(\n"


def patch_compute(src, sfpu, math, skel):
    for old, new in ((SIG_OLD, SIG_NEW), (COMMIT_OLD, COMMIT_NEW), (INIT_OLD, INIT_NEW), (CALL_OLD, CALL_NEW)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    assert src.count('#include "swiglu_sfpu.hpp"\n') == 1
    src = src.replace('#include "swiglu_sfpu.hpp"\n', DUMMY_FN, 1)
    if skel:  # the plain path's matmul_block calls only (matmul_blocks_swiglu's loop is indented the same: both)
        assert src.count(MM_OLD) == 2
        src = src.replace(MM_OLD, MM_OLD.replace("matmul_block(\n", "if (0) matmul_block(\n"))
    defs = f"#define DUMMY_SFPU_OPS {sfpu}\n#define DUMMY_SFPU_OPS_T {max(sfpu, 1)}\n#define DUMMY_MATH_OPS {math}\n"
    return src.replace(
        '#include "api/compute/compute_kernel_api.h"\n', defs + '#include "api/compute/compute_kernel_api.h"\n', 1
    )


def parse_arm(a):
    blocks = None
    if "@" in a:
        a, blocks = a.split("@")
    f = a.split(",")
    return int(f[0]), int(f[1]), len(f) > 2 and f[2] == "skel", blocks


def main():
    B = sys.argv[1]
    arms = sys.argv[2:] or [
        "0,0",
        "4,0",
        "8,0",
        "16,0",
        "32,0",
        "0,1",
        "0,2",
        "0,4",
        "0,0,skel",
        "8,0,skel",
        "32,0,skel",
    ]
    orig = open(COMPUTE).read()
    try:
        for arm in arms:
            sfpu, math, skel, blocks = parse_arm(arm)
            open(COMPUTE, "w").write(patch_compute(orig, sfpu, math, skel))
            d = tempfile.mkdtemp(prefix="qkv_epi_")
            env = dict(
                os.environ,
                TT_METAL_CACHE=os.path.join(d, "cache"),
                TT_METAL_DEVICE_PROFILER="1",
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="40000",
                TT_METAL_PROFILER_DIR=os.path.join(d, "prof"),
            )
            if blocks:
                env["MM_BLOCKS"] = blocks
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", "qkv", B, os.path.join(d, "o.pt")],
                env=env, capture_output=True, text=True, timeout=1200,
            )  # fmt: skip
            us = kernel_us(os.path.join(d, "prof")) if p.returncode == 0 else {}
            label = f"sfpu {sfpu:3d} math {math} {'skel' if skel else 'full'} blocks {blocks or 'preset'}"
            if not us:
                err = [l.split("TT_THROW: ")[-1][:160] for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l][
                    :2
                ]
                print(f"RES qkv B{B} {label:44s} FAILED {err}", flush=True)
                continue
            per = sorted(us.values(), key=len)[-1]  # the 8-call trace
            per = sorted(per)[len(per) // 2]
            print(f"RES qkv B{B} {label:44s} {per:8.1f} us/call device", flush=True)
    finally:
        open(COMPUTE, "w").write(orig)


if __name__ == "__main__":
    main()

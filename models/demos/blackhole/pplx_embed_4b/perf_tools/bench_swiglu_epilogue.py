# Fused-SwiGLU minimal_matmul (bs8 / bs16 config): where the epilogue's time goes. Variants of compute_metal2.cpp's
# swiglu_block (patched in place, restored afterwards; each in its own process with a fresh JIT cache): the inits
# hoisted out of the tile loop (one gate/up pair per DST session kept), and the SiLU, the multiply or both dropped
# (wrong outputs, timing only). Output compared with the baseline's (exact= only meaningful for 'hoisted inits').
# Since NEGATIVE_RESULTS 58 the kernel applies SwiGLU in the last K block and swiglu_block only runs with a fused
# bias, so 'base' is the landed path and the other variants reproduce the §58 numbers only on the kernel before it.
# Usage: bench_swiglu_epilogue.py [batch ...]   (default 8 16; EPI_ONLY="base|hoisted inits" picks variants)
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
SRC = os.path.join(REPO, "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute_metal2.cpp")
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_mm_ablate.py")

LOOP_HEAD = "    for (uint32_t m = 0; m < M_block_tiles; m++) {\n        const uint32_t row_base = m * N_block_tiles;"
COPY_INIT = "            copy_init(in_dfb);\n"
SILU = "            silu_tile_init();\n            silu_tile(GATE_DST);\n"
MUL = "            mul_binary_tile_init();\n            mul_binary_tile(GATE_DST, UP_DST, GATE_DST);\n"
WAIT_PACK = "            tile_regs_wait();\n            pack_tile(GATE_DST, out_dfb);\n"
# silu(gate) * up in one SFPU pass (the moe_compute / moe_gpt swiglu_sfpu.h pattern without GPT-OSS's clamps / alpha /
# up + 1): sigmoid from the bf16 exp and one reciprocal iteration; drivable from the math or the pack thread.
SFPU_SWIGLU = r"""
#if defined(TRISC_PACK) || defined(TRISC_MATH)
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
namespace ckernel::sfpu {
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_swiglu_std(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + _sfpu_exp_21f_bf16_<true>(-gate));
        sfpi::vFloat result = up * (gate * sig);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}
inline void swiglu_std_init() { recip_init<false, false, false>(); }
// silu_tile's own sigmoid and roundings (silu rounded to bf16, then the product): matches silu_tile + mul_binary_tile
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_swiglu_exact(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat silu = gate * _sfpu_sigmoid_<is_fp32_dest_acc_en>(gate);
        if constexpr (!is_fp32_dest_acc_en) {
            silu = sfpi::convert<sfpi::vFloat16b>(silu, sfpi::RoundMode::Nearest);
        }
        sfpi::vFloat result = silu * up;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}
inline void swiglu_exact_init() { sigmoid_init<false>(); }
}  // namespace ckernel::sfpu
namespace ckernel {
inline void swiglu_std_llk_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::swiglu_std_init);
}
inline void swiglu_exact_llk_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::swiglu_exact_init);
}
inline void swiglu_exact_llk(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_swiglu_exact, (false, 8), gate_tile, up_tile, out_tile,
                     VectorMode::RC);
}
inline void swiglu_std_llk(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_swiglu_std, (false, 8), gate_tile, up_tile, out_tile,
                     VectorMode::RC);
}
}  // namespace ckernel
#endif
"""


def variant(src, name):
    if name == "in-loop SFPU (pack)":
        return inloop(src)
    if name == "in-loop add (pack)":
        return "#define SWIGLU_ADD_PARTIALS 1\n" + inloop(src)
    a = src.index("void swiglu_block(")
    b = src.index("#endif  // FUSE_SWIGLU", a)
    blk = src[a:b]
    for s in (LOOP_HEAD, COPY_INIT, SILU, MUL):
        assert blk.count(s) == 1, s
    if name == "hoisted inits":
        blk = blk.replace(COPY_INIT, "").replace(SILU, "            silu_tile(GATE_DST);\n")
        blk = blk.replace(MUL, "            mul_binary_tile(GATE_DST, UP_DST, GATE_DST);\n")
        blk = blk.replace(
            LOOP_HEAD, "    copy_init(in_dfb);\n    silu_tile_init();\n    mul_binary_tile_init();\n" + LOOP_HEAD
        )
    elif name == "no SiLU":
        blk = blk.replace(SILU, "")
    elif name == "no multiply":
        blk = blk.replace(MUL, "")
    elif name == "copy + pack only":
        blk = blk.replace(SILU, "").replace(MUL, "")
    elif name.endswith("SFPU (math)"):
        fn = "swiglu_exact_llk" if name.startswith("exact") else "swiglu_std_llk"
        assert blk.count(WAIT_PACK) == 1
        blk = (
            blk.replace(COPY_INIT, "")
            .replace(SILU, "")
            .replace(MUL, f"            MATH(({fn}(GATE_DST, UP_DST, GATE_DST)));\n")
        )
        blk = blk.replace(LOOP_HEAD, f"    copy_init(in_dfb);\n    MATH(({fn}_init()));\n" + LOOP_HEAD)
        return src[:a] + SFPU_SWIGLU + blk + src[b:]
    elif name.endswith("SFPU (pack)"):
        fn = "swiglu_exact_llk" if name.startswith("exact") else "swiglu_std_llk"
        assert blk.count(WAIT_PACK) == 1
        blk = blk.replace(COPY_INIT, "").replace(SILU, "").replace(MUL, "")
        blk = blk.replace(
            WAIT_PACK,
            "            // tile_regs_wait() that also stalls CFG, so the SETC16 below waits for the math thread\n"
            "            PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG,\n"
            "                             semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));\n"
            "            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));\n"
            f"            PACK(({fn}(GATE_DST, UP_DST, GATE_DST)));\n"
            "            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));\n"
            "            pack_tile(GATE_DST, out_dfb);\n",
        )
        blk = blk.replace(LOOP_HEAD, f"    copy_init(in_dfb);\n    PACK(({fn}_init()));\n" + LOOP_HEAD)
        return src[:a] + SFPU_SWIGLU + blk + src[b:]
    return src[:a] + blk + src[b:]


MM_CALL = """                matmul_blocks(
                    dfb::in0,
                    dfb::in1,
                    dfb::intermediate,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);
"""
EPI_HEAD = """            dfb_intermediate.push_back(out_block_num_tiles);
            pack_reconfig_l1_acc(0);

#ifdef FUSE_SWIGLU
"""
EPI_TAIL = "#elif !defined(FUSE_TERNARY)\n"
INLOOP_FN = r"""
#ifdef FUSE_SWIGLU
// Last K block of a fused-SwiGLU output block: reload the subblock's partial sums (K blocks 0..K-2, packer-L1-accumulated
// in the intermediate) into DST, accumulate the last K block on top, then on the pack thread apply silu(gate) * up to
// each gate/up pair in the packer's DST half and pack the gate slots straight into out -- the SFPU work overlaps the
// math thread's next subblock, and no separate epilogue pass re-reads the intermediate.
void matmul_blocks_swiglu(
    const DFBBindingToken in0_dfb,
    const DFBBindingToken in1_dfb,
    const DFBBindingToken interm_dfb,
    const DFBBindingToken out_dfb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const bool reload) {
    const uint32_t out_full_N = full_N_block_tiles >> 1;
    uint32_t in0_index_offset = 0;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
#ifndef SWIGLU_ADD_PARTIALS
            if (reload) {
                reconfig_data_format_srca(in1_dfb, interm_dfb);
                copy_init(interm_dfb);
                uint32_t d = 0;
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        copy_tile(interm_dfb, (M_start + h) * full_N_block_tiles + N_start + w, d++);
                    }
                }
                reconfig_data_format_srca(interm_dfb, in1_dfb);
                matmul_block_init(in0_dfb, in1_dfb, false, subblock_w, subblock_h, K_block_tiles);
            }
#endif
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(in0_dfb, in1_dfb, in0_index, in1_index, 0, false, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
#ifdef SWIGLU_ADD_PARTIALS
            // the last K block accumulated from zero; add the partial sums once (same rounding order as the packer's
            // L1 accumulation): DST -> SrcB, partial tile -> SrcA
            if (reload) {
                reconfig_data_format_srca(in1_dfb, interm_dfb);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(interm_dfb);
                uint32_t d = 0;
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            interm_dfb, (M_start + h) * full_N_block_tiles + N_start + w, d++);
                    }
                }
                reconfig_data_format_srca(interm_dfb, in1_dfb);
                matmul_block_init(in0_dfb, in1_dfb, false, subblock_w, subblock_h, K_block_tiles);
            }
#endif
            tile_regs_commit();
            // tile_regs_wait() that also stalls CFG, so the SETC16 below waits for the math thread
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w += 2) {
                    const uint32_t g = h * subblock_w + w;
                    PACK((swiglu_exact_llk(g, g + 1, g)));
                }
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w += 2) {
                    pack_tile<true>(h * subblock_w + w, out_dfb, (M_start + h) * out_full_N + ((N_start + w) >> 1));
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}
#endif
"""


def inloop(src):
    for t in (MM_CALL, EPI_HEAD, EPI_TAIL, "void kernel_main() {\n", "    matmul_init(dfb::in0, dfb::in1);\n"):
        assert src.count(t) == 1, t
    src = src.replace("void kernel_main() {\n", SFPU_SWIGLU + INLOOP_FN + "\nvoid kernel_main() {\n")
    src = src.replace(
        "    matmul_init(dfb::in0, dfb::in1);\n",
        "    matmul_init(dfb::in0, dfb::in1);\n#ifdef FUSE_SWIGLU\n    PACK((swiglu_exact_llk_init()));\n#endif\n",
    )
    src = src.replace(
        MM_CALL,
        """#ifdef FUSE_SWIGLU
                if (k_block == K_num_blocks - 1) {
                    if (K_num_blocks > 1) {  // the partial sums become readable for the reload
                        dfb_intermediate.push_back(out_block_num_tiles);
                        dfb_intermediate.wait_front(out_block_num_tiles);
                    }
                    pack_reconfig_l1_acc(0);
                    pack_reconfig_data_format(dfb::out);
                    dfb_out.reserve_back(out_block_num_tiles >> 1);
                    matmul_blocks_swiglu(
                        dfb::in0,
                        dfb::in1,
                        dfb::intermediate,
                        dfb::out,
                        current_M_block_tiles,
                        current_N_block_tiles,
                        N_block_tiles,
                        K_block_tiles,
                        current_subblock_h,
                        current_subblock_w,
                        K_num_blocks > 1);
                    dfb_out.push_back(out_block_num_tiles >> 1);
                    if (K_num_blocks > 1) {
                        dfb_intermediate.pop_front(out_block_num_tiles);
                    }
                } else
#endif
"""
        + MM_CALL,
    )
    a = src.index(EPI_HEAD)
    b = src.index(EPI_TAIL, a)
    src = (
        src[:a]
        + "#ifndef FUSE_SWIGLU\n            dfb_intermediate.push_back(out_block_num_tiles);\n#endif\n            pack_reconfig_l1_acc(0);\n\n#ifdef FUSE_SWIGLU\n            // SwiGLU applied in the last K block (matmul_blocks_swiglu)\n"
        + src[b:]
    )
    return src


VARIANTS = [
    "base",
    "hoisted inits",
    "no SiLU",
    "no multiply",
    "copy + pack only",
    "fused SFPU (math)",
    "fused SFPU (pack)",
    "exact SFPU (math)",
    "exact SFPU (pack)",
    "in-loop SFPU (pack)",
    "in-loop add (pack)",
]
if os.getenv("EPI_ONLY"):
    VARIANTS = [v for v in VARIANTS if v in os.getenv("EPI_ONLY").split("|")]


def reference(B):
    """silu(gate) * up in fp32 for the first 64 rows, from the child's seeded operands (bench_mm_ablate.child)."""
    import torch

    torch.manual_seed(0)
    M, K, N = B * 512, 2560, 19456
    w = torch.randn(1, 1, K, N) * 0.02
    x = torch.randn(1, 1, M, K)
    y = (x[0, 0, :64] @ w[0, 0]).reshape(64, N // 64, 2, 32)  # [rows, output tile, gate / up, 32 columns]
    return (torch.nn.functional.silu(y[:, :, 0]) * y[:, :, 1]).reshape(64, N // 2).flatten()


def main():
    import torch

    batches = [int(b) for b in sys.argv[1:]] or [8, 16]
    orig = open(SRC).read()
    try:
        for B in batches:
            ref = None
            truth = reference(B)
            for v in VARIANTS:
                open(SRC, "w").write(orig if v == "base" else variant(orig, v))
                d = tempfile.mkdtemp(prefix="swiglu_epi_")
                out = os.path.join(d, "out.pt")
                env = dict(os.environ, TT_METAL_CACHE=os.path.join(d, "cache"))
                p = subprocess.run(
                    [sys.executable, CHILD, "--child", os.getenv("EPI_PRESET", "ff13"), str(B), out],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=1200,
                )
                res = [l for l in p.stdout.splitlines() if l.startswith("CHILD")]
                if not res:
                    err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "error:" in l][:3]
                    print(f"RES B{B} {v:18s} FAILED {err}", flush=True)
                    continue
                o = torch.load(out)
                ref = o if v == "base" else ref
                exact = torch.equal(o, ref) if ref is not None else None
                pcc = torch.corrcoef(torch.stack([o, ref]))[0, 1].item() if ref is not None else float("nan")
                maxerr = (o - ref).abs().max().item() if ref is not None else float("nan")
                pcc_t = torch.corrcoef(torch.stack([o, truth]))[0, 1].item()
                print(
                    f"RES B{B} {v:20s} {float(res[0].split()[1]):8.1f} us  exact vs base {exact}  pcc vs base {pcc:.5f}  "
                    f"maxerr {maxerr:.4f}  pcc vs fp32 torch {pcc_t:.5f}",
                    flush=True,
                )
    finally:
        open(SRC, "w").write(orig)


if __name__ == "__main__":
    main()

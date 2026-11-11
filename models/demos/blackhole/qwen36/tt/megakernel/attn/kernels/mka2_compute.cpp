// MKA2 compute, full grid, persistent over NL attention-type layers. Per layer L:
//   STREAM: mixer y = x @ Wmix[:, MIXN cols] (qkv in-proj stand-in).
//   all:    SDPA over this core's KV slice (flash-decode, O_aug = Σ exp(Q·Kᵀ)·[V | 1], fp32 packer L1 accumulation),
//           tree children added into CB_O, the group root normalizes into CB_OA.
//   STREAM: (after GO) fm11 MLP: gate|up -> silu·mul -> h, down -> CB_OUT.
#if defined(MKA_OPTSZ)   // size experiment (spec §17): optimize the whole TU for size; must precede every include
#if MKA_OPTSZ == 1
#pragma GCC optimize("Os")
#else
#pragma GCC optimize("Oz")
#endif
#endif
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#if defined(MK_AR_RSTD) && !defined(MK_AR_RSTD_RED_GDN)   // RED_GDN (iii'): the column sum + transpose run on the GDN host core
#include "api/compute/transpose_dest.h"
#endif
#if defined(MK_AR_RSTD) && (!defined(MK_AR_RSTD_RED_GDN) || (defined(MK_DNORM_SPLIT) && defined(MKA_ROLE_STREAM)))
// one copy of the SFPU fp32 column sum (sfpu_reduce is always-inline): shared by the rstd producer (X^T squares) and the consumer
// (the Σ tile); STREAM text
__attribute__((noinline)) static void mk_colsum(uint32_t idst) {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(idst);
}
#endif
#include "api/debug/kernel_profiler.hpp"
// STEXT (spec v1.138, STREAM kernel text budget): STREAM-only size trims, bit-identical; MKA_NO_STEXT = the old binary
#if defined(MKA_ROLE_STREAM) && !defined(MKA_NO_STEXT)
#define MK_STEXT 1
#define MK_NOUNROLL _Pragma("GCC unroll 1")
#elif defined(MKA_ROLE_EXT) && !defined(MKA_NO_OTEXT)
// OTEXT (spec v1.149, owner / EXT kernel text budget): the same STREAM trims on the EXT (owner) binary, plus a shared out-of-line
// HW init and no full unrolls of the owners' short fixed loops (head norm, v12 osum, wo unit, AR reduce); bit-identical;
// MKA_NO_OTEXT = the old EXT binary
#define MK_OTEXT 1
#define MK_NOUNROLL _Pragma("GCC unroll 1")
#else
#define MK_NOUNROLL
#endif
// per-group TRISC masks (bit t = TRISC t; only where de-unrolling shrinks that TRISC): head norm (H), v12 osum (O), wo unit (W), AR reduce (A)
#ifndef MKA_OT_H
#define MKA_OT_H 6   // math + pack (offline per-TRISC sweep, v1.149)
#endif
#ifndef MKA_OT_O
#define MKA_OT_O 4   // pack only: unpack +440 / math +1016 B when de-unrolled (offline per-TRISC sweep, v1.149)
#endif
#ifndef MKA_OT_W
#define MKA_OT_W 4   // pack (offline per-TRISC sweep, v1.149)
#endif
#ifndef MKA_OT_A
#define MKA_OT_A 6   // math + pack (offline per-TRISC sweep, v1.149)
#endif
#if defined(MK_OTEXT) && ((MKA_OT_H >> COMPILE_FOR_TRISC) & 1)
#define MK_OU_H _Pragma("GCC unroll 1")
#else
#define MK_OU_H
#endif
#if defined(MK_OTEXT) && ((MKA_OT_O >> COMPILE_FOR_TRISC) & 1)
#define MK_OU_O _Pragma("GCC unroll 1")
#else
#define MK_OU_O
#endif
#if defined(MK_OTEXT) && ((MKA_OT_W >> COMPILE_FOR_TRISC) & 1)
#define WOU_NOUNROLL _Pragma("GCC unroll 1")    // wo_unit_c.hpp
#endif
#if defined(MK_OTEXT) && ((MKA_OT_A >> COMPILE_FOR_TRISC) & 1)
#define MKAR_NOUNROLL _Pragma("GCC unroll 1")   // mk_ar_reduce.hpp (other includers keep the default: empty)
#endif
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
#include "../../kernels/mk_ar_reduce.hpp"   // AR reduce on the owners (MKAR_* defines from ar_phase.py)
#endif
// MKA_LEAN_C: no compute-side profiler zones (each one is compiled into all three TRISC binaries; the RWO build is
// otherwise past the 70656 B kernel-config buffer). Phase timing then comes from the dataflow zones.
#if defined(MKA_LEAN_C)
#define CZONE(n)
#else
#define CZONE(n) DeviceZoneScopedN(n)
#endif
#if defined(MK_V12)
#if defined(MK_V12_LOFI) && MK_V12_LOFI
// the engine's compute fidelity is HiFi2 (SDPA); the v12 gate|up / down matmuls on BFP4 weights run at LoFi like the
// standalone fm12 / DSMC (numerics: 0.99996 standalone); the P @ I sums are exact either way
namespace ckernel {
ALWI void v_mm_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct, rt, kt)));
    MATH((llk_math_matmul_init<MathFidelity::LoFi, MM_THROTTLE>(in0, in1, 0, ct, rt)));
    MATH((throttled_mop_status = 0));
}
ALWI void v_mm(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t rt, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
    MATH((llk_math_matmul<MathFidelity::LoFi, MM_THROTTLE>(idst, ct, rt)));
}
}  // namespace ckernel
#define V_MM_INIT(a, b, ct, rt, kt) v_mm_init(a, b, ct, rt, kt)
#define V_MM(a, b, i0, i1, d, ct, rt, kt) v_mm(a, b, i0, i1, d, ct, rt, kt)
#else
#define V_MM_INIT(a, b, ct, rt, kt) matmul_block_init(a, b, 0, ct, rt, kt)
#define V_MM(a, b, i0, i1, d, ct, rt, kt) matmul_block(a, b, i0, i1, d, 0, ct, rt, kt)
#endif
constexpr uint32_t CB_WV = 32, CB_XK = 33, CB_VA = 34, CB_VOWN = 35, CB_VRS = 36, CB_VH = 37, CB_VHO = 38, CB_VHO2 = 39;
constexpr uint32_t CB_VDO = 40, CB_VDO2 = 41, CB_VI = 42, CB_VRSA = 43, CB_VRT = 44, CB_VJ = 30;
constexpr uint32_t CB_VP = 3, CB_VXO = 4, CB_VX0 = 5;
constexpr uint32_t V_NNM = 7, V_NTM = 14, V_KTR = 20, V_OT = 10, V_NCOL = 10, V_SLOT = 420, V_DSB = 5;
// Shared split-K column RS, compute side (spec §18; MLP item = (gate, up), ipt 2; qkv item = 1 tile, ipt 1): exact sum
// of owned item m (ipt tiles) over the 8 K-rows into DST d0 .. d0 + ipt - 1 as P @ I matmuls (identity in srcA, fp32
// DST; bit-identical at LoFi / HiFi2). Own contribution: own_cb tiles own_off ..; sender s: rs_cb tiles s*V_RSO + m*ipt ..
// (DEST_TO_SRCA reuse adds truncate: -3% on an 8-way sum, never use them for this).
constexpr uint32_t V_RSO = 2;
#if defined(MKA_QSK)
#ifndef QSK_GK
#define QSK_GK 6
#endif
constexpr uint32_t QSK_SLOT = 240, QSK_PASS = 6;   // slot M pages; output tiles per DST pass
#endif
ALWI void v_rs_sum(uint32_t own_cb, uint32_t own_off, uint32_t rs_cb, uint32_t k, uint32_t m, uint32_t ipt, uint32_t d0) {
    reconfig_data_format(CB_VI, own_cb);
    matmul_block_init(own_cb, CB_VI, 0, 1, 1, 1);
    for (uint32_t t = 0; t < ipt; ++t) matmul_block(own_cb, CB_VI, own_off + t, 0, d0 + t, 0, 1, 1, 1);
    reconfig_data_format(CB_VI, rs_cb);
    matmul_block_init(rs_cb, CB_VI, 0, 1, 1, 1);
    for (uint32_t s = 0; s < 8; ++s) {
        if (s == k) continue;
        for (uint32_t t = 0; t < ipt; ++t) matmul_block(rs_cb, CB_VI, s * V_RSO + m * ipt + t, 0, d0 + t, 0, 1, 1, 1);
    }
}
#endif
#if defined(MKA_RQKV) || defined(MK_V12)
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#if defined(MKA_RQPOS) && defined(MKA_RQZFILL)
#include "api/compute/eltwise_unary/fill.h"
#endif
#endif
#if defined(MKA_RWO)
#include "wo_unit_c.hpp"
#endif

constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_WM = 6, CB_YM = 7, CB_GO = 8, CB_OUT = 16;
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12, CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28;
constexpr uint32_t CB_RS = 13, CB_QA = 14, CB_QO = 17, CB_QR = 18, CB_PQ = 19, CB_RSP = 20;
#ifndef MKA_WO_MAXT
#define MKA_WO_MAXT 5
#endif
constexpr uint32_t WO_MAXT = MKA_WO_MAXT;
#if defined(MKA_RWO)
constexpr uint32_t QO_SLOT = WO_MAXT;
#else
constexpr uint32_t QO_SLOT = 0;
#endif
constexpr uint32_t CB_A = 15, CB_S = 21, CB_AS = 22, WK = 48, PQ_QSL = (WO_MAXT * 48 > 320) ? WO_MAXT * 48 : 320, CB_XO = 23, CB_HSQ = 31, CB_HG = 30;
#if defined(MKA_KVA)   // KV append (mka2_common.hpp): source bf16 tile, BFP8 source + target tiles (full slot)
constexpr uint32_t CB_KVS = 6, CB_KVB = 8, KVA_NP = 1 + MKA_KVA_PPU * MKA_KVA_NU;
#endif
#if defined(MKA_WORING)
constexpr uint32_t CB_WSL = 31, WSL_N = WO_MAXT * WK;   // own Wo ring
#else
constexpr uint32_t CB_WSL = CB_PQ, WSL_N = PQ_QSL;
#endif
#ifndef MKA_NU
#define MKA_NU 4
#endif
constexpr uint32_t DH = 8, NO = DH + 1, QT = 8 * MKA_NU;
#ifndef MKA_AMASK
#define MKA_AMASK 0xFFFFFFFFu
#endif

#if defined(MKA_ROLE_STREAM) && defined(MKA_GDN24) && defined(MKA_QSK) && defined(MKA_RWO) && defined(MK_V12) && !defined(MKA_OWNPAR)
#define MKA_STREAMLEAN 1   // the EXTLEAN pattern on STREAM (RC config): role args every STREAM core shares are constants (program.py asserts them)
#endif
#if defined(MKA_ROLE_EXT) && defined(MKA_GDN24) && !defined(MKA_OWNPAR) && MKA_NU >= 4   // u1: one SDPA group spans all 96 cores (owners can be tree receivers)
#define MKA_EXTLEAN 1   // spec v1.38 (2): the EXT binary runs only on the 16 owners; their role args are constants (program.py asserts them)
#endif
#if defined(MKA_STREAMLEAN) || defined(MK_OTEXT)
// one out-of-line copy of the full HW (re)init per TRISC binary instead of one inlined copy per call site (STREAM / OTEXT text)
__attribute__((noinline, noclone)) static void mk_hw_startup(uint32_t a, uint32_t b, uint32_t o) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, o);
}
#define MK_HWS(a, b, o) mk_hw_startup(a, b, o)
#else
#define MK_HWS(a, b, o) compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, o)
#endif
#if defined(MK_OTEXT)
#define MK_HWS_O(a, b, o) mk_hw_startup(a, b, o)   // sites that call the inline form directly (STREAM binaries unchanged)
#else
#define MK_HWS_O(a, b, o) compute_kernel_hw_startup<SrcOrder::Reverse>(a, b, o)
#endif
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2), NDN = get_compile_time_arg_val(3);
    constexpr uint32_t GRP = get_compile_time_arg_val(4), NL = get_compile_time_arg_val(5);
    constexpr uint32_t MIXN = get_compile_time_arg_val(6), NBB = get_compile_time_arg_val(7);
    uint32_t a = 0;
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t IS_STREAM = 1; a++;   // per-role binary (spec v0.9.17): EXT-only code compiles out
#elif defined(MKA_ROLE_EXT)
    constexpr uint32_t IS_STREAM = 0; a++;   // per-role binary: STREAM-only (MLP / mixer) code compiles out
#else
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_EXTLEAN) || defined(MKA_STREAMLEAN)
    constexpr uint32_t IS_GU = 0; a++;   // STREAM under V12: the fm11 gate|up roles are off
#else
    const uint32_t IS_GU = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_RQBAL)
    uint32_t u = get_arg_val<uint32_t>(a++);   // the reader's per-pass assignment (KVP table word 3 + T)
#elif defined(MKA_RQPOS)
    const uint32_t u = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_RQPOS)
    uint32_t nblk = get_arg_val<uint32_t>(a++), kvp_m0 = 0x3FF, kvp_m1 = 0x3FF;   // bounded by kvpos at the first SDPA
    bool kvp_rd = false;
#else
    const uint32_t u = get_arg_val<uint32_t>(a++), nblk = get_arg_val<uint32_t>(a++);
#endif
#if (defined(MKA_EXTLEAN) || (defined(MKA_SV) && MKA_SV != 2 && !defined(MKA_RQBAL)))   // MKA_SV 0 / 1: tree leaf (RQBAL: per-pass trees, no fold)
    constexpr uint32_t nrecv = 0, is_root = 0; a += 2;
#elif defined(MKA_RQBAL)
    uint32_t nrecv = get_arg_val<uint32_t>(a++), is_root = get_arg_val<uint32_t>(a++);
#else
    const uint32_t nrecv = get_arg_val<uint32_t>(a++), is_root = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_RQBAL)
    uint32_t qoff = u * DH;
#else
    const uint32_t qoff = u * DH;
#endif
#if defined(MKA_RQKV)
#if defined(MKA_EXTLEAN) && defined(MKA_QSK)
    constexpr uint32_t qrole = 3, qk0 = 0, qnk = 0, qnt = 0, gdn_on = 0; a += 5;   // QR_EXT with no qkv tiles under QSK
#elif defined(MKA_STREAMLEAN)
    constexpr uint32_t qrole = 1; a++;   // every STREAM core is a QSK K-row partial producer (QR_OWNER)
    const uint32_t qk0 = get_arg_val<uint32_t>(a++);
    constexpr uint32_t qnk = QSK_GK; a++;
    const uint32_t qnt = get_arg_val<uint32_t>(a++);
    constexpr uint32_t gdn_on = 1; a++;
#else
    const uint32_t qrole = get_arg_val<uint32_t>(a++), qk0 = get_arg_val<uint32_t>(a++), qnk = get_arg_val<uint32_t>(a++);
    const uint32_t qnt = get_arg_val<uint32_t>(a++), gdn_on = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_RWO) || defined(MKA_QSK)
#if defined(MKA_EXTLEAN) && defined(MKA_QSK) && defined(MKA_RWO)
    constexpr uint32_t qn0 = 0, wo_nt = WO_MAXT; a += 2;
#elif defined(MKA_STREAMLEAN)
    const uint32_t qn0 = get_arg_val<uint32_t>(a++);
    constexpr uint32_t wo_nt = 0; a++;   // wo runs on the owners
#else
    const uint32_t qn0 = get_arg_val<uint32_t>(a++), wo_nt = get_arg_val<uint32_t>(a++);   // first qkv tile, wo tiles (0 = no wo)
#endif
#endif
#if defined(MK_GDN_INPROJ) && defined(MKA_ROLE_EXT)
    constexpr uint32_t gnt = 0, gq = 0; a += 2;
    constexpr uint32_t CB_WMG = 15, GIN_SLOT = 20 * 7;
#elif defined(MK_GDN_INPROJ)
    const uint32_t gnt = get_arg_val<uint32_t>(a++), gq = get_arg_val<uint32_t>(a++);   // column gdn_in tiles, owned (0/1)
    constexpr uint32_t CB_WMG = 15, GIN_SLOT = 20 * 7;
#endif
#if defined(MKA_RQNR)
#if defined(MKA_STREAMLEAN)
    constexpr uint32_t is_head = 0, has_hg = 0; a += 2;   // head-norm cores are owners
#if defined(MKA_KVA)
    constexpr uint32_t kva_c = 0;
#endif
#elif defined(MKA_KVA)
    const uint32_t is_head_kva = get_arg_val<uint32_t>(a++), has_hg = get_arg_val<uint32_t>(a++);
#if defined(MKA_ROLE_STREAM)
    const uint32_t is_head = is_head_kva & 0xFF;
    constexpr uint32_t kva_c = 0;
#else
    const uint32_t is_head = is_head_kva & 0xFF, kva_c = is_head_kva >> 8;   // KV append role (0 none) in bits 8+
#endif
#else
    const uint32_t is_head = get_arg_val<uint32_t>(a++), has_hg = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_RMASK)
#if defined(MKA_STREAMLEAN)
    constexpr uint32_t mask_blk = 0xFFFFFFFFu; a++;   // the mask core is never STREAM (RMASK)
#else
    const uint32_t mask_blk = get_arg_val<uint32_t>(a++);   // local block index of the user's last block, or ~0
#endif
#endif
    // q/k head: y = gamma (.) x * rsqrt(mean(x^2) + eps) over the head's 256 columns; RoPE on dims 0-63 (tiles 0, 1)
    auto head_norm_rope = [&]() {
        CZONE("C-HNR");
        cb_wait_front(CB_HG, 11);   // constants: pushed once, never popped
        cb_wait_front(CB_A, WK);   // full slot (wo reuses CB_A later in the layer)
        MK_HWS_O(CB_A, CB_A, CB_HSQ);
        reconfig_data_format(CB_A, CB_A);
        mul_init(CB_A, CB_A);
        pack_reconfig_data_format(CB_HSQ);
        tile_regs_acquire();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) mul_tiles(CB_A, CB_A, t, t, t);   // x^2
        tile_regs_commit();
        cb_reserve_back(CB_HSQ, 8);
        tile_regs_wait();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) pack_tile(t, CB_HSQ);
        tile_regs_release();
        cb_push_back(CB_HSQ, 8);
        cb_wait_front(CB_HSQ, 8);
        reconfig_data_format(CB_ONE, CB_HSQ);   // row sums: (x^2) @ ONES, accumulated over the 8 tiles in fp32 DST
        matmul_init(CB_HSQ, CB_ONE, 0);
        tile_regs_acquire();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) matmul_tiles(CB_HSQ, CB_ONE, t, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, MKA_INVHD_BITS);
        add_unary_tile(0, MKA_EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
#if defined(MKA_RQPOS)
        if (is_head == 1) mul_unary_tile(0, MKA_QSCALE_BITS);   // q heads: the softmax 1/sqrt(256), exact (power of 2)
#endif
        tile_regs_commit();
        cb_pop_front(CB_HSQ, 8);
        cb_reserve_back(CB_RS, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_RS);
        pack_tile(0, CB_RS);
        tile_regs_release();
        cb_push_back(CB_RS, 1);
        cb_wait_front(CB_RS, 1);
        // y_t = x_t * rstd (bcast cols), then * gamma_t (gamma tile rows all equal: plain eltwise)
        reconfig_data_format(CB_A, CB_RS);
        mul_bcast_cols_init(CB_A, CB_RS);
        tile_regs_acquire();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) mul_tiles_bcast_cols(CB_A, CB_RS, t, 0, t);
        tile_regs_commit();
        cb_reserve_back(CB_HSQ, 8);
        tile_regs_wait();
        pack_reconfig_data_format(CB_HSQ);
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) pack_tile(t, CB_HSQ);
        tile_regs_release();
        cb_push_back(CB_HSQ, 8);
        cb_pop_front(CB_RS, 1);
        cb_pop_front(CB_A, WK);
        cb_wait_front(CB_HSQ, 8);
        reconfig_data_format(CB_HSQ, CB_HG);
        mul_init(CB_HSQ, CB_HG);
        tile_regs_acquire();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) mul_tiles(CB_HSQ, CB_HG, t, t, t);   // y = x * rstd * gamma
        tile_regs_commit();
        cb_pop_front(CB_HSQ, 8);
        cb_reserve_back(CB_HSQ, 8);
        tile_regs_wait();
        MK_OU_H
        for (uint32_t t = 0; t < 8; ++t) pack_tile(t, CB_HSQ);
        tile_regs_release();
        cb_push_back(CB_HSQ, 8);
        cb_wait_front(CB_HSQ, 8);
        // RoPE (rotate-half within the first 64 dims): o0 = y0 c - y1 s, o1 = y1 c + y0 s; tiles 2-7 copied
        cb_reserve_back(CB_XO, 10);
        reconfig_data_format(CB_HSQ, CB_HG);
        mul_init(CB_HSQ, CB_HG);
        tile_regs_acquire();
        mul_tiles(CB_HSQ, CB_HG, 0, 8, 0);   // y0 c
        mul_tiles(CB_HSQ, CB_HG, 1, 9, 1);   // y1 s
        mul_tiles(CB_HSQ, CB_HG, 1, 8, 2);   // y1 c
        mul_tiles(CB_HSQ, CB_HG, 0, 9, 3);   // y0 s
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        add_binary_tile_init();
        add_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_XO);
        pack_tile(0, CB_XO);
        pack_tile(2, CB_XO);
        tile_regs_release();
        reconfig_data_format_srca(CB_HSQ);
        copy_init(CB_HSQ);
        tile_regs_acquire();
        MK_OU_H
        for (uint32_t t = 2; t < 8; ++t) copy_tile(CB_HSQ, t, t - 2);
        tile_regs_commit();
        tile_regs_wait();
        MK_OU_H
        for (uint32_t t = 2; t < 8; ++t) pack_tile(t - 2, CB_XO);
        tile_regs_release();
        cb_push_back(CB_XO, 10);   // full slot (8 valid)
        cb_pop_front(CB_HSQ, 8);
    };
#endif
#endif   // MKA_RQKV (v12 args follow the qkv / head args when present; V12 also runs on the MIXN stand-in)
#if defined(MK_V12)
#if defined(MKA_EXTLEAN)
    constexpr uint32_t vrole_rt = 2; a++;
    const uint32_t vnn = get_arg_val<uint32_t>(a++), vk = get_arg_val<uint32_t>(a++);
#else
    const uint32_t vrole_rt = get_arg_val<uint32_t>(a++), vnn = get_arg_val<uint32_t>(a++), vk = get_arg_val<uint32_t>(a++);
#endif
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t vrole = 1;
#elif defined(MKA_ROLE_EXT)
    const uint32_t vrole = vrole_rt == 2 ? 2u : 0u;
#else
    const uint32_t vrole = vrole_rt;
#endif
    const uint32_t vfirst = get_arg_val<uint32_t>(a++);
#if defined(MK_AR_RSTD) && defined(MKA_ROLE_STREAM) && defined(MK_AR_RSTD_HOST)
    a++;   // the producer's arg (vn) is still passed with the host Σ
#define MK_RSTD_PART() do {} while (0)
#elif defined(MK_AR_RSTD) && defined(MKA_ROLE_STREAM) && defined(MKA_SV) && MKA_SV != 1 && !defined(MK_DNORM_SPLIT)
    a++;   // MKA_SV leaf / parent: not an rstd producer (program.py asserts the variant membership)
#define MK_RSTD_PART() do {} while (0)
#elif defined(MK_AR_RSTD) && defined(MKA_ROLE_STREAM)
    // rstd producer (DESIGN v1.7x, STREAM-side): core (n = k, k) turns every K-row x delivery (CB_XK, 20 bf16 tiles) into its
    // per-row Σx² in fp32: transposed copy (X^T, exact to fp32 DST) + SFPU square + SFPU add over the 20 tiles, then SFPU
    // REDUCE_COL (row 0 = Σ_c x[r, c]^2 at column r: already the Σ tile's row format; same reduce instantiation as the consumers)
    // CB_SS -> the writer, which multicasts row 0 as row k of the Σ tile. Runs before the site's hw startup (which re-inits after it).
    const uint32_t vn_c = get_arg_val<uint32_t>(a++);
#if defined(MKA_SV) || defined(MK_DNORM_SPLIT)
    constexpr bool rprod = true;   // MKA_SV 1: every core of this variant is a producer; DNORM split: every STREAM core
#else
    const bool rprod = vn_c == vk;
#endif
    auto rstd_part = [&]() __attribute__((noinline)) {
        CZONE("C-RSS");
        cb_wait_front(CB_XK, V_KTR);
        cb_reserve_back(MK_AR_CB_SS, 1);
        reconfig_data_format_srca(CB_XK);
        pack_reconfig_data_format(MK_AR_CB_SS);
        tile_regs_acquire();
#if defined(MK_DNORM_SPLIT)
        // DNORM (ii): this core's 2 of the K-row's 20 tiles (2 vn, 2 vn + 1) -> (x^T)^2 summed + column sum (row 0 = per-token partial)
        copy_init(CB_XK, 1, 1);
        copy_tile(CB_XK, 2 * vn_c, 0);
        copy_tile(CB_XK, 2 * vn_c + 1, 1);
        square_tile_init();
        square_tile(0);
        square_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        mk_colsum(0);
#else
        copy_init(CB_XK, 1, 1);
        copy_tile(CB_XK, 0, 0);
        square_tile_init();
        square_tile(0);
        for (uint32_t t0 = 1; t0 < V_KTR; t0 += 6) {
            const uint32_t n_ = (V_KTR - t0) < 6 ? (V_KTR - t0) : 6;
            copy_init(CB_XK, 1, 1);
            for (uint32_t i = 0; i < n_; ++i) copy_tile(CB_XK, t0 + i, 1 + i);
            square_tile_init();
            for (uint32_t i = 0; i < n_; ++i) square_tile(1 + i);
            add_binary_tile_init();
            for (uint32_t i = 0; i < n_; ++i) add_binary_tile(0, 1 + i, 0);
        }
#if !defined(MK_AR_RSTD_RED_GDN)
        mk_colsum(0);
#endif   // RED_GDN: the elementwise Σ_t (x_t^T)^2 tile goes to the GDN host core, which sums the 8 K-rows + column-reduces
#endif   // MK_DNORM_SPLIT
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, MK_AR_CB_SS);
        tile_regs_release();
        cb_push_back(MK_AR_CB_SS, 1);
    };
#define MK_RSTD_PART() do { if (rprod) rstd_part(); } while (0)
#else
#define MK_RSTD_PART() do {} while (0)
#endif
#if defined(MK_GAMMA_CONS) && defined(MKA_ROLE_STREAM)
    // γ at the consumers: CB_XK *= γ (bcast rows of the writer's 20 γ row tiles, CB MK_GC_CB) IN PLACE: packed through MK_GC_XKA,
    // a second format index of CB_XK's memory whose fifo stays at the base like CB_XK's (20 pages each, one delivery at a time)
    auto gc_scale = [&]() __attribute__((noinline)) {
        CZONE("C-GC");
        cb_wait_front(CB_XK, V_KTR);
        cb_wait_front(MK_GC_CB, V_KTR);
        cb_reserve_back(MK_GC_XKA, V_KTR);
        reconfig_data_format(CB_XK, MK_GC_CB);
        pack_reconfig_data_format(MK_GC_XKA);
        mul_bcast_rows_init(CB_XK, MK_GC_CB);
        for (uint32_t t0 = 0; t0 < V_KTR; t0 += 4) {
            tile_regs_acquire();
            MK_NOUNROLL
            for (uint32_t i = 0; i < 4; ++i) mul_tiles_bcast_rows(CB_XK, MK_GC_CB, t0 + i, t0 + i, i);
            tile_regs_commit();
            tile_regs_wait();
            MK_NOUNROLL
            for (uint32_t i = 0; i < 4; ++i) pack_tile<true>(i, MK_GC_XKA, t0 + i);
            tile_regs_release();
        }
        cb_push_back(MK_GC_XKA, V_KTR);
        cb_wait_front(MK_GC_XKA, V_KTR);   // the packed x is in L1 before the consumers unpack CB_XK
        cb_pop_front(MK_GC_XKA, V_KTR);
        cb_pop_front(MK_GC_CB, V_KTR);
    };
#define MK_GC_SCALE() gc_scale()
#else
#define MK_GC_SCALE() do {} while (0)
#endif
    // rstd' from the CB 20 stand-in (same producer as the qkv PROJ): rsqrt(Σ partials / H + eps), col 0 -> CB_VRT
    auto v_rstd = [&]() __attribute__((noinline)) {   // 2 call sites (mixer qkv / gdn_in, MLP): one copy (STREAM text)
#if defined(MK_AR_RSTD) && defined(MK_AR_RSTD_RED_GDN)
        // v1.112 (iii'): the FINAL Σ (column 0 = per-token Σx², reduced on a GDN-role core) -> exact fp32 copy (UnpackToDestFp32 CB 43)
        cb_wait_front(CB_VRSA, 1);
        cb_reserve_back(CB_VRT, 1);
        reconfig_data_format_srca(CB_VRSA);
        copy_init(CB_VRSA);
        tile_regs_acquire();
        copy_tile(CB_VRSA, 0, 0);
#elif defined(MK_AR_RSTD)
        // real rstd: the Σ tile (rsp tile 0, bound as CB_VRSA) holds row k = K-row k's per-token partial (row form, fp32).
        // UnpackToDestFp32 copy (CB 43) -> SFPU column sum (row 0 = Σ_k S[k][c]) -> in-DST transpose (column 0 per token row).
        cb_wait_front(CB_VRSA, 1);
        cb_reserve_back(CB_VRT, 1);
        reconfig_data_format_srca(CB_VRSA);
        copy_init(CB_VRSA);
        tile_regs_acquire();
        copy_tile(CB_VRSA, 0, 0);
        mk_colsum(0);
        transpose_dest_init<true>();
        transpose_dest<true>(0);
#else
        cb_wait_front(CB_VRSA, MKA_NOWN);
        cb_reserve_back(CB_VRT, 1);
        tile_regs_acquire();
#endif
#if defined(MK_AR_RSTD)
#elif MKA_NOWN == 2
        reconfig_data_format(CB_VRSA, CB_VRSA);
        add_init(CB_VRSA, CB_VRSA);
        add_tiles(CB_VRSA, CB_VRSA, 0, 1, 0);
#else
        reconfig_data_format_srca(CB_VRSA);
        copy_init(CB_VRSA);
        copy_tile(CB_VRSA, 0, 0);
#endif
        binop_with_scalar_tile_init();
        mul_unary_tile(0, MKA_INVH_BITS);
        add_unary_tile(0, MKA_EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_VRT);
        pack_tile(0, CB_VRT);
        tile_regs_release();
        cb_push_back(CB_VRT, 1);
#if defined(MK_AR_RSTD)
        cb_pop_front(CB_VRSA, 1);
#else
        cb_pop_front(CB_VRSA, MKA_NOWN);
#endif
    };
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
    mkar::Reduce ar_red;   // [has_work, ring, PT] from the owner's L1 arg table (spec v1.25): has_work = this device owns it
    ar_red.has_work = 0;
    if (vrole == 2) ar_red.init_l1(MK_AR_BLK, MK_AR_BLK_COMP);
#endif
    // owner: Σ of the 10 column down partials (P @ I, fp32 DST) [+ the x chunk while the AR is a stall] -> CB_VXO
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
    // one AR on the owner TRISCs: the reduce (this device's chunk; own partial read in place from own_cb at R = 32) and,
    // with MK_AR_GAMMA, x_deliver = out ⊙ γ into CB_VXO. MLP AR: own_cb = CB_VXO; attention AR: the wo output CB_XO.
    auto ar_step = [&](uint32_t own_cb) __attribute__((noinline)) {   // one copy for both call sites (MLP AR, attention AR): EXT text
        if (ar_red.has_work) {
            CZONE("C-AR");
#if defined(MK_AR_HWSTART)
            compute_kernel_hw_startup(MKAR_CB_RED, MKAR_CB_RED, MKAR_CB_OUT);
#endif
            ar_red.iter(own_cb);   // sets its own unpack / pack formats and add init (v12_osum's hw startup ran this layer)
        }
#if defined(MK_AR_GAMMA)
        {   // spec §19: x_deliver = out ⊙ γ_next on every owner (owning + gather role); out stays un-scaled (residual)
            CZONE("C-ARG");
            cb_wait_front(MK_AR_CBG, MK_AR_GT);   // the writer pushes it once γ landed and the AR output is complete
            cb_reserve_back(CB_VXO, V_OT);        // free once the writer popped this layer's partial (after the AR)
            reconfig_data_format(MKAR_CB_OUT, MK_AR_CBG);
            pack_reconfig_data_format(CB_VXO);
#if MK_AR_GROWS == 0
            mul_bcast_rows_init(MKAR_CB_OUT, MK_AR_CBG);
#else
            mul_init(MKAR_CB_OUT, MK_AR_CBG);   // packed pseudo-tiles, γ pre-replicated: element-wise
#endif
            for (uint32_t t = 0; t < MK_AR_GT; ++t) {
                tile_regs_acquire();
#if MK_AR_GROWS == 0
                mul_tiles_bcast_rows(MKAR_CB_OUT, MK_AR_CBG, t, t, 0);
#else
                mul_tiles(MKAR_CB_OUT, MK_AR_CBG, t, t, 0);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, CB_VXO, t);
                tile_regs_release();
            }
            cb_push_back(CB_VXO, V_OT);
            cb_pop_front(MK_AR_CBG, MK_AR_GT);
        }
#endif
#if defined(MK_AR_RSTD_OWN)
        {   // rstd producer (owner-side variant, over budget): per-row Σx² of this owner's residual chunk (v_x0 = CB_VX0, 10 bf16 tiles) as the diagonal of
            // X X^T (matmul, in1 transposed, fp32 DST accumulate over the 10 tiles) -> CB_SS; the writer extracts the diagonal
            CZONE("C-RSS");
            cb_wait_front(MK_AR_CB_RSYN, 1);      // the writer: this chunk of the residual is final (AR done)
            cb_wait_front(CB_VX0, V_OT);          // pushed once at start, never popped (MK_OWNER_RESID=0)
            cb_reserve_back(MK_AR_CB_SS, 1);
            reconfig_data_format(CB_VX0, CB_VX0);
            pack_reconfig_data_format(MK_AR_CB_SS);
            matmul_block_init(CB_VX0, CB_VX0, 1, 1, 1, 1);
            tile_regs_acquire();
            for (uint32_t t = 0; t < V_OT; ++t) matmul_block(CB_VX0, CB_VX0, t, t, 0, 1, 1, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, MK_AR_CB_SS);
            tile_regs_release();
            cb_push_back(MK_AR_CB_SS, 1);
            cb_pop_front(MK_AR_CB_RSYN, 1);
        }
#endif
    };
#endif
    auto v12_osum = [&](uint32_t L) {
        if (vrole != 2) return;
        if (L == 0) cb_wait_front(CB_VI, 1);
        MK_HWS_O(CB_VP, CB_VI, CB_VXO);
        cb_wait_front(CB_VP, V_NCOL * V_OT);
#if defined(MK_OWNER_RESID) && MK_OWNER_RESID
        if (L == 0) cb_wait_front(CB_VX0, V_OT);
#endif
        cb_reserve_back(CB_VXO, V_OT);
        CZONE("C-OSUM");
        pack_reconfig_data_format(CB_VXO);
        MK_OU_O
        for (uint32_t b = 0; b < 2; ++b) {
            tile_regs_acquire();
            reconfig_data_format(CB_VI, CB_VP);
            matmul_block_init(CB_VP, CB_VI, 0, 1, 1, 1);
            MK_OU_O
            for (uint32_t n = 0; n < V_NCOL; ++n)
                MK_OU_O
                for (uint32_t t = 0; t < 5; ++t) matmul_block(CB_VP, CB_VI, n * V_OT + b * 5 + t, 0, t, 0, 1, 1, 1);
#if defined(MK_OWNER_RESID) && MK_OWNER_RESID
            reconfig_data_format(CB_VI, CB_VX0);
            matmul_block_init(CB_VX0, CB_VI, 0, 1, 1, 1);
            MK_OU_O
            for (uint32_t t = 0; t < 5; ++t) matmul_block(CB_VX0, CB_VI, b * 5 + t, 0, t, 0, 1, 1, 1);
#endif
            tile_regs_commit();
            tile_regs_wait();
            MK_OU_O
            for (uint32_t t = 0; t < 5; ++t) pack_tile<true>(t, CB_VXO, b * 5 + t);
            tile_regs_release();
        }
        cb_push_back(CB_VXO, V_OT);
        cb_pop_front(CB_VP, V_NCOL * V_OT);
        // AR reduce (real AR, spec v0.9.2) follows here on the owner TRISCs
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
        ar_step(CB_VXO);
#endif
    };
#define V12_OSUM(L) v12_osum(L)
#else
#define V12_OSUM(L) do {} while (0)
#endif
#if defined(MKA_RQKV)   // reopened after the v12 block
#if defined(MKA_QSK)
    // spec §18 split-K qkv on STREAM core (column c, K-row vk): partial = x[K-row] @ slot M over this column's N-list
    // (qk0 = nt tiles, passes of <= 6 DST tiles), item-major into CB_VA (+ own items vk, vk + 8 into CB_VOWN). Slot M is
    // popped BEFORE CB_VA is pushed (the writer refills it once the RS starts). Attention layers then sum the owned items
    // over the 8 K-rows (P @ I, exact), scale by rstd (Rcol @ J), sigmoid the gate tiles and pack CB_QO (qnt = 1-2 tiles,
    // first tile qn0, stride 8). GDN-type layers: QSK_GK K tiles, result discarded (in-proj stand-in).
    auto qsk_mix = [&](bool att_layer) {
        CZONE("C-QSK");
#if defined(MK_GDN_INPROJ)   // GDN-type layer = the real in-proj: 20 K tiles x gnt from the BFP4 slot, RS, x rstd
        const bool mrs = true;
        const uint32_t nt = att_layer ? qk0 : gnt, nk = V_KTR, cbw = att_layer ? CB_WM : CB_WMG, cbs = att_layer ? QSK_SLOT : GIN_SLOT;
        const uint32_t nown = att_layer ? qnt : gq;
#else
        const bool mrs = att_layer;
        const uint32_t nt = qk0, nk = att_layer ? V_KTR : QSK_GK, cbw = CB_WM, cbs = QSK_SLOT, nown = qnt;
#endif
        MK_RSTD_PART();
        MK_GC_SCALE();
        MK_HWS(CB_XK, cbw, CB_VA);
        cb_wait_front(CB_XK, V_KTR);
        cb_wait_front(cbw, cbs);
        if (mrs) cb_reserve_back(CB_VA, V_NTM);
        if (mrs && nown) cb_reserve_back(CB_VOWN, 2);
        reconfig_data_format(cbw, CB_XK);
        pack_reconfig_data_format(CB_VA);
        for (uint32_t j0 = 0; j0 < nt; j0 += QSK_PASS) {
            const uint32_t nw = (nt - j0) < QSK_PASS ? (nt - j0) : QSK_PASS;
            matmul_block_init(CB_XK, cbw, 0, nw, 1, V_KTR);
            tile_regs_acquire();
            for (uint32_t kl = 0; kl < nk; ++kl) matmul_block(CB_XK, cbw, kl, kl * nt + j0, 0, 0, nw, 1, V_KTR);
            tile_regs_commit();
            tile_regs_wait();
            if (mrs)
                for (uint32_t jj = 0; jj < nw; ++jj) {
                    const uint32_t j = j0 + jj;
                    pack_tile<true>(jj, CB_VA, j);
                    if (j == vk || j == vk + 8) pack_tile<true>(jj, CB_VOWN, j >= 8 ? 1 : 0);
                }
            tile_regs_release();
        }
        cb_pop_front(cbw, cbs);
        cb_pop_front(CB_XK, V_KTR);
        if (!mrs) return;
        cb_push_back(CB_VA, V_NTM);
        if (!nown) return;   // (GDN: K-rows 6-7 of a 6-tile column own no in-proj tile)
        cb_push_back(CB_VOWN, 2);
        v_rstd();
        cb_wait_front(CB_VRT, 1);
        cb_wait_front(CB_VOWN, 2);
        cb_wait_front(CB_VRS, 16);
        const uint32_t qos = QO_SLOT ? QO_SLOT : qnt;   // same count on both layer types: the CB_QO fifo stays slot-aligned
        cb_reserve_back(CB_QO, qos);
        tile_regs_acquire();
        for (uint32_t m = 0; m < nown; ++m) v_rs_sum(CB_VOWN, m, CB_VRS, vk, m, 1, m);
        reconfig_data_format(CB_VJ, CB_VRT);
        matmul_block_init(CB_VRT, CB_VJ, 0, 1, 1, 1);
        matmul_block(CB_VRT, CB_VJ, 0, 0, 2, 0, 1, 1, 1);
        mul_binary_tile_init();
        for (uint32_t m = 0; m < nown; ++m) mul_binary_tile(m, 2, m);
        for (uint32_t m = 0; m < nown; ++m) {   // gate tiles: sig = 1 / (1 + exp(-x))
            if (!att_layer || qn0 + 8 * m < 64) continue;
            binop_with_scalar_tile_init();
            mul_unary_tile(m, 0xBF800000u);
            exp_tile_init();
            exp_tile(m);
            binop_with_scalar_tile_init();
            add_unary_tile(m, 0x3F800000u);
            recip_tile_init();
            recip_tile(m);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_QO);
        for (uint32_t m = 0; m < nown; ++m) pack_tile(m, CB_QO);
        tile_regs_release();
        cb_push_back(CB_QO, qos);
        cb_pop_front(CB_VRS, 16);
        cb_pop_front(CB_VOWN, 2);
        cb_pop_front(CB_VRT, 1);
    };
#else
    // real qkv PROJ (spec v0.4): y = rstd ⊙ (x @ diag(γ)W); rstd = rsqrt(ΣΣx²/H + eps) from the AR owners' fp32 partials
    auto qkv_proj = [&](bool att_layer) {
        CZONE("C-QKV");
        const uint32_t cbw = qrole == 3 ? CB_PQ : CB_WM;
        MK_HWS(CB_X, cbw, CB_QA);
        reconfig_data_format(cbw, CB_X);
        matmul_block_init(CB_X, cbw, 0, qnt, 1, KB);
        cb_wait_front(cbw, qnk * qnt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < qnk; ++k) matmul_block(CB_X, cbw, qk0 + k, k * qnt, 0, 0, qnt, 1, KB);
        tile_regs_commit();
        if (!att_layer) {   // GDN in-proj stand-in: discard the result WITHOUT touching CB_QA (the partner's writer
            tile_regs_wait();   // reads CB_QA through its own read pointer, which must stay in step with the packer)
            tile_regs_release();
            cb_pop_front(cbw, qnk * qnt);
            return;
        }
        cb_reserve_back(CB_QA, qnt);
        tile_regs_wait();
        pack_reconfig_data_format(CB_QA);
        for (uint32_t j = 0; j < qnt; ++j) pack_tile(j, CB_QA);
        tile_regs_release();
        cb_push_back(CB_QA, qnt);
        cb_pop_front(cbw, qnk * qnt);
        if (qrole == 2) return;   // partner: the writer ships the fp32 partial to the owner
        if (qrole == 1) {   // owner: + partner's K-half partial
            cb_wait_front(CB_QA, 1);
            cb_wait_front(CB_QR, 1);
            tile_regs_acquire();
#if defined(MKA_QADD_FPU)
            reconfig_data_format(CB_QA, CB_QR);   // FPU add: both fp32 partials pass the reduced-precision src registers
            add_init(CB_QA, CB_QR);
            add_tiles(CB_QA, CB_QR, 0, 0, 0);
#else
            // exact fp32: both partials unpack straight to the fp32 DST (copy_tile), SFPU fp32 add
            reconfig_data_format_srca(CB_QA);
            copy_init(CB_QA);
            copy_tile(CB_QA, 0, 0);
            reconfig_data_format_srca(CB_QR);
            copy_init(CB_QR);
            copy_tile(CB_QR, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
#endif
            tile_regs_commit();
            cb_pop_front(CB_QA, 1);
            cb_pop_front(CB_QR, 1);
            cb_reserve_back(CB_QA, 1);
            tile_regs_wait();
            pack_reconfig_data_format(CB_QA);
            pack_tile(0, CB_QA);
            tile_regs_release();
            cb_push_back(CB_QA, 1);
        }
        // rstd (column 0 of each partial tile holds that owner's per-row Σx²)
        cb_wait_front(CB_RSP, MKA_NOWN);
        tile_regs_acquire();
#if MKA_NOWN == 2
        reconfig_data_format(CB_RSP, CB_RSP);
        add_init(CB_RSP, CB_RSP);
        add_tiles(CB_RSP, CB_RSP, 0, 1, 0);
#else
        static_assert(MKA_NOWN == 1, "bench supports 1 or 2 AR-owner partials");
        reconfig_data_format_srca(CB_RSP);
        copy_init(CB_RSP);
        copy_tile(CB_RSP, 0, 0);
#endif
        binop_with_scalar_tile_init();
        mul_unary_tile(0, MKA_INVH_BITS);
        add_unary_tile(0, MKA_EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        cb_reserve_back(CB_RS, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_RS);
        pack_tile(0, CB_RS);
        tile_regs_release();
        cb_push_back(CB_RS, 1);
        cb_pop_front(CB_RSP, MKA_NOWN);
        // scale the output rows at pack
        cb_wait_front(CB_RS, 1);
        cb_wait_front(CB_QA, qnt);
        reconfig_data_format(CB_QA, CB_RS);
        mul_bcast_cols_init(CB_QA, CB_RS);
        tile_regs_acquire();
        for (uint32_t j = 0; j < qnt; ++j) mul_tiles_bcast_cols(CB_QA, CB_RS, j, 0, j);
#if defined(MKA_RWO)
        if (qn0 + qnt > 64) {   // gate tiles (64-111): only sig(gate) is ever used (o (.) sigmoid(gate)).
            // sigmoid = 1 / (1 + exp(-x)) from SFPU ops the kernel already links (exp, recip, scalar binops): a separate
            // sigmoid_tile pushed the program past the 70656 B kernel-config buffer
            for (uint32_t j = 0; j < qnt; ++j) {
                if (qn0 + j < 64) continue;
                binop_with_scalar_tile_init();
                mul_unary_tile(j, 0xBF800000u);   // * -1
                exp_tile_init();
                exp_tile(j);
                binop_with_scalar_tile_init();
                add_unary_tile(j, 0x3F800000u);   // + 1
                recip_tile_init();
                recip_tile(j);
            }
        }
#endif
        tile_regs_commit();
        cb_reserve_back(CB_QO, QO_SLOT ? QO_SLOT : qnt);
        tile_regs_wait();
        pack_reconfig_data_format(CB_QO);
        for (uint32_t j = 0; j < qnt; ++j) pack_tile(j, CB_QO);
        tile_regs_release();
        cb_push_back(CB_QO, QO_SLOT ? QO_SLOT : qnt);
        cb_pop_front(CB_QA, qnt);
        cb_pop_front(CB_RS, 1);
    };
#endif  // MKA_QSK
#endif
#ifndef V12_OSUM
#define V12_OSUM(L) do {} while (0)
#endif
    if (IS_STREAM) MK_HWS(CB_X, CB_W, CB_HL);
    else MK_HWS(CB_Q, CB_K, CB_P);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t L = 0; L < NL; ++L) {
#if defined(MKA_LPER)
        const bool att = ((uint32_t)(MKA_AMASK) >> (L & 31u)) & 1u;
#else
        const bool att = ((uint32_t)(MKA_AMASK) >> L) & 1u;
#endif
#if defined(MK_REAL_GAR) && defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
        if (!IS_STREAM && !att) { if (vrole == 2) ar_step(MK_AR_CB_AIN); V12_OSUM(L); continue; }   // G-d: GDN mixer AR, then MLP
#else
        if (!IS_STREAM && !att) { V12_OSUM(L); continue; }
#endif
#if defined(MKA_RQKV)
        // attention layers: the real qkv PROJ; GDN-type layers: the STREAM K-half matmul alone stands in for the GDN
        // in-proj (64 x 87 KB BFP8 = 5.6 MB, spec 5.9 MB), output discarded
#if defined(MKA_QSK)
        if (IS_STREAM) qsk_mix(att);
#if defined(MKA_RQNR)
        else if (is_head) head_norm_rope();   // ATTN-only cores reach here in attention layers only
#endif
#else
        if (IS_STREAM) {
            cb_wait_front(CB_X, KT);
            if (qrole && (att || gdn_on)) qkv_proj(att);
        } else if (qrole == 3) {
            cb_wait_front(CB_X, KT);
            qkv_proj(true);
            cb_pop_front(CB_X, KT);
#if defined(MKA_RQNR)
            if (att && is_head) head_norm_rope();
#endif
        }
#endif  // MKA_QSK
#if defined(MKA_KVA)
        if (kva_c) {   // KV append: the bf16 K / V tile -> BFP8 (copy + pack, as paged_update_cache's tilize pack)
            cb_wait_front(CB_KVS, 1);
            cb_reserve_back(CB_KVB, KVA_NP);   // full slot (page 0 = packed source; the writer owns pages 1..)
            reconfig_data_format_srca(CB_KVS);
            pack_reconfig_data_format(CB_KVB);
            copy_init(CB_KVS);
            tile_regs_acquire();
            copy_tile(CB_KVS, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_KVB);
            tile_regs_release();
            cb_push_back(CB_KVB, KVA_NP);
            cb_pop_front(CB_KVS, 1);
        }
#endif
        if (false) {
#else
        if (IS_STREAM) {
#endif
            cb_wait_front(CB_X, KT);
            CZONE("C-MIX");
#if !defined(MKA_NO_REINIT)
            if (L > 0) MK_HWS(CB_X, CB_W, CB_HL);
#endif
            reconfig_data_format(CB_WM, CB_X);
            matmul_block_init(CB_X, CB_WM, 0, MIXN, 1, KB);
            cb_wait_front(CB_WM, KT * MIXN);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) matmul_block(CB_X, CB_WM, k, k * MIXN, 0, 0, MIXN, 1, KB);
            tile_regs_commit();
            cb_reserve_back(CB_YM, MIXN);
            tile_regs_wait();
            pack_reconfig_data_format(CB_YM);
            for (uint32_t j = 0; j < MIXN; ++j) pack_tile(j, CB_YM);
            tile_regs_release();
            cb_push_back(CB_YM, MIXN);
            cb_pop_front(CB_WM, KT * MIXN);
        }
        if (att) {   // ---- SDPA phase (type 5) ----
            CZONE("C-SDPA");
#if !defined(MKA_NO_REINIT)
            // the previous layer's tree adds (fp32 copy_tile = unpack-to-dest) and the root's eltwise normalize leave
            // unpacker / math state that a short matmul_init does not undo: re-run the full HW init per SDPA phase
#if defined(MKA_RQKV)
#if defined(MKA_KVA)
            if (L > 0 || qrole || kva_c) MK_HWS(CB_Q, CB_K, CB_P);   // after the qkv bcast-mul / KV pack
#else
            if (L > 0 || qrole) MK_HWS(CB_Q, CB_K, CB_P);   // after the qkv bcast-mul
#endif
#else
#if defined(MKA_LPER)   // NL64: any attention layer before L
            if (L > 0 && (IS_STREAM || (L >= 32u ? (uint32_t)(MKA_AMASK) : ((uint32_t)(MKA_AMASK) & ((1u << L) - 1u))))) MK_HWS(CB_Q, CB_K, CB_P);
#else
            if (L > 0 && (IS_STREAM || ((uint32_t)(MKA_AMASK) & ((1u << L) - 1u)))) MK_HWS(CB_Q, CB_K, CB_P);
#endif
#endif
#endif
            cb_wait_front(CB_Q, QT);
#if defined(MKA_RQPOS)
            if (!kvp_rd) {   // the reader's per-pass word (mka2_common.hpp), read by all three TRISCs
                volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKA_KVP_INFO);
                uint32_t v;
                do { asm volatile("fence" ::: "memory"); v = *w; } while (!(v >> 31));
                nblk = v & 0x7FF; kvp_m0 = (v >> 11) & 0x3FF; kvp_m1 = (v >> 21) & 0x3FF;
#if defined(MKA_RQBAL)
                {
                    const uint32_t ws = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKA_KVP_ASGN);
                    u = ws & 0xFF; qoff = u * DH;
#if !defined(MKA_EXTLEAN)
                    nrecv = (ws >> 8) & 0xFF; is_root = (ws >> 16) & 1;
#endif
                }
#endif
                kvp_rd = true;
            }
#endif
            cb_reserve_back(CB_O, NO);
            bool first = true;
            for (uint32_t b = 0; b < nblk; b += NBB) {
                const uint32_t nb = (nblk - b) < NBB ? (nblk - b) : NBB;
                cb_wait_front(CB_K, NBB * DH);   // full groups (a short last group is padded; only nb blocks are used)
                cb_wait_front(CB_V, NBB * DH);
                reconfig_data_format(CB_K, CB_Q);
                matmul_init(CB_Q, CB_K, 1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < nb; ++i)
                    for (uint32_t t = 0; t < DH; ++t) matmul_tiles(CB_Q, CB_K, qoff + t, i * DH + t, i);
#if defined(MKA_SCLAMP)
                unary_min_tile_init();   // RC guard for the no-max softmax: S <= 60 (exp(60) ~ 1e26, fp32-safe sums); bit-identical
                for (uint32_t i = 0; i < nb; ++i) unary_min_tile(i, 0x42700000u);   // when |S| < 60 (real 8-32K traces: max 21.85)
#endif
#if defined(MKA_RQPOS)
                if (b == kvp_m0 || b == kvp_m1) {   // causal mask from kvpos, ADDITIVE on S (0 / -9984) before exp (NBB = 1);
                    cb_wait_front(CB_ONE, 3);         // the writer's tiles: CB_ONE 1 = block m0, 2 = block m1 (once per pass)
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_ONE);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_ONE, b == kvp_m0 ? 1 : 2, 0);
                }
#endif
                exp_tile_init();
                for (uint32_t i = 0; i < nb; ++i) exp_tile(i);
#if defined(MKA_RQPOS)
#elif defined(MKA_RQNR) && defined(MKA_RMASK)
                if (!IS_STREAM && mask_blk >= b && mask_blk < b + nb) {   // causal mask on the last block: P = exp(S) (.) M
                    cb_wait_front(CB_HG, 11);
                    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_HG);
                    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_HG, 10, mask_blk - b);
                }
#endif
                tile_regs_commit();
                cb_reserve_back(CB_P, NBB);
                tile_regs_wait();
                pack_reconfig_data_format(CB_P);
                for (uint32_t i = 0; i < nb; ++i) pack_tile(i, CB_P);
                tile_regs_release();
                cb_push_back(CB_P, NBB);
                cb_wait_front(CB_P, NBB);
                reconfig_data_format(CB_V, CB_P);
                matmul_init(CB_P, CB_V, 0);
                tile_regs_acquire();
                for (uint32_t i = 0; i < nb; ++i) {
                    MK_NOUNROLL
                    for (uint32_t t = 0; t < DH; ++t) matmul_tiles(CB_P, CB_V, i, i * DH + t, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_O);
                pack_reconfig_l1_acc(first ? 0 : 1);
                MK_NOUNROLL
                for (uint32_t t = 0; t < DH; ++t) pack_tile<true>(t, CB_O, t);
                pack_reconfig_l1_acc(0);
                tile_regs_release();
                reconfig_data_format(CB_ONE, CB_P);
                matmul_init(CB_P, CB_ONE, 0);
                tile_regs_acquire();
                for (uint32_t i = 0; i < nb; ++i) matmul_tiles(CB_P, CB_ONE, i, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(first ? 0 : 1);
                pack_tile<true>(0, CB_O, DH);
                pack_reconfig_l1_acc(0);
                tile_regs_release();
                first = false;
                cb_pop_front(CB_P, NBB);
                cb_pop_front(CB_K, NBB * DH);
                cb_pop_front(CB_V, NBB * DH);
            }
            cb_pop_front(CB_Q, QT);
#if defined(MKA_RQPOS)
            if (first) {   // no valid KV block on this core (short user): zero partial (O, l) for the tree
                tile_regs_acquire();   // DST half is zero: the packer clears it on release (ZEROACC in pack_dest_section_done)
#if defined(MKA_RQZFILL)
                fill_tile_init();
                for (uint32_t t = 0; t <= DH; ++t) fill_tile(t, 0.0f);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_O);
                MK_NOUNROLL
                for (uint32_t t = 0; t <= DH; ++t) pack_tile<true>(t, CB_O, t);
                tile_regs_release();
            }
#endif
            for (uint32_t i = 0; i < nrecv; ++i) {
                cb_wait_front(CB_R, NO);
                reconfig_data_format_srca(CB_R);
                copy_init(CB_R);
                for (uint32_t t0 = 0; t0 < NO; t0 += DH) {
                    const uint32_t n = (NO - t0) < DH ? (NO - t0) : DH;
                    tile_regs_acquire();
                    MK_NOUNROLL
                    for (uint32_t t = 0; t < n; ++t) copy_tile(CB_R, t0 + t, t);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(CB_O);
                    pack_reconfig_l1_acc(1);
                    MK_NOUNROLL
                    for (uint32_t t = 0; t < n; ++t) pack_tile<true>(t, CB_O, t0 + t);
                    pack_reconfig_l1_acc(0);
                    tile_regs_release();
                }
                cb_pop_front(CB_R, NO);
            }
            cb_push_back(CB_O, NO);
            if (is_root) {
                cb_wait_front(CB_O, NO);
                reconfig_data_format_srca(CB_O);
                copy_init(CB_O);
                tile_regs_acquire();
                copy_tile(CB_O, DH, 0);
                recip_tile_init();
                recip_tile(0);
                tile_regs_commit();
                cb_reserve_back(CB_D, 1);
                tile_regs_wait();
                pack_reconfig_data_format(CB_D);
                pack_tile(0, CB_D);
                tile_regs_release();
                cb_push_back(CB_D, 1);
                cb_wait_front(CB_D, 1);
                reconfig_data_format(CB_O, CB_D);
                mul_init(CB_O, CB_D);
                cb_reserve_back(CB_OA, DH);
                pack_reconfig_data_format(CB_OA);
                MK_NOUNROLL
                for (uint32_t t = 0; t < DH; ++t) {
                    tile_regs_acquire();
#if defined(MKA_DBGRAW)
                    // debug: raw O_aug tiles 0..6 and the denominator tile in slot 7 (no normalize)
                    reconfig_data_format_srca(CB_O); copy_init(CB_O); copy_tile(CB_O, t == DH - 1 ? DH : t, 0);
#else
                    mul_tiles(CB_O, CB_D, t, 0, 0);
#endif
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, CB_OA);
                    tile_regs_release();
                }
                cb_push_back(CB_OA, DH);
                cb_pop_front(CB_D, 1);
                cb_pop_front(CB_O, NO);
            }
        }
#if defined(MKA_RWO)
        if (!IS_STREAM && att && wo_nt) {   // real wo, split-N: A_g = A (.) sig(gate) in place over S, then A_g @ Wo slice (ext ring)
            CZONE("C-WO");
            MK_HWS_O(CB_A, CB_S, CB_AS);
            wou::gate_mul(CB_A, CB_S, CB_AS, WK);                              // A_g = A (.) sig(gate), in place over S
            wou::matmul_nt(CB_AS, CB_WSL, WSL_N, CB_QO, QO_SLOT, WK, wo_nt);   // A_g @ Wo slice (ring) -> CB_QO
        }
#endif
#if defined(MK_REAL_AAR) && defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
        if (!IS_STREAM) { if (att && vrole == 2) ar_step(MK_AR_CB_AIN); V12_OSUM(L); continue; }   // attention AR, then MLP
#else
        if (!IS_STREAM) { V12_OSUM(L); continue; }
#endif
#if defined(MKA_NOMLP)
        cb_wait_front(CB_GO, 1); cb_pop_front(CB_GO, 1); cb_pop_front(CB_X, KT); continue;
#endif
#if defined(MK_V12)
        cb_wait_front(CB_GO, 1);
        cb_pop_front(CB_GO, 1);
        if (vrole == 1) {
            const bool own_h = vk < vnn;
            if (L == 0) { cb_wait_front(CB_VI, 1); cb_wait_front(CB_VJ, 1); }
            MK_RSTD_PART();
        MK_GC_SCALE();
            MK_HWS(CB_XK, CB_WV, CB_VA);
            cb_wait_front(CB_WV, V_SLOT);
            cb_wait_front(CB_XK, V_KTR);
            cb_reserve_back(CB_VA, V_NTM);
            if (own_h) cb_reserve_back(CB_VOWN, 2);
            {
                CZONE("C-VGU");
                reconfig_data_format(CB_WV, CB_XK);
                pack_reconfig_data_format(CB_VA);
                for (uint32_t half = 0; half < 2; ++half) {
                    V_MM_INIT(CB_XK, CB_WV, vnn, 1, V_KTR);
                    tile_regs_acquire();
                    for (uint32_t kk = 0; kk < V_KTR; ++kk) V_MM(CB_XK, CB_WV, kk, half * V_KTR * vnn + kk * vnn, 0, vnn, 1, V_KTR);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < vnn; ++j) pack_tile<true>(j, CB_VA, 2 * j + half);   // item-major: (gate j, up j)
                    if (own_h) pack_tile<true>(vk, CB_VOWN, half);
                    tile_regs_release();
                }
            }
            cb_push_back(CB_VA, V_NTM);
            if (own_h) cb_push_back(CB_VOWN, 2);
            cb_pop_front(CB_XK, V_KTR);
            if (own_h) {
                v_rstd();
                // h_k = silu(rstd ⊙ Σ_r gate_k[r]) * (rstd ⊙ Σ_r up_k[r]); sums as P @ I (exact, fp32 DST); rstd
                // broadcast along the row as Rcol @ J (J: row 0 = 1)
                cb_wait_front(CB_VRT, 1);
                cb_wait_front(CB_VOWN, 2);
                cb_wait_front(CB_VRS, 16);
                CZONE("C-VRS");
                cb_reserve_back(CB_VHO, 1);
                cb_reserve_back(CB_VHO2, 1);
                tile_regs_acquire();
                v_rs_sum(CB_VOWN, 0, CB_VRS, vk, 0, 2, 0);   // (gate k, up k) -> DST 0, 1
                reconfig_data_format(CB_VJ, CB_VRT);
                matmul_block_init(CB_VRT, CB_VJ, 0, 1, 1, 1);
                matmul_block(CB_VRT, CB_VJ, 0, 0, 2, 0, 1, 1, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                mul_binary_tile(1, 2, 1);
                silu_tile_init();
                silu_tile(0);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_VHO);
                pack_tile(0, CB_VHO);
                pack_tile(0, CB_VHO2);
                tile_regs_release();
                cb_push_back(CB_VHO, 1);
                cb_push_back(CB_VHO2, 1);
                cb_pop_front(CB_VRS, 16);
                cb_pop_front(CB_VOWN, 2);
                cb_pop_front(CB_VRT, 1);
            }
            cb_wait_front(CB_VH, V_NNM);
            {
                CZONE("C-VDN");
                reconfig_data_format(CB_WV, CB_VH);
                pack_reconfig_data_format(CB_VDO);
                V_MM_INIT(CB_VH, CB_WV, V_DSB, 1, vnn);
                for (uint32_t bi = 0; bi < 4; ++bi) {
                    // the first-finished chunk (per device, §15) -> CB_VDO2 (reader); the other -> CB_VDO (writer).
                    // CB_WV is popped before CB_VDO is pushed (the writer's next plan_layer reuses the slot).
                    const uint32_t b = (bi + 2 * vfirst) & 3;
                    const uint32_t ocb = bi < 2 ? CB_VDO2 : CB_VDO;
                    if ((bi & 1) == 0) cb_reserve_back(ocb, 2 * V_DSB);
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vnn; ++j) V_MM(CB_VH, CB_WV, j, 2 * V_KTR * vnn + b * vnn * V_DSB + j * V_DSB, 0, V_DSB, 1, vnn);
                    tile_regs_commit();
                    tile_regs_wait();
                    MK_NOUNROLL
                    for (uint32_t t = 0; t < V_DSB; ++t) pack_tile<true>(t, ocb, (b & 1) * V_DSB + t);
                    tile_regs_release();
                    if (bi == 1) cb_push_back(CB_VDO2, 2 * V_DSB);
                    if (bi == 3) { cb_pop_front(CB_WV, V_SLOT); cb_push_back(CB_VDO, 2 * V_DSB); }
                }
            }
            cb_pop_front(CB_VH, V_NNM);
        }
#if !defined(MKA_QSK)
        cb_pop_front(CB_X, KT);
#endif
#else
        cb_wait_front(CB_GO, 1);
        cb_pop_front(CB_GO, 1);
        if (IS_GU) {
            reconfig_data_format(CB_W, CB_X);
            matmul_block_init(CB_X, CB_W, 0, 2, 1, KB);
            CZONE("C-GU");
            tile_regs_acquire();
            constexpr uint32_t NBG = KT / KB;
            for (uint32_t b0 = 0, g = 0; b0 < NBG; b0 += GRP, ++g) {
                const uint32_t nb = (NBG - b0) < GRP ? (NBG - b0) : GRP;
                const uint32_t cbw = (g & 1) ? CB_W2 : CB_W;
                cb_wait_front(cbw, nb * KB * 2);
                for (uint32_t b = 0; b < nb; ++b)
                    for (uint32_t k = 0; k < KB; ++k)
                        matmul_block(CB_X, cbw, (b0 + b) * KB + k, b * KB * 2 + k * 2, 0, 0, 2, 1, KB);
                cb_pop_front(cbw, nb * KB * 2);
            }
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(CB_HL, 1);
            tile_regs_wait();
            pack_reconfig_data_format(CB_HL);
            pack_tile(0, CB_HL);
            tile_regs_release();
            cb_push_back(CB_HL, 1);
        }
        cb_pop_front(CB_X, KT);
        reconfig_data_format(CB_WD, CB_H);
        matmul_block_init(CB_H, CB_WD, 0, NDN, 1, HT);
        cb_wait_front(CB_WD, HT * NDN);
#if defined(MK_DNPIPE) && MK_DNPIPE
        CZONE("C-DN");
        tile_regs_acquire();
        {   // -mk-tail port: accumulate the down K-dim root chunk by root chunk as h arrives (4 roots x HT/4 tiles)
            constexpr uint32_t HCHC = HT / 4;
            for (uint32_t i = 0; i < 4; ++i) {
                cb_wait_front(CB_H, HCHC * (i + 1));
                for (uint32_t k = i * HCHC; k < (i + 1) * HCHC; ++k) matmul_block(CB_H, CB_WD, k, k * NDN, 0, 0, NDN, 1, HT);
            }
        }
#else
        cb_wait_front(CB_H, HT);
        CZONE("C-DN");
        tile_regs_acquire();
        for (uint32_t k = 0; k < HT; ++k) matmul_block(CB_H, CB_WD, k, k * NDN, 0, 0, NDN, 1, HT);
#endif
        tile_regs_commit();
        cb_reserve_back(CB_OUT, NDN);
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        for (uint32_t j = 0; j < NDN; ++j) pack_tile(j, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, NDN);
        cb_pop_front(CB_WD, HT * NDN);
        cb_pop_front(CB_H, HT);
#endif
    }
}

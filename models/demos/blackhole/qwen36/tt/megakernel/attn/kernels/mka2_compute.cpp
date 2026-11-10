// MKA2 compute, full grid, persistent over NL attention-type layers. Per layer L:
//   STREAM: mixer y = x @ Wmix[:, MIXN cols] (qkv in-proj stand-in).
//   all:    SDPA over this core's KV slice (flash-decode, O_aug = Σ exp(Q·Kᵀ)·[V | 1], fp32 packer L1 accumulation),
//           tree children added into CB_O, the group root normalizes into CB_OA.
//   STREAM: (after GO) fm11 MLP: gate|up -> silu·mul -> h, down -> CB_OUT.
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
#include "api/debug/kernel_profiler.hpp"
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
#endif
#if defined(MKA_RQKV)
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_WM = 6, CB_YM = 7, CB_GO = 8, CB_OUT = 16;
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12, CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28;
constexpr uint32_t CB_RS = 13, CB_QA = 14, CB_QO = 17, CB_QR = 18, CB_PQ = 19, CB_RSP = 20;
#if defined(MKA_RWO)
constexpr uint32_t QO_SLOT = 5;
#else
constexpr uint32_t QO_SLOT = 0;
#endif
constexpr uint32_t CB_A = 15, CB_S = 21, CB_AS = 22, WK = 48, PQ_QSL = 320;
#ifndef MKA_NU
#define MKA_NU 4
#endif
constexpr uint32_t DH = 8, NO = DH + 1, QT = 8 * MKA_NU;
#ifndef MKA_AMASK
#define MKA_AMASK 0xFFFFFFFFu
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
    const uint32_t IS_GU = get_arg_val<uint32_t>(a++);
    const uint32_t u = get_arg_val<uint32_t>(a++), nblk = get_arg_val<uint32_t>(a++);
    const uint32_t nrecv = get_arg_val<uint32_t>(a++), is_root = get_arg_val<uint32_t>(a++);
    const uint32_t qoff = u * DH;
#if defined(MKA_RQKV)
    const uint32_t qrole = get_arg_val<uint32_t>(a++), qk0 = get_arg_val<uint32_t>(a++), qnk = get_arg_val<uint32_t>(a++);
    const uint32_t qnt = get_arg_val<uint32_t>(a++), gdn_on = get_arg_val<uint32_t>(a++);
#if defined(MKA_RWO)
    const uint32_t qn0 = get_arg_val<uint32_t>(a++), wo_nt = get_arg_val<uint32_t>(a++);   // first qkv tile, wo tiles (0 = no wo)
#endif
#if defined(MK_V12)
    const uint32_t vrole_rt = get_arg_val<uint32_t>(a++), vnn = get_arg_val<uint32_t>(a++), vk = get_arg_val<uint32_t>(a++);
#if defined(MKA_ROLE_STREAM)
    constexpr uint32_t vrole = 1;
#elif defined(MKA_ROLE_EXT)
    const uint32_t vrole = vrole_rt == 2 ? 2u : 0u;
#else
    const uint32_t vrole = vrole_rt;
#endif
    const uint32_t vfirst = get_arg_val<uint32_t>(a++);
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
    mkar::Reduce ar_red;   // [has_work, ring, PT] from the owner's L1 arg table (spec v1.25): has_work = this device owns it
    ar_red.has_work = 0;
    if (vrole == 2) ar_red.init_l1(MK_AR_BLK, MK_AR_BLK_COMP);
#endif
    // owner: Σ of the 10 column down partials (P @ I, fp32 DST) [+ the x chunk while the AR is a stall] -> CB_VXO
    auto v12_osum = [&](uint32_t L) {
        if (vrole != 2) return;
        if (L == 0) cb_wait_front(CB_VI, 1);
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_VP, CB_VI, CB_VXO);
        cb_wait_front(CB_VP, V_NCOL * V_OT);
#if defined(MK_OWNER_RESID) && MK_OWNER_RESID
        if (L == 0) cb_wait_front(CB_VX0, V_OT);
#endif
        cb_reserve_back(CB_VXO, V_OT);
        CZONE("C-OSUM");
        pack_reconfig_data_format(CB_VXO);
        for (uint32_t b = 0; b < 2; ++b) {
            tile_regs_acquire();
            reconfig_data_format(CB_VI, CB_VP);
            matmul_block_init(CB_VP, CB_VI, 0, 1, 1, 1);
            for (uint32_t n = 0; n < V_NCOL; ++n)
                for (uint32_t t = 0; t < 5; ++t) matmul_block(CB_VP, CB_VI, n * V_OT + b * 5 + t, 0, t, 0, 1, 1, 1);
#if defined(MK_OWNER_RESID) && MK_OWNER_RESID
            reconfig_data_format(CB_VI, CB_VX0);
            matmul_block_init(CB_VX0, CB_VI, 0, 1, 1, 1);
            for (uint32_t t = 0; t < 5; ++t) matmul_block(CB_VX0, CB_VI, b * 5 + t, 0, t, 0, 1, 1, 1);
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 5; ++t) pack_tile<true>(t, CB_VXO, b * 5 + t);
            tile_regs_release();
        }
        cb_push_back(CB_VXO, V_OT);
        cb_pop_front(CB_VP, V_NCOL * V_OT);
        // AR reduce (real AR, spec v0.9.2) follows here on the owner TRISCs
#if defined(MK_REAL_AR) && defined(MKA_ROLE_EXT)
        if (ar_red.has_work) {
            CZONE("C-AR");
            compute_kernel_hw_startup(MKAR_CB_RED, MKAR_CB_RED, MKAR_CB_OUT);
            ar_red.iter();
        }
#endif
    };
#define V12_OSUM(L) v12_osum(L)
#else
#define V12_OSUM(L) do {} while (0)
#endif
    // real qkv PROJ (spec v0.4): y = rstd ⊙ (x @ diag(γ)W); rstd = rsqrt(ΣΣx²/H + eps) from the AR owners' fp32 partials
    auto qkv_proj = [&](bool att_layer) {
        CZONE("C-QKV");
        const uint32_t cbw = qrole == 3 ? CB_PQ : CB_WM;
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, cbw, CB_QA);
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
#endif
#ifndef V12_OSUM
#define V12_OSUM(L) do {} while (0)
#endif
    if (IS_STREAM) compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_HL);
    else compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_P);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t L = 0; L < NL; ++L) {
        const bool att = ((uint32_t)(MKA_AMASK) >> L) & 1u;
        if (!IS_STREAM && !att) { V12_OSUM(L); continue; }
#if defined(MKA_RQKV)
        // attention layers: the real qkv PROJ; GDN-type layers: the STREAM K-half matmul alone stands in for the GDN
        // in-proj (64 x 87 KB BFP8 = 5.6 MB, spec 5.9 MB), output discarded
        if (IS_STREAM) {
            cb_wait_front(CB_X, KT);
            if (qrole && (att || gdn_on)) qkv_proj(att);
        } else if (qrole == 3) {
            cb_wait_front(CB_X, KT);
            qkv_proj(true);
            cb_pop_front(CB_X, KT);
        }
        if (false) {
#else
        if (IS_STREAM) {
#endif
            cb_wait_front(CB_X, KT);
            CZONE("C-MIX");
#if !defined(MKA_NO_REINIT)
            if (L > 0) compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_HL);
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
            if (L > 0 || qrole) compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_P);   // after the qkv bcast-mul
#else
            if (L > 0 && (IS_STREAM || ((uint32_t)(MKA_AMASK) & ((1u << L) - 1u)))) compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_P);
#endif
#endif
            cb_wait_front(CB_Q, QT);
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
                exp_tile_init();
                for (uint32_t i = 0; i < nb; ++i) exp_tile(i);
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
                for (uint32_t i = 0; i < nb; ++i)
                    for (uint32_t t = 0; t < DH; ++t) matmul_tiles(CB_P, CB_V, i, i * DH + t, t);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_O);
                pack_reconfig_l1_acc(first ? 0 : 1);
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
            for (uint32_t i = 0; i < nrecv; ++i) {
                cb_wait_front(CB_R, NO);
                reconfig_data_format_srca(CB_R);
                copy_init(CB_R);
                for (uint32_t t0 = 0; t0 < NO; t0 += DH) {
                    const uint32_t n = (NO - t0) < DH ? (NO - t0) : DH;
                    tile_regs_acquire();
                    for (uint32_t t = 0; t < n; ++t) copy_tile(CB_R, t0 + t, t);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(CB_O);
                    pack_reconfig_l1_acc(1);
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
            compute_kernel_hw_startup<SrcOrder::Reverse>(CB_A, CB_S, CB_AS);
            cb_wait_front(CB_A, WK);
            cb_wait_front(CB_S, WK);
            reconfig_data_format(CB_A, CB_S);
            mul_init(CB_A, CB_S);
            pack_reconfig_data_format(CB_AS);
            for (uint32_t t = 0; t < WK; ++t) {
                tile_regs_acquire();
                mul_tiles(CB_A, CB_S, t, t, 0);
                tile_regs_commit();
                cb_reserve_back(CB_AS, 1);
                tile_regs_wait();
                pack_tile(0, CB_AS);   // S slot t, already unpacked
                tile_regs_release();
                cb_push_back(CB_AS, 1);
            }
            cb_pop_front(CB_A, WK);
            cb_pop_front(CB_S, WK);
            cb_wait_front(CB_AS, WK);
            cb_wait_front(CB_PQ, PQ_QSL);
            reconfig_data_format(CB_PQ, CB_AS);
            matmul_block_init(CB_AS, CB_PQ, 0, wo_nt, 1, WK);
            tile_regs_acquire();
            for (uint32_t k = 0; k < WK; ++k) matmul_block(CB_AS, CB_PQ, k, k * wo_nt, 0, 0, wo_nt, 1, WK);
            tile_regs_commit();
            cb_reserve_back(CB_QO, QO_SLOT);
            tile_regs_wait();
            pack_reconfig_data_format(CB_QO);
            for (uint32_t j = 0; j < wo_nt; ++j) pack_tile(j, CB_QO);
            tile_regs_release();
            cb_push_back(CB_QO, QO_SLOT);
            cb_pop_front(CB_AS, WK);
            cb_pop_front(CB_PQ, PQ_QSL);
        }
#endif
        if (!IS_STREAM) { V12_OSUM(L); continue; }
#if defined(MKA_NOMLP)
        cb_wait_front(CB_GO, 1); cb_pop_front(CB_GO, 1); cb_pop_front(CB_X, KT); continue;
#endif
#if defined(MK_V12)
        cb_wait_front(CB_GO, 1);
        cb_pop_front(CB_GO, 1);
        if (vrole == 1) {
            const bool own_h = vk < vnn;
            if (L == 0) { cb_wait_front(CB_VI, 1); cb_wait_front(CB_VJ, 1); }
            compute_kernel_hw_startup<SrcOrder::Reverse>(CB_XK, CB_WV, CB_VA);
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
                    for (uint32_t j = 0; j < vnn; ++j) pack_tile<true>(j, CB_VA, half * V_NNM + j);
                    if (own_h) pack_tile<true>(vk, CB_VOWN, half);
                    tile_regs_release();
                }
            }
            cb_push_back(CB_VA, V_NTM);
            if (own_h) cb_push_back(CB_VOWN, 2);
            cb_pop_front(CB_XK, V_KTR);
            if (own_h) {
                // rstd' from the CB 20 stand-in (same producer as the qkv PROJ): rsqrt(Σ partials / H + eps), col 0
                cb_wait_front(CB_VRSA, MKA_NOWN);
                cb_reserve_back(CB_VRT, 1);
                tile_regs_acquire();
#if MKA_NOWN == 2
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
                cb_pop_front(CB_VRSA, MKA_NOWN);
                // h_k = silu(rstd ⊙ Σ_r gate_k[r]) * (rstd ⊙ Σ_r up_k[r]); sums as P @ I (exact, fp32 DST); rstd
                // broadcast along the row as Rcol @ J (J: row 0 = 1)
                cb_wait_front(CB_VRT, 1);
                cb_wait_front(CB_VOWN, 2);
                cb_wait_front(CB_VRS, 16);
                CZONE("C-VRS");
                cb_reserve_back(CB_VHO, 1);
                cb_reserve_back(CB_VHO2, 1);
                tile_regs_acquire();
                reconfig_data_format(CB_VI, CB_VOWN);
                matmul_block_init(CB_VOWN, CB_VI, 0, 1, 1, 1);
                matmul_block(CB_VOWN, CB_VI, 0, 0, 0, 0, 1, 1, 1);
                matmul_block(CB_VOWN, CB_VI, 1, 0, 1, 0, 1, 1, 1);
                reconfig_data_format(CB_VI, CB_VRS);
                matmul_block_init(CB_VRS, CB_VI, 0, 1, 1, 1);
                for (uint32_t r = 0; r < 8; ++r) {
                    if (r == vk) continue;
                    matmul_block(CB_VRS, CB_VI, 2 * r, 0, 0, 0, 1, 1, 1);
                    matmul_block(CB_VRS, CB_VI, 2 * r + 1, 0, 1, 0, 1, 1, 1);
                }
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
                    for (uint32_t t = 0; t < V_DSB; ++t) pack_tile<true>(t, ocb, (b & 1) * V_DSB + t);
                    tile_regs_release();
                    if (bi == 1) cb_push_back(CB_VDO2, 2 * V_DSB);
                    if (bi == 3) { cb_pop_front(CB_WV, V_SLOT); cb_push_back(CB_VDO, 2 * V_DSB); }
                }
            }
            cb_pop_front(CB_VH, V_NNM);
        }
        cb_pop_front(CB_X, KT);
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

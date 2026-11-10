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
#if defined(MKA_RQKV)
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_WM = 6, CB_YM = 7, CB_GO = 8, CB_OUT = 16;
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12, CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28;
constexpr uint32_t CB_RS = 13, CB_QA = 14, CB_QO = 17, CB_QR = 18, CB_PQ = 19, CB_RSP = 20;
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
    const uint32_t IS_STREAM = get_arg_val<uint32_t>(a++), IS_GU = get_arg_val<uint32_t>(a++);
    const uint32_t u = get_arg_val<uint32_t>(a++), nblk = get_arg_val<uint32_t>(a++);
    const uint32_t nrecv = get_arg_val<uint32_t>(a++), is_root = get_arg_val<uint32_t>(a++);
    const uint32_t qoff = u * DH;
#if defined(MKA_RQKV)
    const uint32_t qrole = get_arg_val<uint32_t>(a++), qk0 = get_arg_val<uint32_t>(a++), qnk = get_arg_val<uint32_t>(a++);
    const uint32_t qnt = get_arg_val<uint32_t>(a++);
    // real qkv PROJ (spec v0.4): y = rstd ⊙ (x @ diag(γ)W); rstd = rsqrt(ΣΣx²/H + eps) from the AR owners' fp32 partials
    auto qkv_proj = [&](bool att_layer) {
        DeviceZoneScopedN("C-QKV");
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
        tile_regs_commit();
        cb_reserve_back(CB_QO, qnt);
        tile_regs_wait();
        pack_reconfig_data_format(CB_QO);
        for (uint32_t j = 0; j < qnt; ++j) pack_tile(j, CB_QO);
        tile_regs_release();
        cb_push_back(CB_QO, qnt);
        cb_pop_front(CB_QA, qnt);
        cb_pop_front(CB_RS, 1);
    };
#endif
    if (IS_STREAM) compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_HL);
    else compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_P);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t L = 0; L < NL; ++L) {
        const bool att = ((uint32_t)(MKA_AMASK) >> L) & 1u;
        if (!IS_STREAM && !att) continue;
#if defined(MKA_RQKV)
        // attention layers: the real qkv PROJ; GDN-type layers: the STREAM K-half matmul alone stands in for the GDN
        // in-proj (64 x 87 KB BFP8 = 5.6 MB, spec 5.9 MB), output discarded
        if (IS_STREAM) {
            cb_wait_front(CB_X, KT);
            if (qrole) qkv_proj(att);
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
            DeviceZoneScopedN("C-MIX");
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
            DeviceZoneScopedN("C-SDPA");
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
        if (!IS_STREAM) continue;
#if defined(MKA_NOMLP)
        cb_wait_front(CB_GO, 1); cb_pop_front(CB_GO, 1); cb_pop_front(CB_X, KT); continue;
#endif
        cb_wait_front(CB_GO, 1);
        cb_pop_front(CB_GO, 1);
        if (IS_GU) {
            reconfig_data_format(CB_W, CB_X);
            matmul_block_init(CB_X, CB_W, 0, 2, 1, KB);
            DeviceZoneScopedN("C-GU");
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
        DeviceZoneScopedN("C-DN");
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
        DeviceZoneScopedN("C-DN");
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
    }
}

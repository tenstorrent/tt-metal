// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear, compute. y[n] = sum_kb s[n, kb] * sum_{k in kb} x'[k] T[k, n], kb = 128-wide K blocks.
// The activation arrives block-diagonal: compact tile kt holds x'[32kt .. 32kt+31] in row (kt / 4) % 32, so a matmul
// over row block m (32 K-blocks, 128 K tiles) leaves P[kb, n] in row kb - 32m. Per row block: LoFi matmul with the
// ternary tiles (exact +-1 in srcA), times the block's fp16 scale tile in fp32 on the SFPU, packed with L1
// accumulation into Q (fp32). Then y = ones @ Q (HiFi4: Q in srcA keeps 9 bits) sums Q's rows.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_act = tt::CBIndex::c_0;
constexpr uint32_t cb_w0 = tt::CBIndex::c_1;  // even records (NCRISC)
constexpr uint32_t cb_w1 = tt::CBIndex::c_6;  // odd records (BRISC)
constexpr uint32_t cb_ones = tt::CBIndex::c_3;
constexpr uint32_t cb_q = tt::CBIndex::c_24;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

ALWI void ternary_mm_init(uint32_t cb_w) {
    matmul_init(cb_act, cb_w);
    MATH((llk_math_matmul_init<MathFidelity::LoFi, MM_THROTTLE>(cb_act, cb_w, 0)));
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127);
    }));
}

// The scale tile rides in cb_w: 8 pages of 256 B holding one 2048-B fp16 tile.
ALWI void scale_copy_init() {
    copy_init(cb_q);
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        // Initialised as cb_q (Float32, srcA Tf32); the scale tiles are IEEE fp16, which Tf32 holds exactly.
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Tf32, 2048 >> 4, FACE_R_DIM, 4);
    }));
}

// The format tracker in matmul_init / copy_init cannot see the manual reconfigs above: it believes srcA holds the
// declared format of the last CB it configured, so after scale_copy_init (initialised as cb_q) a cb_q init is
// skipped as a no-op. Every cb_q read sets srcA back to Float32 itself.
ALWI void srca_fp32() {
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Tf32, 4096 >> 4, FACE_R_DIM, 4);
    }));
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(2);         // 1: P of block 0, 2: P*S of block 0, 3: Q
    constexpr uint32_t record_pages = get_compile_time_arg_val(3);  // 256-B pages per record, 4 KB padding included
    constexpr uint32_t pad_pages = record_pages - Kt - Mt * 8;
    const uint32_t num_nt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w0, cb_out);

    if constexpr (DEBUG == 8) {
        cb_wait_front(cb_act, Kt);
    }
    if constexpr (DEBUG == 9 || DEBUG == 8) {
        // Streaming only: consume every page, no math; output tiles are left as they are.
        for (uint32_t i = 0; i < num_nt; ++i) {
            const uint32_t cb_w = (i & 1) ? cb_w1 : cb_w0;
            for (uint32_t p = 0; p < record_pages; p += 8) {
                cb_wait_front(cb_w, 8);
                cb_pop_front(cb_w, 8);
            }
            cb_reserve_back(cb_out, 1);
            cb_push_back(cb_out, 1);
        }
        return;
    }
    cb_wait_front(cb_act, Kt);
    cb_wait_front(cb_ones, 1);
    for (uint32_t i = 0; i < num_nt; ++i) {
        const uint32_t cb_w = (i & 1) ? cb_w1 : cb_w0;
        cb_reserve_back(cb_q, 1);
        for (uint32_t m = 0; m < Mt; ++m) {
            const uint32_t k0 = m * 128;
            const uint32_t nk = (Kt - k0 < 128) ? (Kt - k0) : 128;
            ternary_mm_init(cb_w);
            tile_regs_acquire();
            // nk is a multiple of 16 for every Bonsai K (5120, 6144, 17408): one 4-KB DRAM page per wait.
            for (uint32_t k = 0; k < nk; k += 16) {
                cb_wait_front(cb_w, 16);
                for (uint32_t j = 0; j < 16; ++j) {
                    UNPACK((llk_unpack_AB_matmul(cb_act, cb_w, k0 + k + j, j)));
                    MATH((llk_math_matmul<MathFidelity::LoFi, MM_THROTTLE>(0)));
                }
                cb_pop_front(cb_w, 16);
            }
            scale_copy_init();
            cb_wait_front(cb_w, 8);
            copy_tile(cb_w, 0, 1);
            cb_pop_front(cb_w, 8);
            if constexpr (DEBUG != 1) {
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_q);
            pack_reconfig_l1_acc((m > 0 && DEBUG != 1 && DEBUG != 2) ? 1 : 0);
            pack_tile<true>(0, cb_q, 0);
            tile_regs_release();
        }
        if constexpr (pad_pages > 0) {
            cb_wait_front(cb_w, pad_pages);
            cb_pop_front(cb_w, pad_pages);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_q, 1);

        if constexpr (DEBUG != 0) {
            cb_wait_front(cb_q, 1);
            tile_regs_acquire();
            copy_init(cb_q);
            srca_fp32();
            copy_tile(cb_q, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_q, 1);
            continue;
        }
        matmul_init(cb_ones, cb_q);
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(cb_ones, cb_q, 0)));
        srca_fp32();
        cb_wait_front(cb_q, 1);
        tile_regs_acquire();
        UNPACK((llk_unpack_AB_matmul(cb_ones, cb_q, 0, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(0)));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_q, 1);
    }
}

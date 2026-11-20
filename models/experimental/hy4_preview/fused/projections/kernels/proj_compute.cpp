// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One projection unit of the Hy4 decode projection programs (TRISC) (fused/projections/program.py): the 8 token rows
// ([ROWS, 32] BF16 row tiles, resident in L1) times this core's `ct` output tile columns of a weight streamed from
// DRAM (custom_mm, LoFi, FP32 accumulation), then the unit's finishing in the SFPU (proj_sfpu.hpp) and one pack.
//
// The weights arrive K-major in blocks of kb K tiles (per K tile the ct output tiles); with several terms the same
// K range arrives again and accumulates into the same sums (a weight stored as the sum of two BFP4 terms).
//
// runtime args: [in_first, aux]   the unit's first left-operand tile; RoPE modes: the cos tile in cb_aux (sin is
//                                  the next one), FIN_GATE: the unit's first gate tile in cb_aux
// compile-time args: [cb_in, cb_w, cb_out, cb_sc, cb_stat, cb_aux, in_wait, kb, blocks_per_term, ct, fin, rnd,
//                     param]      param: proj_sfpu.hpp finish

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif
#include "proj_common.hpp"
#include "proj_sfpu.hpp"

void kernel_main() {
#ifdef HY4_EMPTY  // measurement only: the launch floor of the program
    return;
#endif
    const uint32_t in_first = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t aux = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stat = get_compile_time_arg_val(4);
    constexpr uint32_t cb_aux = get_compile_time_arg_val(5);
    constexpr uint32_t in_wait = get_compile_time_arg_val(6);
    constexpr uint32_t kb = get_compile_time_arg_val(7);
    constexpr uint32_t blocks_per_term = get_compile_time_arg_val(8);
    constexpr uint32_t ct = get_compile_time_arg_val(9);
    constexpr uint32_t fin = get_compile_time_arg_val(10);
    constexpr uint32_t rnd = get_compile_time_arg_val(11);
    constexpr uint32_t param = get_compile_time_arg_val(12);

    constexpr bool has_const = hy4proj::fin_has_const(fin);
    constexpr bool has_rope = fin == hy4proj::FIN_NORM_ROPE || fin == hy4proj::FIN_SCALE_ROPE;
    constexpr bool has_sfpu = fin != hy4proj::FIN_NONE;
    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DenseSplitAcc;
    static_assert(ct >= 1 && ct <= 14, "the output tiles and the constant page share DEST");
    static_assert(fin != hy4proj::FIN_STAT || ct <= 12, "the statistic tile is in standard slot 6");
    static_assert(!has_rope || ct == 2, "RoPE: the 64 rope dims are 2 tiles");
    static_assert(fin != hy4proj::FIN_GATE || ct <= 4, "the gate tiles sit in standard slots 2 .. 5");

    custom_mm_block_init<transpose, dest_layout>(cb_in, cb_w, cb_out, ct);

    uint32_t n_blocks = 0;  // 0: the core was not released, nothing to do
    UNPACK(n_blocks = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n_blocks = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n_blocks = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (n_blocks == 0) {
        return;
    }

    cb_wait_front(cb_in, in_wait);
    if constexpr (has_const) {
        cb_wait_front(cb_sc, 1);
    }
    tile_regs_acquire();

    // what the SFPU reads besides the sums: copied into the upper DEST slots before the matmul
    if constexpr (has_const) {
        reconfig_full_operand_srca(cb_sc);
        copy_init(cb_sc);
        copy_tile(cb_sc, 0, 7);
    }
    if constexpr (has_rope) {
        reconfig_full_operand_srca(cb_aux);
        copy_init(cb_aux);
        copy_tile(cb_aux, aux, 5);
        copy_tile(cb_aux, aux + 1, 6);
    }
    if constexpr (fin == hy4proj::FIN_GATE) {
        reconfig_full_operand_srca(cb_aux);
        copy_init(cb_aux);
        for (uint32_t i = 0; i < ct; ++i) {
            copy_tile(cb_aux, aux + i, 2 + i);
        }
    }
    if constexpr (has_const || fin == hy4proj::FIN_GATE) {
        reconfig_full_operand(cb_w, cb_in);
        custom_mm_block_init_short<transpose, dest_layout>(cb_in, cb_w, cb_out, ct);
    }

    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint32_t k0 = in_first + (b % blocks_per_term) * kb;
        cb_wait_front(cb_w, kb * ct);
#ifndef HY4_NO_MATH
        if (b + 1 == n_blocks) {
            custom_mm_block<dest_layout, true>(cb_in, cb_w, k0, 0, 0, kb, ct);
        } else {
            custom_mm_block<dest_layout, false>(cb_in, cb_w, k0, 0, 0, kb, ct);
        }
#endif
        cb_pop_front(cb_w, kb * ct);
    }

    if constexpr (has_sfpu) {
        MATH((_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>()));
        if constexpr (fin == hy4proj::FIN_SIGMOID) {
            MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));
        }
        if constexpr (fin == hy4proj::FIN_NORM || fin == hy4proj::FIN_NORM_ROPE) {
            MATH((ckernel::sfpu::sqrt_init<false>()));
        }
        MATH((_llk_math_eltwise_unary_sfpu_params_(hy4proj::finish<ct, fin, rnd, param>, 0, VectorMode::RC_custom)));
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, ct);
    if constexpr (fin == hy4proj::FIN_STAT) {
        cb_reserve_back(cb_stat, 1);
    }
    tile_regs_wait();
    for (uint32_t c = 0; c < ct; ++c) {
        pack_tile(c, cb_out);
    }
    if constexpr (fin == hy4proj::FIN_STAT) {
        // (the format change puts the packer back on the standard 64-row tile stride: the statistic tile is in
        // standard slot 6)
        pack_reconfig_data_format(cb_stat);
        pack_tile(6, cb_stat);
    }
    tile_regs_release();
    cb_push_back(cb_out, ct);
    if constexpr (fin == hy4proj::FIN_STAT) {
        cb_push_back(cb_stat, 1);
    }
    custom_mm_block_uninit<dest_layout>(cb_out);
}

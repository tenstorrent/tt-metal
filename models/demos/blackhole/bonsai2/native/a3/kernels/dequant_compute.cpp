// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill dequant, compute. Per unit: 32 code tiles x their group's scale row (row-broadcast
// straight out of the fp32 scale_rows tile via bcast_row_idx), packed to the output format.
// Code tile t of the unit is in group t/4, whose scale is row (G%4)*8 + t/4 of the scale tile.
// Runtime args: u0, nu (same split as the reader).

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "scale_loc.h"

// A's BONSAI_SCALE16: scale tiles hold fp16 (2 KB) in a CB declared bf16 as a carrier; SrcB is reprogrammed to
// Float16 for the bcast-row multiply (exact: the scales are fp16 in the GGUF).
template <bool s16>
inline void scale_srcb_fp16() {
    if constexpr (s16) {
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Float16, 2048)));
        MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>((uint32_t)DataFormat::Float16)));
    }
}
#include "api/compute/compute_kernel_api.h"

// A2b (PackedW): code tiles are 256 B with no exponent section; unpacker 0 (SrcA) forces exponent 127.
template <bool on>
inline void force_shared_exp() {
    UNPACK((cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127)));
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(on ? 1 : 0)));
}

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t ncols = get_compile_time_arg_val(2);
    constexpr bool sx = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t cpb = get_compile_time_arg_val(4), pk = get_compile_time_arg_val(5);
    constexpr uint32_t s_tiles = get_compile_time_arg_val(6), tail_t = get_compile_time_arg_val(7);
    constexpr uint32_t s_full = get_compile_time_arg_val(8), rem = get_compile_time_arg_val(9);
    constexpr bool s16 = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t dt = get_compile_time_arg_val(1);  // tiles per DST half: 8 (bf16 DST) or 4 (fp32)
    constexpr uint32_t cb_w = 0, cb_s = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_w, cb_s, cb_out);
    mul_bcast_rows_init(cb_w, cb_s);
    scale_srcb_fp16<s16>();
    if constexpr (sx) {
        force_shared_exp<true>();
    }
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t G = u / ncols;
        const uint32_t row0 = scale_loc<pk, s_tiles, tail_t, s_full, rem>((u % ncols) % cpb, G).row;
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_w, 32);
        for (uint32_t q = 0; q < 32 / dt; ++q) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < dt; ++t) {
                const uint32_t i = q * dt + t;
                mul_tiles_bcast_rows(cb_w, cb_s, i, 0, t, row0 + i / 4);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, dt);
            tile_regs_wait();
            for (uint32_t t = 0; t < dt; ++t) {
                pack_tile(t, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, dt);
        }
        cb_pop_front(cb_w, 32);
        cb_pop_front(cb_s, 1);
    }
}

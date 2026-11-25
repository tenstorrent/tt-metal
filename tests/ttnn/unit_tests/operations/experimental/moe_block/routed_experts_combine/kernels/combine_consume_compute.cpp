// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combine consume microbench, compute: the accumulator compute (routed_experts_combine accumulator_compute.cpp,
// combine_snap q = 2^-14) on rows of any quantized_row format, through quantized_row::RowWeigher
// (kernels/compute/quantized_row_weigh.hpp): each row x its gate into DEST, snapped to int32 counts, packed with L1
// accumulation into its token's sum; after the rows, each sum back to fp32 x q, bf16 out.
//
// Compile-time args: format, n_rows, H, tokens
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/compute/quantized_row_weigh.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// A DEST face: sign-magnitude int32 count -> fp32 count x q (accumulator_compute.cpp's readout)
template <uint32_t step_bits>
inline void count_to_float() {
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vSMag count = sfpi::dst_reg[0];
        sfpi::dst_reg[0] =
            sfpi::convert<sfpi::vFloat>(count, sfpi::RoundMode::Nearest) * __builtin_bit_cast(float, step_bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t kStepBits = 0x38800000u;         // q = 2^-14
constexpr uint32_t kInverseStepBits = 0x46800000u;  // 1 / q = 2^14
constexpr uint32_t kDestTiles = compute_kernel_lib::DEST_AUTO_LIMIT;

void kernel_main() {
    constexpr auto format = static_cast<quantized_row::Format>(get_compile_time_arg_val(0));
    constexpr uint32_t n_rows = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t tokens = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = quantized_row::flat_tiles(H);
    constexpr bool fp8 = format == quantized_row::Format::Fp8E4m3;
    constexpr uint32_t cb_row = 0, cb_w = 1, cb_fac = 2, cb_meta = 3, cb_out = 16, cb_acc = 24;
    constexpr uint32_t cb_b = fp8 ? cb_fac : cb_w;  // the weigh's srcB: the factors, or the gate
    using Weigher = quantized_row::RowWeigher<format, cb_row, cb_b, kInverseStepBits>;

    compute_kernel_hw_startup(cb_row, cb_b, cb_acc);
    Weigher::init();
    uint8_t has_rows[tokens];
    for (uint32_t i = 0; i < tokens; ++i) {
        has_rows[i] = 0;
    }
    cb_reserve_back(cb_acc, tokens * row_tiles);
    {
        DeviceZoneScopedN("CMB-CMP");
        for (uint32_t n = 0; n < n_rows; ++n) {
            cb_wait_front(cb_meta, 1);
            const uint32_t token = read_tile_value(cb_meta, 0, 0);
            const uint32_t gate = read_tile_value(cb_meta, 0, 1);
            cb_pop_front(cb_meta, 1);
            cb_wait_front(cb_row, row_tiles);
            cb_wait_front(cb_b, fp8 ? row_tiles : 1);
            pack_reconfig_l1_acc(has_rows[token] ? 1 : 0);
            for (uint32_t first = 0; first < row_tiles; first += kDestTiles) {
                const uint32_t count = row_tiles - first < kDestTiles ? row_tiles - first : kDestTiles;
                tile_regs_acquire();
                Weigher::weigh(first, count, gate);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < count; ++j) {
                    pack_tile<true>(j, cb_acc, token * row_tiles + first + j);
                }
                tile_regs_release();
            }
            has_rows[token] = 1;
            cb_pop_front(cb_row, row_tiles);
            cb_pop_front(cb_b, fp8 ? row_tiles : 1);
        }
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, tokens * row_tiles);
    cb_wait_front(cb_acc, tokens * row_tiles);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    MATH((SFPU_UNARY_INIT(unused)));
    for (uint32_t token = 0; token < tokens; ++token) {  // (every token got rows: n_rows >= tokens)
        cb_reserve_back(cb_out, row_tiles);
        for (uint32_t first = 0; first < row_tiles; first += kDestTiles) {
            const uint32_t count = row_tiles - first < kDestTiles ? row_tiles - first : kDestTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < count; ++j) {
                copy_tile(cb_acc, token * row_tiles + first + j, j);
            }
            for (uint32_t j = 0; j < count; ++j) {
                MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, count_to_float, (kStepBits), j, VectorMode::RC)));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < count; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, row_tiles);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reconfig_data_format.h"

// Same SFPU operations over only the physically necessary top faces.
// One SFPU iteration covers two rows; use the established <=4-row policy.
template <int Iterations>
ALWI void multiply_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, Iterations, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}
template <int Iterations>
ALWI void add_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, Iterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}
ALWI void multiply_valid(uint32_t a, uint32_t b, uint32_t out, uint32_t rows) {
    if (rows <= 4) {
        multiply_rows<2>(a, b, out);
    } else if (rows <= 8) {
        multiply_rows<4>(a, b, out);
    } else if (rows <= 16) {
        multiply_rows<8>(a, b, out);
    } else {
        mul_binary_tile(a, b, out);
    }
}
ALWI void add_valid(uint32_t a, uint32_t b, uint32_t out, uint32_t rows) {
    if (rows <= 4) {
        add_rows<2>(a, b, out);
    } else if (rows <= 8) {
        add_rows<4>(a, b, out);
    } else if (rows <= 16) {
        add_rows<8>(a, b, out);
    } else {
        add_binary_tile(a, b, out);
    }
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr bool copy_once = get_compile_time_arg_val(0);
    constexpr bool valid_compute = get_compile_time_arg_val(1);
    constexpr uint32_t cb_left = 2, cb_right = 3, cb_out = 4;
    compute_kernel_hw_startup(cb_left, cb_right, cb_out);
    if constexpr (copy_once) {
        copy_init(cb_left);
    }
    tile_regs_acquire();
    // One independently zeroed CPU-BRGEMM K block per core. Match the
    // existing fused indexer row masks and copy-once BF16 input contract.
    const uint32_t width = get_arg_val<uint32_t>(0);
    fill_tile_init();
    fill_tile(2, 0.0f);
    for (uint32_t k = 0; k < width; ++k) {
        cb_wait_front(cb_left, 1);
        cb_wait_front(cb_right, 1);
        if constexpr (!copy_once) {
            reconfig_data_format_srca(cb_left);
            copy_init(cb_left);
        }
        copy_tile(cb_left, 0, 0);
        if constexpr (!copy_once) {
            reconfig_data_format_srca(cb_right);
            copy_init(cb_right);
        }
        copy_tile(cb_right, 0, 1);
        // BF16 products are exact FP32 for the normal finite target
        // operands. Keep each product and accumulator addition separate:
        // the reader sends high,low; there is no pair sum or tree reduce.
        mul_binary_tile_init();
        if constexpr (valid_compute) {
            multiply_valid(0, 1, 0, rows);
        } else {
            mul_binary_tile(0, 1, 0);
        }
        add_binary_tile_init();
        if constexpr (valid_compute) {
            add_valid(2, 0, 2, rows);
        } else {
            add_binary_tile(2, 0, 2);
        }
        cb_pop_front(cb_left, 1);
        cb_pop_front(cb_right, 1);
    }
    cb_reserve_back(cb_out, 1);
    tile_regs_commit();
    tile_regs_wait();
    // Lossless FP32 partial; no BF16 boundary or reduced-mantissa reload.
    pack_tile(2, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

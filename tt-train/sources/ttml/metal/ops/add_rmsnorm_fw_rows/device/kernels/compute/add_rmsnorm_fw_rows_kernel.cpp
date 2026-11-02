// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rmsnorm_fw_rows compute (one item (row r, slice s); residual / delta streamed, gamma and the sum resident):
//   pass 1: sum_j = residual_j + delta_j (FPU add, fp32 DEST, packed to bf16 cb::sum); partial = sum over the
//           slice of (bf16 sum_j)^2 (FPU products accumulated onto one fp32 DEST register)  -> exchanged between
//           the row's cores
//   rms = sqrt(rowsum(sum_s partial_s) / C + eps), inv = 1 / rms
//   pass 2: out_j = (sum_j * gamma_j) * inv   (FPU row broadcast, then the inv scale through reuse_dest)
// Iteration 41 (iteration 29's move for the backward): the SFPU per-tile chains (five ops per tile in pass 1,
// four in pass 2, one tile per acquire) go to the FPU, four tiles per DEST acquire; the squares are taken from the
// bf16-rounded sums (the ones written out) and fp32 operands enter the FPU through 19-bit source registers.

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/add_rmsnorm_fw_rows_cbs.hpp"

namespace cb = add_rmsnorm_fw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t inv_c_bits = get_compile_time_arg_val(4);
constexpr uint32_t eps_bits = get_compile_time_arg_val(5);

FORCE_INLINE void load_tile(uint32_t cb_id, uint32_t idx, uint32_t reg) {
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, idx, reg);
}

// reg = X @ K for single tiles X (cb_x) and K (cb_k); K is a constant bf16 tile.
FORCE_INLINE void matmul_with_constant(uint32_t cb_x, uint32_t cb_k, uint32_t reg) {
    reconfig_data_format(cb_k, cb_x);
    matmul_init(cb_x, cb_k, 0);
    matmul_tiles(cb_x, cb_k, 0, 0, reg);
}

void kernel_main() {
    [[maybe_unused]] const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
    constexpr uint32_t group = 4U;  // pass-1 group: the fp32 DEST's four tiles (the reader's push size)

    compute_kernel_hw_startup(cb::residual, cb::delta, cb::sum);
    cb_wait_front(cb::zero, 1);
    cb_wait_front(cb::ones, 1);
    cb_wait_front(cb::gamma, St);

    // pass 1: sums (FPU, packed bf16 into the resident cb::sum) and the slice's sum of squares (FPU products of the
    // bf16 sums accumulated onto fp32 DEST register 0, round-tripped through cb::acc between groups since DEST does
    // not persist across a pack).
    for (uint32_t c = 0; c < ncols; c += group) {
        const uint32_t n = (c + group <= ncols) ? group : (ncols - c);
        const bool last = c + n >= ncols;
        cb_wait_front(cb::residual, group);
        cb_wait_front(cb::delta, group);
#ifdef DELTA_TRANSPOSED
        // delta's tiles are the transposes of the residual's: through DEST transposed (exact) into cb::delta_t
        tile_regs_acquire();
        reconfig_data_format_srca(cb::delta);
        transpose_init(cb::delta);
        for (uint32_t i = 0; i < n; ++i) {
            transpose_tile(cb::delta, i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb::delta_t, group);
        tile_regs_wait();
        pack_reconfig_data_format(cb::delta_t);
        for (uint32_t i = 0; i < group; ++i) {
            pack_tile(i < n ? i : 0U, cb::delta_t);
        }
        tile_regs_release();
        cb_push_back(cb::delta_t, group);
        cb_wait_front(cb::delta_t, group);
        constexpr uint32_t cb_d = cb::delta_t;
#else
        constexpr uint32_t cb_d = cb::delta;
#endif
        tile_regs_acquire();
        reconfig_data_format(cb::residual, cb_d);
        add_init(cb::residual, cb_d);
        for (uint32_t i = 0; i < n; ++i) {
            add_tiles(cb::residual, cb_d, i, i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb::sum, n);
        tile_regs_wait();
        pack_reconfig_data_format(cb::sum);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb::sum);
        }
        tile_regs_release();
        cb_push_back(cb::sum, n);
        cb_pop_front(cb::residual, group);
        cb_pop_front(cb::delta, group);
#ifdef DELTA_TRANSPOSED
        cb_pop_front(cb::delta_t, group);
#endif

        cb_wait_front(cb::sum, c + n);
        tile_regs_acquire();
        if (c == 0) {
            load_tile(cb::zero, 0, 0);
        } else {
            cb_wait_front(cb::acc, 1);
            load_tile(cb::acc, 0, 0);
            cb_pop_front(cb::acc, 1);
        }
        reconfig_data_format(cb::sum, cb::sum);
        mul_init(cb::sum, cb::sum, /*acc_to_dest=*/true);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(cb::sum, cb::sum, c + i, c + i, 0);  // DEST[0] += sum_j * sum_j
        }
        tile_regs_commit();
        pack_and_push(0, last ? cb::partial_out : cb::acc);
    }

    // row_acc = sum of the row's partials (slot order = slice order on every core: the same summation everywhere)
    cb_wait_front(cb::partials, S);
    tile_regs_acquire();
    load_tile(cb::partials, 0, 0);
    add_binary_tile_init();
    for (uint32_t j = 1; j < S; ++j) {
        load_tile(cb::partials, j, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    pack_and_push(0, cb::row_acc);
    cb_pop_front(cb::partials, S);

    // rms = sqrt(rowsum(row_acc) / C + eps), broadcast along columns; packed fp32 for the scale and bf16 for the
    // rms output (slice 0 writes it)
    cb_wait_front(cb::row_acc, 1);
    tile_regs_acquire();
    matmul_with_constant(cb::row_acc, cb::ones, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_c_bits);
    add_unary_tile(0, eps_bits);
    sqrt_tile_init();
    sqrt_tile(0);
    tile_regs_commit();
    cb_reserve_back(cb::rms_tmp, 1);
    cb_reserve_back(cb::rms_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb::rms_tmp);
    pack_tile(0, cb::rms_tmp);
    pack_reconfig_data_format(cb::rms_out);
    pack_tile(0, cb::rms_out);
    tile_regs_release();
    cb_push_back(cb::rms_tmp, 1);
    cb_push_back(cb::rms_out, 1);
    cb_pop_front(cb::row_acc, 1);

    // inv = 1 / rms
    cb_wait_front(cb::rms_tmp, 1);
    tile_regs_acquire();
    load_tile(cb::rms_tmp, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    pack_and_push(0, cb::inv);
    cb_pop_front(cb::rms_tmp, 1);
    cb_wait_front(cb::inv, 1);

    // pass 2: out = (sum * gamma) * inv from the resident bf16 sum, four tiles per DEST acquire on the FPU (the row
    // broadcast writes DEST, the inv scale takes the tile from DEST), in full blocks for the writer
    cb_wait_front(cb::sum, ncols);
    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
        tile_regs_acquire();
        reconfig_data_format(cb::sum, cb::gamma);
        mul_bcast_rows_init(cb::sum, cb::gamma);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles_bcast_rows(cb::sum, cb::gamma, c + i, c + i, i);
        }
        reconfig_data_format_srcb(cb::inv);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb::inv);
        for (uint32_t i = 0; i < n; ++i) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb::inv, 0, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb::out, block);
        tile_regs_wait();
        pack_reconfig_data_format(cb::out);
        for (uint32_t i = 0; i < block; ++i) {  // a shorter tail pads the block with unused tiles
            pack_tile(i < n ? i : 0U, cb::out);
        }
        tile_regs_release();
        cb_push_back(cb::out, block);
    }
    cb_pop_front(cb::inv, 1);
    cb_pop_front(cb::sum, ncols);
    cb_pop_front(cb::gamma, St);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Row-split add+RMSNorm compute (one Wc-tile slice of one row per core):
//   sum = a + b -> CB16 (+ bfp8 copy CB5); x2 = sum^2 -> CB6; partial = row-sum(x2)/W -> CB7 (to the writer)
//   ms = sum of the R partials (CB8, filled by the row group) ; inv = rsqrt(ms + eps) -> CB9
//   out = (sum * bcast_cols(inv)) * gamma_slice -> CB17
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_a = 0, cb_b = 1, cb_g = 2, cb_sc = 3, cb_eps = 4;
constexpr uint32_t cb_s = 5, cb_x2 = 6, cb_part = 7, cb_parts = 8, cb_inv = 9, cb_sum_out = 16, cb_out = 17;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
}  // namespace

void kernel_main() {
    constexpr uint32_t Wc = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t CH = get_compile_time_arg_val(2);
    static_assert(Wc % CH == 0, "slice width must be a multiple of the DST chunk");
    constexpr uint32_t n_chunks = Wc / CH;

    CircularBuffer ca(cb_a), cbb(cb_b), cg(cb_g), csc(cb_sc), ceps(cb_eps);
    CircularBuffer cs(cb_s), x2(cb_x2), part(cb_part), parts(cb_parts), inv(cb_inv), so(cb_sum_out), no(cb_out);
    compute_kernel_hw_startup(cb_a, cb_b, cb_sum_out);
    cg.wait_front(Wc);
    csc.wait_front(1);
    ceps.wait_front(1);
    ca.wait_front(Wc);
    cbb.wait_front(Wc);

    // sum = a + b -> residual slice + bfp8 working copy
    so.reserve_back(Wc);
    cs.reserve_back(Wc);
    reconfig_data_format(cb_a, cb_b);
    add_init(cb_a, cb_b);
    for (uint32_t c = 0; c < n_chunks; ++c) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < CH; ++j) {
            add_tiles(cb_a, cb_b, c * CH + j, c * CH + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_out);
        for (uint32_t j = 0; j < CH; ++j) {
            pack_tile(j, cb_sum_out, c * CH + j);
        }
        pack_reconfig_data_format(cb_s);
        for (uint32_t j = 0; j < CH; ++j) {
            pack_tile(j, cb_s, c * CH + j);
        }
        tile_regs_release();
    }
    so.push_back(Wc);
    cs.push_back(Wc);
    ca.pop_front(Wc);
    cbb.pop_front(Wc);
    // x2 = sum^2
    cs.wait_front(Wc);
    reconfig_data_format(cb_s, cb_s);
    pack_reconfig_data_format(cb_x2);
    mul_init(cb_s, cb_s);
    x2.reserve_back(Wc);
    for (uint32_t c = 0; c < n_chunks; ++c) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < CH; ++j) {
            mul_tiles(cb_s, cb_s, c * CH + j, c * CH + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < CH; ++j) {
            pack_tile(j, cb_x2, c * CH + j);
        }
        tile_regs_release();
    }
    x2.push_back(Wc);
    // partial = sum over this slice of x2 * (1/W)
    reconfig_data_format(cb_x2, cb_sc);
    pack_reconfig_data_format(cb_part);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_sc, cb_part);
    x2.wait_front(Wc);
    part.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wc; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_sc, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_part);
    tile_regs_release();
    part.push_back(1);
    x2.pop_front(Wc);
    reduce_uninit(cb_x2);
    // ms = sum of the row group's R partials; inv = rsqrt(ms + eps)
    parts.wait_front(R);
    reconfig_data_format_srca(cb_parts);
    pack_reconfig_data_format(cb_inv);
    copy_tile_init(cb_parts);
    inv.reserve_back(1);
    tile_regs_acquire();
    copy_tile(cb_parts, 0, 0);
    add_reuse_dest_init<D2B>(cb_parts);
    for (uint32_t j = 1; j < R; ++j) {
        add_reuse_dest_tiles<D2B>(cb_parts, j, 0);
    }
    reconfig_data_format_srca(cb_eps);
    add_reuse_dest_init<D2B>(cb_eps);
    add_reuse_dest_tiles<D2B>(cb_eps, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    inv.push_back(1);
    parts.pop_front(R);
    // out = (sum * inv) * gamma_slice
    inv.wait_front(1);
    no.reserve_back(Wc);
    for (uint32_t c = 0; c < n_chunks; ++c) {
        reconfig_data_format(cb_s, cb_inv);
        mul_bcast_cols_init(cb_s, cb_inv);
        tile_regs_acquire();
        for (uint32_t j = 0; j < CH; ++j) {
            mul_tiles_bcast_cols(cb_s, cb_inv, c * CH + j, 0, j);
        }
        reconfig_data_format_srca(cb_g);
        mul_reuse_dest_init<D2B>(cb_g);
        for (uint32_t j = 0; j < CH; ++j) {
            mul_reuse_dest_tiles<D2B>(cb_g, c * CH + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t j = 0; j < CH; ++j) {
            pack_tile(j, cb_out, c * CH + j);
        }
        tile_regs_release();
    }
    no.push_back(Wc);
    inv.pop_front(1);
    cs.pop_front(Wc);
}

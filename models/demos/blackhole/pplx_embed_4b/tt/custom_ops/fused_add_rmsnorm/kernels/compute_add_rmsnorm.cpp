// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused residual add + RMSNorm, one tile-row (Wt tiles) at a time:
//   sum  = a + b            -> CB 16 (residual dtype) and CB 5 (bfp8 copy the norm reads)
//   x2   = sum * sum        -> CB 6
//   ms   = row-sum(x2) / W  -> CB 7   (reduce with the 1/W scaler tile)
//   inv  = rsqrt(ms + eps)  -> CB 8
//   out  = (sum * bcast_cols(inv)) * gamma -> CB 17, gamma applied on the DST tile
// Every session touches <= CHUNK (4) tiles so fp32 accumulation in DST is fine.
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
constexpr uint32_t cb_s = 5, cb_x2 = 6, cb_ms = 7, cb_inv = 8, cb_sum_out = 16, cb_out = 17;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
}  // namespace

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t CH = get_compile_time_arg_val(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    static_assert(Wt % CH == 0, "row width must be a multiple of the DST chunk");
    constexpr uint32_t n_chunks = Wt / CH;

    CircularBuffer ca(cb_a), cbb(cb_b), cg(cb_g), csc(cb_sc), ceps(cb_eps);
    CircularBuffer cs(cb_s), x2(cb_x2), ms(cb_ms), inv(cb_inv), so(cb_sum_out), no(cb_out);
    compute_kernel_hw_startup(cb_a, cb_b, cb_sum_out);
    cg.wait_front(Wt);
    csc.wait_front(1);
    ceps.wait_front(1);

    for (uint32_t r = 0; r < num_rows; ++r) {
        ca.wait_front(Wt);
        cbb.wait_front(Wt);
        // sum = a + b -> residual output + bfp8 working copy
        so.reserve_back(Wt);
        cs.reserve_back(Wt);
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
        so.push_back(Wt);
        cs.push_back(Wt);
        ca.pop_front(Wt);
        cbb.pop_front(Wt);
        // x2 = sum^2
        cs.wait_front(Wt);
        reconfig_data_format(cb_s, cb_s);
        pack_reconfig_data_format(cb_x2);
        mul_init(cb_s, cb_s);
        x2.reserve_back(Wt);
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
        x2.push_back(Wt);
        // ms = mean(x2) over the row
        reconfig_data_format(cb_x2, cb_sc);
        pack_reconfig_data_format(cb_ms);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_sc, cb_ms);
        x2.wait_front(Wt);
        ms.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_sc, j, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_ms);
        tile_regs_release();
        ms.push_back(1);
        x2.pop_front(Wt);
        reduce_uninit(cb_x2);
        // inv = rsqrt(ms + eps)
        reconfig_data_format(cb_ms, cb_eps);
        pack_reconfig_data_format(cb_inv);
        add_init(cb_ms, cb_eps);
        ms.wait_front(1);
        inv.reserve_back(1);
        tile_regs_acquire();
        add_tiles(cb_ms, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_inv);
        tile_regs_release();
        inv.push_back(1);
        ms.pop_front(1);
        // out = (sum * inv) * gamma
        inv.wait_front(1);
        no.reserve_back(Wt);
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
        no.push_back(Wt);
        inv.pop_front(1);
        cs.pop_front(Wt);
    }
}

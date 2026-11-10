// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool compute, the pool phase of the block_topk_route compute (included there). Per unit: O[n] = sum over the
// group's tile rows k of P[k] @ X[k, n] (DT output tiles accumulated in fp32 DST), with the pooling tiles P[k]
// delivered by the reader in batches of 4 rows alongside the data; groups without tile rows output zero tiles. The
// units come from the reader's CB_UNITS pages, in claim order, up to the page holding UNITS. The caller has run
// compute_kernel_hw_startup; the unpack and pack formats are set here.
//
// Compile-time args from CT: 0 H, 1 RT, 2 DT, 3 NBPT, 4 NX, 5 NY, 6 UNITS, 7 NCORES, 8 NBGT.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "block_pool_units.hpp"

namespace block_pool {
namespace compute {
constexpr uint32_t cb_x = 3, cb_ps = 4, cb_zero = 5, cb_out = 30;
constexpr uint32_t ROWS_PER_BATCH = 4;
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace compute

template <uint32_t CT>
void pool() {
    using namespace compute;
    constexpr uint32_t H = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t RT = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t DT = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t UNITS = get_compile_time_arg_val(CT + 6);
    static_assert(DT <= 4, "DT output tiles must fit the fp32 DST");
    using Layout = block_pool::Layout<H, RT, NBPT>;
    static_assert(Layout::UNITS == UNITS, "host unit count must match the kernel layout");
    CircularBuffer cbx(cb_x), cbps(cb_ps), cbz(cb_zero), cbo(cb_out), cbu(CB_UNITS);
    auto next_unit = [&]() {
        cbu.wait_front(1);
        const uint32_t u = cbu.read_tile_value(0, 0);
        cbu.pop_front(1);
        return u;
    };

    reconfig_data_format<SrcOrder::Reverse>(cb_ps, cb_x);
    pack_reconfig_data_format(cb_out);

    bool zero_ready = false;
    for (uint32_t u = next_unit(); u < UNITS; u = next_unit()) {
        const uint32_t g = Layout::decode(u).g;
        const uint32_t k_begin = g * BLOCK;
        const uint32_t k_end = min_u(k_begin + BLOCK, RT);

        if (k_end <= k_begin && !zero_ready) {
            cbz.wait_front(1);
            zero_ready = true;
        }
        tile_regs_acquire();
        if (k_end <= k_begin) {
            copy_init(cb_zero);
            for (uint32_t n = 0; n < DT; ++n) {
                copy_tile(cb_zero, 0, n);
            }
        } else {
            matmul_init(cb_ps, cb_x);
            for (uint32_t k0 = k_begin; k0 < k_end; k0 += ROWS_PER_BATCH) {
                const uint32_t nk = min_u(ROWS_PER_BATCH, k_end - k0);
                cbx.wait_front(ROWS_PER_BATCH * DT);
                cbps.wait_front(ROWS_PER_BATCH);
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    for (uint32_t n = 0; n < DT; ++n) {
                        matmul_tiles(cb_ps, cb_x, kk, kk * DT + n, n);
                    }
                }
                cbps.pop_front(ROWS_PER_BATCH);
                cbx.pop_front(ROWS_PER_BATCH * DT);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cbo.reserve_back(DT);
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile<true>(n, cb_out, n);
        }
        cbo.push_back(DT);
        tile_regs_release();
    }
}

}  // namespace block_pool

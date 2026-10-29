// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-core decode boundary, helper reader (NCRISC) (fused_decode/boundary.py DecodeBoundary with helpers): helper
// h owns pages [page0, page0 + PH) of the row-tile flat buffers (multi-row boundaries, e.g. the DFlash verify block).
// Feeds its compute: the RMSNorm weight pages, the reduce scaler and row-sum matrix P, its residual pages (backed by
// its shard of the residual tensor, or built from the embedding rows / zero), its pages of every device's partial
// (the fabric sender delivers page range h of every source into this helper's receive slots; wait for `expected`
// increments of the receive semaphore), then the other helpers' row sums of squares (program semaphore XCHG_SEM: the
// helper writers copy their tile into slot h of every helper's cb_xchg).
//
// runtime args: [gamma_addr, sem_addr, res_src_addr, page0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "bnd_tiles.hpp"

void kernel_main() {
    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_src_addr = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t PH = get_compile_time_arg_val(4);  // pages per helper
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t expected = get_compile_time_arg_val(6);
    constexpr uint32_t res_mode = get_compile_time_arg_val(7);   // 0 residual tensor, 1 embedding rows, 2 zero
    constexpr uint32_t row_bytes = get_compile_time_arg_val(8);  // hidden * 2
    constexpr uint32_t rows = get_compile_time_arg_val(9);
    constexpr uint32_t cb_p = get_compile_time_arg_val(10);
    constexpr uint32_t cb_xchg = get_compile_time_arg_val(11);
    constexpr uint32_t helpers = get_compile_time_arg_val(12);
    constexpr uint32_t xchg_sem = get_compile_time_arg_val(13);
    constexpr auto gamma_args = TensorAccessorArgs<14>();
    constexpr auto res_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t page = 2048;
    cb_reserve_back(cb_gamma, PH);
    {
        const auto s_gamma = TensorAccessor(gamma_args, gamma_addr, page);
        for (uint32_t t = 0; t < PH; ++t) {
            noc_async_read(s_gamma.get_noc_addr(page0 + t), get_write_ptr(cb_gamma) + t * page, page);
        }
    }
    cb_reserve_back(cb_res, PH);
    if constexpr (res_mode >= 1) {
        const uint32_t dst = get_write_ptr(cb_res);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t z = 0; z < PH * page; z += MEM_ZEROS_SIZE) {
            const uint32_t n = PH * page - z < MEM_ZEROS_SIZE ? PH * page - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst + z, n);
        }
        noc_async_read_barrier();
        if constexpr (res_mode == 1) {
            // embedding row r (page r of the [1, 1, rows, hidden] row-major input): values 32 j .. 32 j + 31 -> row r
            // of row tile j (rows * 64 bytes per row tile; this helper's tiles are those in its pages)
            const auto s_res = TensorAccessor(res_args, res_src_addr, row_bytes);
            constexpr uint32_t tb = rows * 64;
            const uint32_t j0 = page0 * page / tb;
            uint32_t j1 = (page0 + PH) * page / tb;
            if (j1 > row_bytes / 64) {
                j1 = row_bytes / 64;
            }
            for (uint32_t r = 0; r < rows; ++r) {
                const uint64_t src = s_res.get_noc_addr(r);
                for (uint32_t j = j0; j < j1; ++j) {
                    const uint32_t t = dst + (j - j0) * tb;
                    noc_async_read(src + j * 64, t + r * 32, 32);
                    noc_async_read(src + j * 64 + 32, t + rows * 32 + r * 32, 32);
                }
            }
        }
    }
    // reduce scaler (1.0 in the first row of each face) and P[R][R'] = 1 iff R = R' (mod rows)
    cb_reserve_back(cb_scaler, 1);
    bnd_tiles::scaler(get_write_ptr(cb_scaler));
    cb_push_back(cb_scaler, 1);
    cb_reserve_back(cb_p, 1);
    bnd_tiles::row_sum_matrix<rows>(get_write_ptr(cb_p));
    cb_push_back(cb_p, 1);
    if constexpr (res_mode >= 1) {
        noc_async_read_barrier();
    }
    cb_push_back(cb_res, PH);
    if constexpr (slots > 0) {
        {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            noc_semaphore_wait_min(sem, expected);
            noc_semaphore_set(sem, 0);
        }
        cb_reserve_back(cb_recv, slots * PH);
        cb_push_back(cb_recv, slots * PH);
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, PH);
    // the helpers' row sums of squares (written into slot h of this cb by helper h's writer)
    cb_reserve_back(cb_xchg, helpers);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xchg_sem)), helpers);
    cb_push_back(cb_xchg, helpers);
}

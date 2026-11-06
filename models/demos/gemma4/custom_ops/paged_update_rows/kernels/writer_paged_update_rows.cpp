// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer of PagedUpdateRows: per unit the job table and the untilized source rows; per job the untilized cache
// tile row (CB_UNT_A) gets the job's rows patched in place (the stock's NoC loopback copy of one WBYTES row per
// position, from the untilized source block), is handed to the compute as CB_UNT_B (the same L1 region: both
// rings advance by Wt pages per job, so A's front is B's back) and the re-tilized row (CB_OUT) is written back to
// its cache tiles; a null job is popped without a write. Ends with a write barrier.
// runtime args: [cache_addr, n_units]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"
#include "paged_update_rows_common.hpp"

void kernel_main() {
    using namespace pur;
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_units = get_arg_val<uint32_t>(1);
    constexpr auto src_args = TensorAccessorArgs<CT_END>();
    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto cache_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto s_cache = TensorAccessor(cache_args, cache_addr, CACHE_TILE);

    CircularBuffer cb_unt_src(CB_UNT_SRC), cb_a(CB_UNT_A), cb_b(CB_UNT_B), cb_out(CB_OUT), cb_jobs(CB_JOBS);
    const uint32_t nx = my_x[noc_index], ny = my_y[noc_index];

    for (uint32_t u = 0; u < n_units; ++u) {
        cb_jobs.wait_front(1);
        volatile tt_l1_ptr uint32_t* jobs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_jobs.get_read_ptr());
        cb_unt_src.wait_front(N_SRC * Wt);
        const uint32_t src_l1 = cb_unt_src.get_read_ptr();  // row r of the unit's rows at r * WBYTES
        for (uint32_t j = 0; j < J; ++j) {
            cb_a.wait_front(Wt);
            cb_b.reserve_back(Wt);
            const uint32_t blk = cb_a.get_read_ptr();
            ASSERT(blk == cb_b.get_write_ptr());  // the aliased rings stay in lockstep
            for (uint32_t p = 0; p < P; ++p) {
                volatile tt_l1_ptr uint32_t* e = jobs + 2 * J + 3 * p;
                if (e[0] == j) {
                    noc_async_read(get_noc_addr(nx, ny, src_l1 + e[2] * WBYTES), blk + e[1] * WBYTES, WBYTES);
                }
            }
            noc_async_read_barrier();
            cb_b.push_back(Wt);
            cb_a.pop_front(Wt);

            cb_out.wait_front(Wt);
            if (jobs[2 * j]) {
                const uint32_t out_l1 = cb_out.get_read_ptr();
                const uint32_t tile0 = jobs[2 * j + 1];
                for (uint32_t w = 0; w < Wt; ++w) {
                    noc_async_write(out_l1 + w * CACHE_TILE, s_cache.get_noc_addr(tile0 + w), CACHE_TILE);
                }
                noc_async_write_barrier();
            }
            cb_out.pop_front(Wt);
        }
        cb_unt_src.pop_front(N_SRC * Wt);
        cb_jobs.pop_front(1);
    }
    noc_async_write_barrier();
}

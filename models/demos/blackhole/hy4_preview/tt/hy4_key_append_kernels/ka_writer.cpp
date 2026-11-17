// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B32 key append, writer: per row wait for the re-tilized block (cb_out), write its 4 tiles to the pool unless the
// row is skipped, barrier, signal cb_done to the reader. CT: 0 cb_out, 1 cb_done, 2 cb_meta, 3 S, 4 TB_C, then
// TensorAccessorArgs(pool). Common: pool addr.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_done = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t TB_C = get_compile_time_arg_val(4);
    constexpr auto pool_args = TensorAccessorArgs<5>();
    constexpr uint32_t WT = 4;
    const auto pool_acc = TensorAccessor(pool_args, get_common_arg_val<uint32_t>(0), TB_C);
    Noc noc;
    CircularBuffer out(cb_out), done(cb_done), meta(cb_meta);
    for (uint32_t j = 0; j < S; ++j) {
        meta.wait_front(1);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta.get_read_ptr());
        const bool skip = mp[0] != 0;
        const uint32_t tile_base = mp[1];
        out.wait_front(WT);
        if (!skip) {
            for (uint32_t t = 0; t < WT; ++t) {
                noc.async_write(
                    out, pool_acc, TB_C, {.offset_bytes = t * TB_C}, {.page_id = tile_base + t, .offset_bytes = 0});
            }
            noc.async_write_barrier();
        }
        out.pop_front(WT);
        meta.pop_front(1);
        done.reserve_back(1);
        done.push_back(1);
    }
}

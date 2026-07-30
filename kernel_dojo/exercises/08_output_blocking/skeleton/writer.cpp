// SPDX-License-Identifier: Apache-2.0
//
// Exercise 08 — 2-D blocked matmul writer (provided).
//
// The compute kernel emits output in block order: for each row-block, for each
// column nt, Mb tiles running down the rows. Those Mb tiles are Nt apart in C,
// so this writer scatters rather than writing a contiguous run.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t Nt = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t n_blocks = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Mb = get_compile_time_arg_val(1);

    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const uint32_t end_block = start_block + n_blocks;

    for (uint32_t blk = start_block; blk < end_block; blk++) {
        const uint32_t row0 = blk * Mb;

        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_out, Mb);
            const uint32_t base = get_read_ptr(cb_out);

            for (uint32_t m = 0; m < Mb; m++) {
                // C tile (row0 + m, nt) is at page (row0 + m) * Nt + nt.
                noc_async_write_page((row0 + m) * Nt + nt, dst, base + m * tile_bytes);
            }
            noc_async_write_barrier();

            cb_pop_front(cb_out, Mb);
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//
// Exercise 08 — 2-D blocked matmul reader, reference solution.
//
// Works a row-BLOCK at a time: Mb rows of A are held resident together, so
// each B column read feeds Mb output tiles instead of one. That cuts B's DRAM
// traffic by a factor of Mb.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t start_block = get_arg_val<uint32_t>(4);
    const uint32_t n_blocks = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);

    constexpr auto a_args = TensorAccessorArgs<3>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t end_block = start_block + n_blocks;

    for (uint32_t blk = start_block; blk < end_block; blk++) {
        const uint32_t row0 = blk * Mb;

        // The whole Mb x Kt sub-block of A, read once and reused across all Nt
        // columns. Laid out row-major so the compute kernel can address tile
        // (m, kt) at window index m * Kt + kt.
        cb_reserve_back(cb_a, Mb * Kt);
        const uint32_t base_a = get_write_ptr(cb_a);
        for (uint32_t m = 0; m < Mb; m++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                noc_async_read_page(
                    (row0 + m) * Kt + kt, a, base_a + (m * Kt + kt) * tile_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, Mb * Kt);

        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_reserve_back(cb_b, Kt);
            const uint32_t base_b = get_write_ptr(cb_b);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                noc_async_read_page(kt * Nt + nt, b, base_b + kt * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_b, Kt);
        }
    }
}

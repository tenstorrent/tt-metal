// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t value_columns = get_compile_time_arg_val(0);
    static_assert(value_columns == 1 || value_columns == 2 || value_columns == 4);
    constexpr uint32_t splits = 4 / value_columns;
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(0), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1), 512);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    // CB10 is writer-private scratch, not shared with the reader/compute.
    const uint32_t scratch = get_write_ptr(10);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t work = first + item * stride;
        const uint32_t head = work / splits;
        const uint32_t first_column = (work % splits) * value_columns;
        cb_wait_front(7, 4 * value_columns);
        // Compute and NoC only read S_new now. Start the DRAM write while
        // compute performs q^T S_new, but retain CB7 until CB8 proves that
        // reduction is done and the NoC barrier proves the writes are done.
        const uint32_t state_l1 = get_read_ptr(7);
        for (uint32_t kr = 0; kr < 4; ++kr) {
            for (uint32_t vc = 0; vc < value_columns; ++vc) {
                noc_async_write_tile(
                    head * 16 + kr * 4 + first_column + vc, state, state_l1 + (kr * value_columns + vc) * 4096);
            }
        }
        cb_wait_front(8, value_columns);
        const auto* tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(8));
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < value_columns * 32; ++i) {
            row[i] = tiles[(i / 32) * 1024 + ((i % 32) / 16) * 256 + i % 16];
        }
        // Each partition owns a disjoint, 128-byte-aligned portion of the
        // same 512-byte row. No read/modify/write or cross-core reduction.
        noc_async_write(scratch, output.get_noc_addr(head, first_column * 128), value_columns * 128);
        noc_async_write_barrier();
        cb_pop_front(7, 4 * value_columns);
        cb_pop_front(8, value_columns);
    }
}

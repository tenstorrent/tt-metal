// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 (or R-row) decode all-reduce, step 3 writer (Laguna): core c's per output tiles = zeros with rows 0..R-1
// set from the summed flat block (row r, tile i: elements (r * per + i) * 32 .. + 15 -> face (r / 16) * 2 row r % 16,
// the next 16 -> the face after it), written whole to the output ([32, H] bf16 TILE, any layout the accessor
// describes).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t per = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t stage = get_write_ptr(17);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < per * 2048; off += MEM_ZEROS_SIZE) {
        const uint32_t n = per * 2048 - off < MEM_ZEROS_SIZE ? per * 2048 - off : MEM_ZEROS_SIZE;
        noc_async_read(zeros, stage + off, n);
    }
    noc_async_read_barrier();
    cb_wait_front(16, 1);
    const uint32_t sum = get_read_ptr(16);
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t off = ((r / 16) * 2) * 512 + (r % 16) * 32;
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t src = sum + (r * per + i) * 64;
            noc_async_read(get_noc_addr(src), stage + i * 2048 + off, 32);
            noc_async_read(get_noc_addr(src + 32), stage + i * 2048 + off + 512, 32);
        }
    }
    noc_async_read_barrier();
    cb_pop_front(16, 1);
    for (uint32_t i = 0; i < per; ++i) {
        noc_async_write(stage + i * 2048, o.get_noc_addr(c * per + i), 2048);
    }
    noc_async_write_barrier();
}

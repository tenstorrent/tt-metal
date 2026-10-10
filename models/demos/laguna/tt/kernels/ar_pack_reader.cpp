// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 (or R-row) decode all-reduce, step 1 (Laguna): rows 0..R-1 of a [32, H] bf16 TILE tensor -> [R, H]
// row-major rows. Row r of a 32x32 tile is 16 bf16 of face f = (r / 16) * 2 at byte f * 512 + (r % 16) * 32
// followed by 16 of face f + 1 (512 bytes later). Core c packs tiles [c * per, (c + 1) * per) of each row into
// 64-byte pieces of a local stage and writes them to output row r at byte c * per * 64.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t per = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1));
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t stage = get_write_ptr(0);
    // aligned 64-byte reads into tmp (a DRAM source needs a destination with the same 64-byte alignment), then
    // local 32-byte copies of the two pieces of each row
    const uint32_t tmp = get_write_ptr(1);
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t off = ((r / 16) * 2) * 512 + (r % 16) * 32;
        for (uint32_t i = 0; i < per; ++i) {
            const uint64_t a = x.get_noc_addr(c * per + i) + (off & ~63u);
            noc_async_read(a, tmp + (r * per + i) * 128, 64);
            noc_async_read(a + 512, tmp + (r * per + i) * 128 + 64, 64);
        }
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t sub = (((r / 16) * 2) * 512 + (r % 16) * 32) & 63u;
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t t = tmp + (r * per + i) * 128 + sub;
            noc_async_read(get_noc_addr(t), stage + (r * per + i) * 64, 32);
            noc_async_read(get_noc_addr(t + 64), stage + (r * per + i) * 64 + 32, 32);
        }
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < R; ++r) {
        noc_async_write(stage + r * per * 64, o.get_noc_addr(r) + c * per * 64, per * 64);
    }
    noc_async_write_barrier();
}

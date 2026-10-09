// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode all-reduce, step 3 reader (Laguna): core c's 32 * per columns of the D partial rows and of row 0 of
// the residual ([32, H] bf16 TILE), each laid out contiguously at the start of its own CB slot, so the compute can add
// them as flat element blocks. The partial rows are either one gathered [1, D, 1, H] row-major tensor (page p =
// partial p; split = 0) or D separate [1, 1, 1, H] tensors from all_broadcast (split = 1; addresses in the common
// runtime args after the residual's).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t per = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t D = get_compile_time_arg_val(2);
    constexpr uint32_t split = get_compile_time_arg_val(3);
    constexpr uint32_t slot = 2048;
    constexpr auto g_args = TensorAccessorArgs<4>();
    constexpr auto r_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0));
    const auto r = TensorAccessor(r_args, get_common_arg_val<uint32_t>(1), 2048);
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    cb_reserve_back(0, D + 1);
    const uint32_t base = get_write_ptr(0);
    for (uint32_t p = 0; p < D; ++p) {
        uint64_t src;
        if constexpr (split) {
            src = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2 + p)).get_noc_addr(0);
        } else {
            src = g.get_noc_addr(p);
        }
        noc_async_read(src + c * per * 64, base + p * slot, per * 64);
    }
    // residual row 0: aligned 64-byte reads into tmp (a DRAM residual needs a destination with the same 64-byte
    // alignment), then local 32-byte copies into the residual's slot
    const uint32_t tmp = get_write_ptr(1);
    for (uint32_t i = 0; i < per; ++i) {
        const uint64_t a = r.get_noc_addr(c * per + i);
        noc_async_read(a, tmp + i * 128, 64);
        noc_async_read(a + 512, tmp + i * 128 + 64, 64);
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < per; ++i) {
        noc_async_read(get_noc_addr(tmp + i * 128), base + D * slot + i * 64, 32);
        noc_async_read(get_noc_addr(tmp + i * 128 + 64), base + D * slot + i * 64 + 32, 32);
    }
    noc_async_read_barrier();
    cb_push_back(0, D + 1);
}

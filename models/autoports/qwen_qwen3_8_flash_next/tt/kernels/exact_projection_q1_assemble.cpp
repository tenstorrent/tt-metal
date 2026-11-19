// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto scratch = TensorAccessor(sa, get_arg_val<uint32_t>(0), 128);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1), 4096);
    const uint32_t first = get_arg_val<uint32_t>(2), stride = get_arg_val<uint32_t>(3);
    const uint32_t tiles = get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    const uint32_t sp = get_write_ptr(0), op = get_write_ptr(1);
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(sp);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(op);
    for (uint32_t tile = first; tile < tiles; tile += stride) {
        for (uint32_t i = 0; i < 1024; ++i) {
            dst[i] = 0;
        }
        for (uint32_t g = 0; g < 32 / rows; ++g) {
            noc_async_read(scratch.get_noc_addr(tile * (32 / rows) + g), sp, 128);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t c = g * rows + r;
                dst[(c / 16) * 256 + c % 16] = src[r];
            }
        }
        noc_async_write(op, output.get_noc_addr(tile), 4096);
        noc_async_write_barrier();
    }
}

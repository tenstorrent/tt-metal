// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// pool-prefill: in-chunk causal staircase of the indexer scores, in place. score RM [1,1,R,N] bf16 (page = query row i,
// N columns = this column's local keys, the LAST 64 hold this chunk's own slice: tokens s + 64c + j, j < 64). Query row
// i sees local key j iff 64 c + j <= i  ->  columns with 64 c + j > i get -inf. colid: uint32 page (element 0 = column
// c). N % 64 == 0.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);  // row width (bucket)
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_s = get_common_arg_val<uint32_t>(0), a_c = get_common_arg_val<uint32_t>(1),
                   n_local = get_common_arg_val<uint32_t>(2);
    Noc noc;
    const auto s_acc = TensorAccessor(s_args, a_s, N * 2);
    const auto c_acc = TensorAccessor(c_args, a_c, 64);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    noc.async_read(c_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t c = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    constexpr uint32_t TAIL = 128;
    const uint32_t OFF = (n_local - 64) * 2;
    for (uint32_t i = core; i < R; i += NC) {
        noc.async_read(s_acc, cb, TAIL, {.page_id = i, .offset_bytes = OFF}, {.offset_bytes = 64});
        noc.async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 64);
        // first masked j: 64 c + j > i  ->  j > i - 64 c
        const int32_t lim = (int32_t)i - (int32_t)(64 * c);  // last visible j (may be < 0 -> all masked, >= 63 -> none)
        for (int32_t j = 0; j < 64; ++j) {
            if (j > lim) {
                t[j] = 0xFF80u;
            }
        }
        noc.async_write(cb, s_acc, TAIL, {.offset_bytes = 64}, {.page_id = i, .offset_bytes = OFF});
        noc.async_write_barrier();
    }
}

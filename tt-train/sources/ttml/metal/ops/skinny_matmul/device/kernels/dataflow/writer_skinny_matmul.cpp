// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for both skinny_matmul phases.
//   phase A: per row of the item, Nt fp32 partial tiles to pages r*S*Nt + s*Nt + n
//   phase B (REDUCE): one bf16 output tile to page r*Nt + n

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/skinny_matmul_cbs.hpp"

namespace cb = skinny_matmul_cb;

constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t S = get_compile_time_arg_val(4);
constexpr uint32_t St = get_compile_time_arg_val(5);

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto out_args = TensorAccessorArgs<7>();
    const auto out_acc = TensorAccessor(out_args, out_addr);

#ifdef REDUCE
    const uint32_t tile_bytes = get_tile_size(cb::out);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        cb_wait_front(cb::out, 1);
        noc_async_write_page(work, out_acc, get_read_ptr(cb::out));  // work == r*Nt + n
        noc_async_write_barrier();
        cb_pop_front(cb::out, 1);
    }
#else
    const uint32_t tile_bytes = get_tile_size(cb::partial_out);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t g = work / S;
        const uint32_t s = work - g * S;
        const uint32_t r0 = g * R;
        const uint32_t nr = (r0 + R <= rows) ? R : (rows - r0);
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            cb_wait_front(cb::partial_out, Nt);
            uint32_t l1 = get_read_ptr(cb::partial_out);
            for (uint32_t n = 0; n < Nt; ++n) {
                noc_async_write_page(r * S * Nt + s * Nt + n, out_acc, l1);
                l1 += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb::partial_out, Nt);
        }
    }
#endif
}

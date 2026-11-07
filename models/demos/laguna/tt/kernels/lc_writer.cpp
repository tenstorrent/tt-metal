// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Local-only combine (Laguna), writer: per unit, row r < valid of the untilized tile row goes to page
// token * K + slot of the [T, K, H] bf16 row-major output (token, slot from the row's dispatch metadata).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t meta_page = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wm = 2, cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), row_bytes);
    while (true) {
        cb_wait_front(cb_wm, 1);
        invalidate_l1_cache();  // the reader's NoC reads landed the metadata behind this RISC's L1 cache
        const uint32_t wm = get_read_ptr(cb_wm);
        const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wm)[0];
        if (valid == 0xFFFFFFFFu) {
            cb_pop_front(cb_wm, 1);
            break;
        }
        cb_wait_front(cb_out, 32);
        const uint32_t rows = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < valid; ++r) {
            volatile tt_l1_ptr int32_t* m =
                reinterpret_cast<volatile tt_l1_ptr int32_t*>(((wm + 32 + 63) & ~63u) + r * meta_page);
            noc_async_write(rows + r * row_bytes, out.get_noc_addr(m[1] * K + m[2]), row_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 32);
        cb_pop_front(cb_wm, 1);
    }
}

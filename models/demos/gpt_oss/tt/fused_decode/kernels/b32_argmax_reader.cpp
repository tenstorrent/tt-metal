// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 LM-head argmax, reader (BRISC): this core's logits tiles (tile row rt, columns col0 .. col0 + ncol - 1) into
// cb_in, B tiles per NOC batch (cb_in holds 2 B: a batch never straddles the CB's end).
//
// runtime args: [rt, col0, ncol]; common runtime args: [logits]; compile-time args: [cb_in, NT, B, accessor (logits)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t rt = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t ncol = get_arg_val<uint32_t>(2);
    const uint32_t logits_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);  // tile columns per tile row
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr auto l_args = TensorAccessorArgs<3>();
    const uint32_t tile = get_tile_size(cb_in);
    const auto s_l = TensorAccessor(l_args, logits_addr, tile);
    for (uint32_t k = 0; k < ncol; k += B) {
        const uint32_t n = ncol - k < B ? ncol - k : B;
        cb_reserve_back(cb_in, n);
        const uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(s_l.get_noc_addr(rt * NT + col0 + k + i), l1 + i * tile, tile);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            cb_push_back(cb_in, 1);
        }
    }
}

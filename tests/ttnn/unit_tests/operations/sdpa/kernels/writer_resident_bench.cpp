// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Compute-throughput benchmark writer: drains every output row; only the last Q block is written.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t q_repeats = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(2);
    constexpr auto oa = TensorAccessorArgs<3>();
    auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t q = 0; q < q_repeats; ++q) {
        for (uint32_t row = 0; row < q_tiles; ++row) {
            cb.wait_front(d_tiles);
            if (q == q_repeats - 1) {
                for (uint32_t i = 0; i < d_tiles; ++i) {
                    noc.async_write(cb, out, 2048, {.offset_bytes = i * 2048}, {.page_id = row * d_tiles + i});
                }
                noc.async_write_barrier();
            }
            cb.pop_front(d_tiles);
        }
    }
}

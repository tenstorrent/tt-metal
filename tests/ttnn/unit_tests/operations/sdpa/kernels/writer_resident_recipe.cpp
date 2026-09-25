// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t q_repeats = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer cb(16);
    // Drain row by row; only the last Q block is transferred for correctness.
    for (uint32_t q = 0; q < q_repeats; ++q) {
        for (uint32_t row = 0; row < 8; ++row) {
            cb.wait_front(4);
            if (q == q_repeats - 1) {
                for (uint32_t i = 0; i < 4; ++i) {
                    noc.async_write(cb, out, 2048, {.offset_bytes = i * 2048}, {.page_id = row * 4 + i});
                }
                noc.async_write_barrier();
            }
            cb.pop_front(4);
        }
    }
}

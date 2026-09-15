// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    const uint32_t first_job = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t job = first_job; job < first_job + jobs; ++job) {
        for (uint32_t row = 0; row < q_tiles; ++row) {
            cb.wait_front(4);
            for (uint32_t col = 0; col < 4; ++col) {
                noc.async_write(
                    cb, out, 2048, {.offset_bytes = col * 2048}, {.page_id = job * q_tiles * 4 + row * 4 + col});
            }
            noc.async_write_barrier();
            cb.pop_front(4);
        }
    }
}

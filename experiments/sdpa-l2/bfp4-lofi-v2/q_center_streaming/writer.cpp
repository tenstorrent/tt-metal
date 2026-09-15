// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t row = 0; row < 4; ++row) {
        cb.wait_front(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_write(cb, out, 2048, {.offset_bytes = j * 2048}, {.page_id = row * 4 + j});
        }
        noc.async_write_barrier();
        cb.pop_front(4);
    }
}

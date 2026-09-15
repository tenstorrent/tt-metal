// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr auto ta = TensorAccessorArgs<1>();
    auto input = TensorAccessor(ta, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer cb(0);
    cb.reserve_back(n);
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(input, cb, 4096, {.page_id = i}, {.offset_bytes = i * 4096});
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

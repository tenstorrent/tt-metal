// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr auto ta = TensorAccessorArgs<1>();
    auto output = TensorAccessor(ta, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer cb(16);
    cb.wait_front(n);
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_write(cb, output, 2048, {.offset_bytes = i * 2048}, {.page_id = i});
    }
    noc.async_write_barrier();
    cb.pop_front(n);
}

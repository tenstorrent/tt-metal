// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_tile_size(16);
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t i = 0; i < count; i += 4) {
        cb.wait_front(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_write(cb, output, bytes, {.offset_bytes = j * bytes}, {.page_id = start + i + j});
        }
        noc.async_write_barrier();
        cb.pop_front(4);
    }
}

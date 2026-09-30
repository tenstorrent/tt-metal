// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto output = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0));
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer buffer(1);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        buffer.wait_front(1);
        noc.async_write(buffer, output, 4096, {}, {.page_id = tile});
        noc.async_write_barrier();
        buffer.pop_front(1);
    }
}

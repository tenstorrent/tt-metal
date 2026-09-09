// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t seed_cb = get_compile_time_arg_val(0);
    constexpr auto seed_args = TensorAccessorArgs<1>();

    const uint32_t seed_addr = get_arg_val<uint32_t>(0);
    const auto seed_accessor = TensorAccessor(seed_args, seed_addr);
    CircularBuffer cb(seed_cb);
    Noc noc;

    cb.reserve_back(1);
    noc.async_read(seed_accessor, cb, get_tile_size(seed_cb), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb.push_back(1);
}

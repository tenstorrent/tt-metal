// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t granularity = get_compile_time_arg_val(1);
    constexpr bool do_io = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();

    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t num_events = get_arg_val<uint32_t>(2);

    const auto s = TensorAccessor(args, addr);
    const uint32_t bytes = get_local_cb_interface(cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb(cb_id);
    for (uint32_t e = 0; e < num_events; ++e) {
        cb.wait_front(granularity);
        if constexpr (do_io) {
            for (uint32_t k = 0; k < granularity; ++k) {
                noc.async_write(cb, s, bytes, {.offset_bytes = k * bytes}, {.page_id = start + e * granularity + k});
            }
            noc.async_write_barrier();
        }
        cb.pop_front(granularity);
    }
}

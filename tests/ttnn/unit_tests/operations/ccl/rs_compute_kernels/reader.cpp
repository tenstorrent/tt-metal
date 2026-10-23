// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

// Feeds a reduce_scatter reduction kernel one granularity-sized chunk per event: c_0 and c_1 on every event,
// c_2 on the events whose flag bit is set.
void kernel_main() {
    constexpr uint32_t granularity = get_compile_time_arg_val(0);
    constexpr bool do_io = get_compile_time_arg_val(1);
    constexpr auto args0 = TensorAccessorArgs<2>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();

    const uint32_t addr0 = get_arg_val<uint32_t>(0);
    const uint32_t addr1 = get_arg_val<uint32_t>(1);
    const uint32_t addr2 = get_arg_val<uint32_t>(2);
    const uint32_t start0 = get_arg_val<uint32_t>(3);
    const uint32_t start1 = get_arg_val<uint32_t>(4);
    const uint32_t start2 = get_arg_val<uint32_t>(5);
    const uint32_t num_events = get_arg_val<uint32_t>(6);

    const auto s0 = TensorAccessor(args0, addr0);
    const auto s1 = TensorAccessor(args1, addr1);
    const auto s2 = TensorAccessor(args2, addr2);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb0(0), cb1(1), cb2(2);
    uint32_t chunk2 = 0;
    for (uint32_t e = 0; e < num_events; ++e) {
        const bool use2 = (get_arg_val<uint32_t>(7 + e / 32) >> (e % 32)) & 1;
        cb0.reserve_back(granularity);
        cb1.reserve_back(granularity);
        if (use2) {
            cb2.reserve_back(granularity);
        }
        if constexpr (do_io) {
            for (uint32_t k = 0; k < granularity; ++k) {
                noc.async_read(s0, cb0, bytes, {.page_id = start0 + e * granularity + k}, {.offset_bytes = k * bytes});
                noc.async_read(s1, cb1, bytes, {.page_id = start1 + e * granularity + k}, {.offset_bytes = k * bytes});
                if (use2) {
                    noc.async_read(
                        s2, cb2, bytes, {.page_id = start2 + chunk2 * granularity + k}, {.offset_bytes = k * bytes});
                }
            }
            noc.async_read_barrier();
        }
        cb0.push_back(granularity);
        cb1.push_back(granularity);
        if (use2) {
            cb2.push_back(granularity);
            ++chunk2;
        }
    }
}

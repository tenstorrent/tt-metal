// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t pages = get_compile_time_arg_val(1);
    constexpr uint32_t depth = get_compile_time_arg_val(2);
    constexpr uint32_t bytes = 1088;
    static_assert(mode <= 2 && pages >= 1 && pages <= 15 && depth >= 1 && depth <= 8);
    constexpr auto source_args = TensorAccessorArgs<3>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), bytes);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t blocks = (count + pages - 1) / pages;
    const uint64_t bank_base = source.get_noc_addr(first);
    if constexpr (mode == 2) {
        noc_async_read_one_packet_set_state<true>(bank_base, pages * bytes, vc);
    }
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t slot = block % depth;
        const uint32_t trid = slot + 1;
        // Each in-flight packet owns a separate CB. Complete and publish the
        // old owner before waiting for its consumer and recycling this slot.
        if (block >= depth) {
            noc_async_read_barrier_with_trid(trid);
            cb_push_back(slot, pages);
        }
        cb_reserve_back(slot, pages);
        const uint32_t dest = get_write_ptr(slot);
        const uint32_t offset = block * pages;
        const uint32_t valid = count - offset < pages ? count - offset : pages;
        noc_async_read_set_trid(trid);
        if constexpr (mode == 2) {
            if (valid != pages) {
                noc_async_read_one_packet_set_state<true>(bank_base, valid * bytes, vc);
            }
            noc_async_read_one_packet_with_state_with_trid(
                static_cast<uint32_t>(bank_base), offset * bytes, dest, trid);
        } else {
            for (uint32_t page = 0; page < valid; ++page) {
                const uint64_t address = source.get_noc_addr(first + (offset + page) * stride);
                noc_async_read_one_packet_set_state<true>(address, bytes, vc);
                noc_async_read_one_packet_with_state_with_trid(
                    static_cast<uint32_t>(address), 0, dest + page * bytes, trid);
            }
        }
    }
    // Drain every outstanding tag in consumer order, including short tails.
    const uint32_t pending = blocks < depth ? 0 : blocks - depth;
    for (uint32_t block = pending; block < blocks; ++block) {
        const uint32_t slot = block % depth;
        noc_async_read_barrier_with_trid(slot + 1);
        cb_push_back(slot, pages);
    }
    noc_async_read_set_trid(0);
}

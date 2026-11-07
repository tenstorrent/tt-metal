// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Global-CB receiver (runs on sub-device 1). Waits for each page in the remote CB, then either copies it to
// page out_start + i of the output tensor (mode 0, byte-exact check on the host) or checks its tag word
// (mode 1: word 0 must be tag_base + i % ring), then pops it. In mode 1 the last `ring` pages are also copied
// out at the end so the host can check them byte for byte. Stats page `stats_page`:
// [t_start, t_end, pages, tag mismatches, t_first_page].
#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_start = get_arg_val<uint32_t>(2);
    const uint32_t stats_addr = get_arg_val<uint32_t>(3);
    const uint32_t stats_page = get_arg_val<uint32_t>(4);
    const uint32_t tag_base = get_arg_val<uint32_t>(5);
    constexpr uint32_t remote_cb_id = get_named_compile_time_arg_val("cb_remote");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr uint32_t ring = get_named_compile_time_arg_val("ring");
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto st = TensorAccessor(st_args, stats_addr);

    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};
    Noc noc;
    remote_cb.set_sender_page_size(noc, page_size);

    uint32_t bad = 0;
    uint32_t slot = 0;
    const uint64_t t_start = get_timestamp();
    uint64_t t_first = t_start;
    for (uint32_t i = 0; i < num_pages; ++i) {
        remote_cb.wait_front(1);
        if (i == 0) {
            t_first = get_timestamp();
        }
        const uint32_t rd = get_remote_receiver_cb_interface(remote_cb_id).fifo_rd_ptr;
        if constexpr (mode == 0) {
            noc_async_write(rd, out.get_noc_addr(out_start + i), page_size);
            noc_async_write_barrier();
        } else {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd) != tag_base + slot) {
                bad++;
            }
            slot = (slot + 1 == ring) ? 0 : slot + 1;
            if (i + ring >= num_pages) {
                noc_async_write(rd, out.get_noc_addr(out_start + i + ring - num_pages), page_size);
                noc_async_write_barrier();
            }
        }
        remote_cb.pop_front(noc, 1);
    }
    const uint64_t t_end = get_timestamp();
    remote_cb.commit();
    noc_async_atomic_barrier();

    cb_reserve_back(cb_stats, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = 0;
    }
    s[0] = static_cast<uint32_t>(t_start);
    s[1] = static_cast<uint32_t>(t_start >> 32);
    s[2] = static_cast<uint32_t>(t_end);
    s[3] = static_cast<uint32_t>(t_end >> 32);
    s[4] = num_pages;
    s[5] = bad;
    s[6] = static_cast<uint32_t>(t_first);
    s[7] = static_cast<uint32_t>(t_first >> 32);
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(stats_page), 64);
    noc_async_write_barrier();
}

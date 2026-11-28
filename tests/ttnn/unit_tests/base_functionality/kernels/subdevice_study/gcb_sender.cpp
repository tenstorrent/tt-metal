// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Global-CB sender (runs on sub-device 0). mode 0 (stream): every page comes from the DRAM reader through
// cb_src and is pushed into the remote CB. mode 1 (repeat): the reader loads `ring` pages once and this
// kernel pushes them round-robin, so the timed loop moves only L1 -> L1 over the NoC. Ends with a barrier on
// the receiver's acks. Stats page `stats_page`: [t_start, t_end, pages].
#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t stats_page = get_arg_val<uint32_t>(2);
    constexpr uint32_t remote_cb_id = get_named_compile_time_arg_val("cb_remote");
    constexpr uint32_t cb_src = get_named_compile_time_arg_val("cb_src");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr uint32_t ring = get_named_compile_time_arg_val("ring");
    constexpr auto st_args = TensorAccessorArgs<0>();
    const auto st = TensorAccessor(st_args, stats_addr);

    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};
    Noc noc;
    remote_cb.set_receiver_page_size(noc, page_size);

    uint64_t t_start;
    if constexpr (mode == 0) {
        t_start = get_timestamp();
        for (uint32_t i = 0; i < num_pages; ++i) {
            cb_wait_front(cb_src, 1);
            remote_cb.reserve_back(1);
            experimental::remote_cb_push_back_and_write_pages<true>(
                remote_cb_id, get_read_ptr(cb_src), 1, 1, 1, page_size, noc_index);
            noc_async_posted_writes_flushed();
            cb_pop_front(cb_src, 1);
        }
    } else {
        cb_wait_front(cb_src, ring);
        const uint32_t base = get_read_ptr(cb_src);
        t_start = get_timestamp();
        uint32_t slot = 0;
        for (uint32_t i = 0; i < num_pages; ++i) {
            remote_cb.reserve_back(1);
            experimental::remote_cb_push_back_and_write_pages<true>(
                remote_cb_id, base + slot * page_size, 1, 1, 1, page_size, noc_index);
            slot = (slot + 1 == ring) ? 0 : slot + 1;
        }
    }
    remote_cb.barrier();
    const uint64_t t_end = get_timestamp();
    remote_cb.commit();
    noc_async_posted_writes_flushed();
    noc_async_atomic_barrier();
    if constexpr (mode == 1) {
        cb_pop_front(cb_src, ring);
    }

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
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(stats_page), 64);
    noc_async_write_barrier();
}

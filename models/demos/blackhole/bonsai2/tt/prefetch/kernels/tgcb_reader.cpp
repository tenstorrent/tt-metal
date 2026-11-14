// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear fed by the weight prefetcher, ring reader (BRISC, NOC 0). This core's weight records arrive, in
// order, in its GlobalCircularBuffer ring (remote CB 31; one ring page = one prefetcher block of BLOCK_PAGES 4-KB
// pages). Per block: wait until the prefetcher has landed it, copy its pages (L1 to L1 over the NOC) into the local c_1
// the compute reads, and give the block back to the sender at once. The compute never reads the ring itself: with c_1
// bound to the GCB the compute hung on its last kb batch (math waiting for one more unpack than the unpacker issued).
// The last block may carry padding pages; they are not copied.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_copy = get_arg_val<uint32_t>(1);  // which DRAM-bank copy of x' this core reads
    const uint32_t pages = get_arg_val<uint32_t>(2);   // 4-KB record pages this core's compute reads
    const uint32_t blocks = get_arg_val<uint32_t>(3);  // ring blocks this op takes (pages plus padding)
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_PAGES = get_compile_time_arg_val(1);
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t remote_cb = tt::CBIndex::c_31;
    constexpr uint32_t PAGE = 4096;
    constexpr uint32_t CBP = PAGE / 256;  // c_1 pages per 4-KB page
    constexpr uint32_t X_BYTES = Kt * 64;

    const auto xs = TensorAccessor(x_args, x_addr, X_BYTES);
    cb_reserve_back(cb_x, Kt);
    noc_async_read(xs.get_noc_addr(x_copy), get_write_ptr(cb_x), X_BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);

#ifdef DRAM_FEED
    // debug: the same pages straight from the records in DRAM (runtime args 4, 5: their address, first page)
    {
        constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
        const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(4), PAGE);
        const uint32_t p0 = get_arg_val<uint32_t>(5);
        constexpr uint32_t ppr = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
        for (uint32_t p = 0; p < pages; ++p) {
#ifdef TWO_CB
            const uint32_t cb = ((p / ppr) & 1) ? tt::CBIndex::c_6 : cb_w;  // odd records to c_6, as the DRAM op
#else
            const uint32_t cb = cb_w;
#endif
            cb_reserve_back(cb, CBP);
            noc_async_read(w.get_noc_addr(p0 + p), get_write_ptr(cb), PAGE);
            noc_async_read_barrier();
            cb_push_back(cb, CBP);
        }
        return;
    }
#endif
    for (uint32_t b = 0; b < blocks; ++b) {
        experimental::remote_cb_wait_front(remote_cb, 1);
        const uint32_t src = get_remote_receiver_cb_interface(remote_cb).fifo_rd_ptr;
        const uint32_t first = b * BLOCK_PAGES;
        const uint32_t n = first >= pages ? 0 : (pages - first < BLOCK_PAGES ? pages - first : BLOCK_PAGES);
        if (n) {
            // c_1 holds two records (a whole, even number of 4-KB pages) and every block starts at an even page, so
            // the n pages are contiguous in c_1.
            cb_reserve_back(cb_w, n * CBP);
            noc_async_read(get_noc_addr(src), get_write_ptr(cb_w), n * PAGE);
            noc_async_read_barrier();
            cb_push_back(cb_w, n * CBP);
        }
        experimental::remote_cb_pop_front(remote_cb, 1);
    }
    experimental::update_remote_cb_config_in_l1(remote_cb);
    noc_async_atomic_barrier();
}

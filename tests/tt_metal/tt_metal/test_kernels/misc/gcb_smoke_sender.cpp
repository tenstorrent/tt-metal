// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Smoke-test DRISC sender for DramSenderGlobalCircularBuffer.
//
// Pushes a pre-loaded pattern from DRISC L1 to each configured receiver via the
// experimental remote_cb_* API. The receivers' RemoteReceiverCBInterface is set
// up by the normal CB attachment path (via experimental::CreateCircularBuffer
// receiver overload); the sender side here is hand-managed because we don't
// allocate a sender-side CB on DRAM cores.
//
// With BROADCAST defined, the per-receiver unicast scatter is replaced by one NoC
// multicast of a single page to the whole receiver rectangle, with the credit
// increments left per-receiver and unicast. That is the NoC mechanism the broadcast
// Tensor prefetcher uses; this kernel exercises it independently of the prefetcher's
// own DMA/chunking loop (which the ttnn broadcast mcast-in1 tests cover end to end).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "experimental/drisc_mode.h"

// DRISC firmware does not define cb_interface (no CB infrastructure on DRAM cores);
// the remote_cb_* API references cb_interface[cb_id], so define it here.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    // Compile time
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t pages_sent_drisc_l1_base = get_compile_time_arg_val(4);
    constexpr uint32_t noc_xy_drisc_l1_base = get_compile_time_arg_val(5);
    constexpr uint32_t config_drisc_l1_base = get_compile_time_arg_val(6);
    constexpr uint32_t data_drisc_l1_base = get_compile_time_arg_val(7);
    constexpr uint32_t fifo_size_per_receiver = get_compile_time_arg_val(8);
    constexpr uint32_t receiver_buffer_address = get_compile_time_arg_val(9);
    // Worker L1 address where each receiver reads its own pages_sent counter. The DRISC
    // sender NOC-incs pages_sent here so the receiver sees it locally.
    constexpr uint32_t remote_pages_sent_worker_l1_addr = get_compile_time_arg_val(10);

    // Runtime: 2*num_receivers entries (x, y per receiver)
    uint32_t rt_idx = 0;

    // 1) Populate the noc_xy table in DRISC L1.
    volatile tt_l1_ptr uint32_t* noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_drisc_l1_base);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        noc_xy_ptr[2 * i] = get_arg_val<uint32_t>(rt_idx++);
        noc_xy_ptr[2 * i + 1] = get_arg_val<uint32_t>(rt_idx++);
    }

    // 2) Zero the pages_sent/pages_acked semaphore region. On DRISC the slots are
    // packed at uint32 stride (2 uint32_t per receiver: pages_sent then pages_acked),
    // matching REMOTE_CB_LOCAL_PAGES_STRIDE in remote_circular_buffer.h.
    volatile tt_l1_ptr uint32_t* psem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pages_sent_drisc_l1_base);
    const uint32_t psem_uints = 2 * num_receivers;
    for (uint32_t i = 0; i < psem_uints; ++i) {
        psem[i] = 0;
    }

    // 3) Stand up a mock config block: only [3] (fifo_size) is dereferenced by the
    // sender path (resize_remote_sender_cb_interface, remote_cb_reserve_back). The
    // kernel does not call update_remote_cb_config_in_l1, so the fifo_rd_ptr slot at
    // offsetof(RemoteReceiverCBInterface, fifo_rd_ptr) (= byte 16) is left unallocated.
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_drisc_l1_base);
    cfg[0] = 1;                        // is_sender (unused on sender path, but set)
    cfg[1] = num_receivers;            // unused
    cfg[2] = receiver_buffer_address;  // unused
    cfg[3] = fifo_size_per_receiver;   // <- read by reserve/resize

    // 4) Populate the sender CB interface manually.
    RemoteSenderCBInterface& iface = get_remote_sender_cb_interface(remote_cb_id);
    iface.config_ptr = config_drisc_l1_base;
    iface.fifo_start_addr = receiver_buffer_address;
    iface.fifo_wr_ptr = receiver_buffer_address;
    iface.receiver_noc_xy_ptr = noc_xy_drisc_l1_base;
    iface.aligned_pages_sent_ptr = pages_sent_drisc_l1_base;
    // num_receivers and the worker-side pages_sent override share a packed 32-bit slot
    // (see RemoteSenderCBInterface). Sender pages_sent counters live in DRISC L1
    // (pages_sent_drisc_l1_base), but the NoC inc target lives in worker L1 (where
    // receivers read their local pages_sent) — different L1 address spaces, so we use
    // the override.
    iface.num_receivers_and_remote_pages_sent_ptr = remote_cb_pack(num_receivers, remote_pages_sent_worker_l1_addr);

    // DRISC needs stream mode for NIU-initiated NoC traffic.
    experimental::drisc_set_stream_mode();

    experimental::resize_remote_sender_cb_interface<false>(remote_cb_id, page_size, noc_index);

    // 5) Reserve and push num_pages, one page at a time.
#ifdef BROADCAST
    // Multicast rectangle in virtual worker coords, min corner first (NOC0 orientation).
    constexpr uint32_t mcast_start_x = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_start_y = get_compile_time_arg_val(12);
    constexpr uint32_t mcast_end_x = get_compile_time_arg_val(13);
    constexpr uint32_t mcast_end_y = get_compile_time_arg_val(14);
    const uint32_t sx = noc_index == 0 ? mcast_start_x : mcast_end_x;
    const uint32_t sy = noc_index == 0 ? mcast_start_y : mcast_end_y;
    const uint32_t ex = noc_index == 0 ? mcast_end_x : mcast_start_x;
    const uint32_t ey = noc_index == 0 ? mcast_end_y : mcast_start_y;
    const uint32_t mcast_noc_xy = uint32_t(NOC_MULTICAST_ENCODING(
        DYNAMIC_NOC_X(noc_index, sx),
        DYNAMIC_NOC_Y(noc_index, sy),
        DYNAMIC_NOC_X(noc_index, ex),
        DYNAMIC_NOC_Y(noc_index, ey)));

    for (uint32_t i = 0; i < num_pages; ++i) {
        experimental::remote_cb_reserve_back(remote_cb_id, 1);
        // One multicast delivers the page to every receiver at the same fifo offset.
        noc_async_write_multicast_one_packet(
            data_drisc_l1_base + i * page_size,
            get_noc_addr_helper(mcast_noc_xy, iface.fifo_wr_ptr),
            page_size,
            num_receivers,
            /*linked=*/false,
            noc_index);
        // Multicast is non-posted; wait for the acks so the data has landed before the
        // credits that announce it.
        noc_async_write_barrier();

        // Credits stay per-receiver and unicast, exactly as prefetcher_finalize_block does.
        const uint32_t pages_sent = page_size / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
        volatile tt_l1_ptr uint32_t* local_sent =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
        uint32_t remote_sent_base = remote_cb_remote_pages_sent_ptr(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);
        for (uint32_t r = 0; r < num_receivers; ++r) {
            const uint32_t remote_noc_xy =
                uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc_index, xy[0]), DYNAMIC_NOC_Y(noc_index, xy[1])));
            *local_sent += pages_sent;
            noc_semaphore_inc<true>(get_noc_addr_helper(remote_noc_xy, remote_sent_base), pages_sent, noc_index);
            local_sent += experimental::REMOTE_CB_LOCAL_PAGES_STRIDE / sizeof(uint32_t);
            remote_sent_base += 2 * L1_ALIGNMENT;
            xy += 2;
        }
        iface.fifo_wr_ptr += page_size;
    }
#else
    for (uint32_t i = 0; i < num_pages; ++i) {
        experimental::remote_cb_reserve_back(remote_cb_id, 1);
        experimental::remote_cb_push_back_and_write_pages<false>(
            remote_cb_id,
            data_drisc_l1_base + i * page_size,  // src
            1,                                   // num_pages
            1,                                   // num_rows
            1,                                   // coalesced_num_pages_per_row
            page_size,                           // coalesced_page_size
            noc_index);
        noc_async_posted_writes_flushed();
    }
#endif  // BROADCAST

    experimental::remote_cb_sender_barrier(remote_cb_id);
    // Production prefetcher kernels call update_remote_cb_config_in_l1() here to checkpoint
    // fifo_rd_ptr for the next program; this smoke kernel doesn't have a follow-on consumer
    // and the mock config block above doesn't reserve that slot, so skip it.
    noc_async_atomic_barrier();
    experimental::drisc_set_noc2axi_mode();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused monitor+drain: no poll phase. One read per core covers the control vector AND all five rings,
// because profiler_msg_t is contiguous -- 64 control words followed by 5 x 2 KB, 10,496 B total, still
// one NoC packet.
//
// That inverts the adaptive decision. The old shape asked "should I read this core?" and paid a
// separate 256 B poll to answer it, then read hot cores a second time. Here the read is unconditional
// and the decision is "should I push what I already have?" -- which fits the measured cost structure,
// where a read costs ~40 cycles regardless of payload but egress is the scarce resource.
//
// Dropping the 30,720 B poll ring frees L1 for 8-core batches, so reads run at depth 8 again.
//
// A hot core is pushed as its own 10,240 B page (the data portion, skipping the control vector), which
// avoids having to compact non-adjacent hot cores into a fixed-size page.
//
// The NIU must already be in stream mode -- set by a prior program so the socket config lands in L1.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kCoresPerBatch = get_compile_time_arg_val(0);
    constexpr uint32_t kCoreSpan = get_compile_time_arg_val(1);    // 10496 = control vector + 5 rings
    constexpr uint32_t kDataOffset = get_compile_time_arg_val(2);  // 256, past the control vector
    constexpr uint32_t kPageBytes = get_compile_time_arg_val(3);   // 10240, one core's rings
    constexpr uint32_t kThresholdWords = get_compile_time_arg_val(4);
    constexpr uint32_t kBufBase = get_compile_time_arg_val(5);
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(7);

    constexpr uint32_t kTailWordOffset =
        5;  // = kernel_profiler::SPSC_RING_TAIL_0, the first of the 5 per-RISC tails in the control vector
    constexpr uint32_t kNumRisc = 5;
    static_assert(kCoreSpan <= NOC_MAX_BURST_SIZE, "control vector + rings must fit one NoC packet");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t num_sweeps = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);  // start of profiler_msg_t on the worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(3));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    uint64_t t_wbar = 0;
    uint64_t t_read = 0;
    uint64_t t_decide = 0;
    uint64_t t_push = 0;
    uint32_t pages = 0;
    uint32_t pending_acc = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        uint32_t core = 0;
        while (core < num_cores) {
            const uint64_t w0 = get_timestamp();
            noc_async_write_barrier();  // pages written out of this buffer must have landed
            const uint64_t w1 = get_timestamp();

            uint32_t n = 0;
            for (; n < kCoresPerBatch && core < num_cores; n++, core++) {
                const uint32_t xy = coords[core];
                CoreLocalMem<uint32_t> dst(kBufBase + n * kCoreSpan);
                noc.async_read<NocOptions::DEFAULT, kCoreSpan>(
                    src, dst, kCoreSpan, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = src_addr}, {});
            }
            noc.async_read_barrier();
            const uint64_t w2 = get_timestamp();

            // Decide and push from data already in L1 -- no second visit to the core.
            for (uint32_t j = 0; j < n; j++) {
                const uint32_t slot = kBufBase + j * kCoreSpan;
                volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                uint32_t full = 0;
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    full += cv[kTailWordOffset + r];  // heads are 0 here, so tail - head == tail
                }
                pending_acc += full;
                if (full >= kThresholdWords) {
                    const uint64_t d0 = get_timestamp();
                    socket_reserve_pages(sender, 1);
                    noc_write_page_chunked(pcie_xy_enc, slot + kDataOffset, pcie_base + sender.write_ptr, kPageBytes);
                    socket_push_pages(sender, 1);
                    socket_notify_receiver(sender);
                    const uint64_t d1 = get_timestamp();
                    t_push += d1 - d0;
                    pages++;
                }
            }
            const uint64_t w3 = get_timestamp();

            t_wbar += w1 - w0;
            t_read += w2 - w1;
            t_decide += (w3 - w2);  // push time is subtracted on the host side
        }
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = pages;
    out[3] = static_cast<uint32_t>(t_wbar & 0xFFFFFFFFu);
    out[4] = static_cast<uint32_t>(t_read & 0xFFFFFFFFu);
    out[5] = static_cast<uint32_t>(t_read >> 32);
    out[6] = static_cast<uint32_t>(t_decide & 0xFFFFFFFFu);
    out[7] = static_cast<uint32_t>(t_decide >> 32);
    out[8] = static_cast<uint32_t>(t_push & 0xFFFFFFFFu);
    out[9] = static_cast<uint32_t>(t_push >> 32);
    out[10] = pending_acc;

    update_socket_config(sender);
}

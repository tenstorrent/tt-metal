// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC -> host egress benchmark over a real D2H socket.
//
// Same push sequence as the worker-core benchmark (pcie_socket_sender_benchmark.cpp): reserve a page,
// write it to the host FIFO through the PCIe tile, publish it, notify the receiver. The point is to
// find what a DRAM core can actually sustain to host, since ingest on this core measured far faster
// than any egress path is likely to be.
//
// The NIU must already be in stream mode when this runs -- a DRISC cannot initiate NoC traffic at all
// in NOC2AXI mode, and the socket's config/ack writes need to land in L1. drisc_niu_mode.cpp does that
// in a prior program and deliberately does not restore, so this kernel neither sets nor restores it.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores) -- same shim the shipping
// DRISC prefetcher kernel uses so socket_api.h links.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(0);
    constexpr uint32_t kSrcL1 = get_compile_time_arg_val(1);
    constexpr uint32_t kPageSize = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    constexpr uint32_t kNumPages = get_compile_time_arg_val(4);
    // Publish bytes_sent every kNotifyEvery pages instead of every page. socket_push_pages is local
    // state (write_ptr + a counter), but socket_notify_receiver is a 4 B PCIe write costing ~275 ns --
    // 19% of a page at the protocol ceiling. The caller must keep kNotifyEvery well under the FIFO
    // depth: the host cannot free space it has not been told about.
    constexpr uint32_t kNotifyEvery = get_compile_time_arg_val(5);

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageSize);

    // Per-phase accounting. Unlike the read benchmarks -- where a ~40-cycle operation cannot be timed
    // with a 26-cycle timer -- a page here costs thousands of cycles, so 4 probes per iteration is ~2%
    // and in-loop timestamps are the right tool. Each phase is inflated by roughly one timer read.
    //
    //   t_wait   socket_reserve_pages spinning on the host to free FIFO space
    //   t_write  issuing the page to the PCIe tile. The writes are posted, so this is issue cost plus
    //            any NoC/PCIe back-pressure absorbed at the command buffer -- not completion.
    //   t_notify socket_push_pages + socket_notify_receiver: the bytes_sent publish over PCIe
    uint64_t t_wait = 0;
    uint64_t t_write = 0;
    uint64_t t_notify = 0;
    const uint64_t t_start = get_timestamp();
    uint32_t since_notify = 0;
    for (uint32_t i = 0; i < kNumPages; i++) {
        const uint64_t a = get_timestamp();
        socket_reserve_pages(sender, 1);
        const uint64_t b = get_timestamp();
        noc_write_page_chunked(pcie_xy_enc, kSrcL1, pcie_base + sender.write_ptr, kPageSize);
        const uint64_t c = get_timestamp();
        socket_push_pages(sender, 1);
        if (++since_notify == kNotifyEvery) {
            socket_notify_receiver(sender);
            since_notify = 0;
        }
        const uint64_t d = get_timestamp();
        t_wait += b - a;
        t_write += c - b;
        t_notify += d - c;
    }
    // Flush the tail: socket_barrier waits for bytes_acked == bytes_sent, and the host cannot ack
    // pages it was never told about.
    socket_notify_receiver(sender);
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    // Cost of the instrument itself, so the breakdown can be discounted honestly.
    uint64_t timer_overhead = 0;
    {
        constexpr uint32_t kProbes = 1024;
        const uint64_t p0 = get_timestamp();
        for (uint32_t i = 0; i < kProbes; i++) {
            (void)get_timestamp();
        }
        timer_overhead = (get_timestamp() - p0) / kProbes;
    }

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(t_wait & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(t_wait >> 32);
    out[4] = kNumPages;
    out[5] = static_cast<uint32_t>(t_write & 0xFFFFFFFFu);
    out[6] = static_cast<uint32_t>(t_write >> 32);
    out[7] = static_cast<uint32_t>(t_notify & 0xFFFFFFFFu);
    out[8] = static_cast<uint32_t>(t_notify >> 32);
    out[9] = static_cast<uint32_t>(timer_overhead);

    update_socket_config(sender);
}

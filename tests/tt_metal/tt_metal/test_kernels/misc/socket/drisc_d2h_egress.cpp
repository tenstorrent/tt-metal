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

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageSize);

    // Split the wait from the transfer: socket_reserve_pages spins until the host has freed space, so
    // lumping it in would report host-consumption rate rather than device egress rate. t_wait isolates it.
    uint64_t t_wait = 0;
    const uint64_t t_start = get_timestamp();
    for (uint32_t i = 0; i < kNumPages; i++) {
        const uint64_t w0 = get_timestamp();
        socket_reserve_pages(sender, 1);
        t_wait += get_timestamp() - w0;
        noc_write_page_chunked(pcie_xy_enc, kSrcL1, pcie_base + sender.write_ptr, kPageSize);
        socket_push_pages(sender, 1);
        socket_notify_receiver(sender);
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(t_wait & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(t_wait >> 32);
    out[4] = kNumPages;

    update_socket_config(sender);
}

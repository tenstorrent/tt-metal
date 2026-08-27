// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Pipeline leg B: drain a DRAM sweep buffer to the host over a D2H socket.
//
// B lives on a different bank than A, so it cannot use its DMA engine to reach A's bank -- gddr_dma is
// bank-local. Instead it NoC-reads A's DRAM through that bank's NOC1 worker endpoint, which is still in
// NOC2AXI mode and therefore forwards DRAM-range addresses to GDDR. That endpoint must be left alone:
// A sits on the bank's free subchannel precisely so this path stays intact.
//
// Per page: pull kNocReadsPerPage x NOC_MAX_BURST-sized reads out of DRAM into one L1 page, then push
// that page to the host. Reads are split because a page is larger than one NoC packet.
//
// Flow control mirrors A: B waits for fill_idx > drain_idx, drains that buffer, then publishes
// drain_idx into A's L1. t_block is how long B waits on A -- if it is large, B is not the bottleneck.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "experimental/drisc_mode.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(0);
    constexpr uint32_t kL1Buf = get_compile_time_arg_val(1);
    constexpr uint32_t kPageBytes = get_compile_time_arg_val(2);
    constexpr uint32_t kNocChunk = get_compile_time_arg_val(3);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(4);
    constexpr uint32_t kFillFlagAddr = get_compile_time_arg_val(5);  // in B's L1, written by A
    constexpr uint32_t kSweepBytes = get_compile_time_arg_val(6);

    static_assert(kNocChunk <= NOC_MAX_BURST_SIZE, "DRAM read chunk must fit one NoC packet");
    static_assert(kPageBytes % kNocChunk == 0, "page must be a whole number of NoC reads");
    static_assert(kSweepBytes % kPageBytes == 0, "sweep must be a whole number of pages");
    constexpr uint32_t kNocReadsPerPage = kPageBytes / kNocChunk;
    constexpr uint32_t kPagesPerSweep = kSweepBytes / kPageBytes;

    const uint32_t num_sweeps = get_arg_val<uint32_t>(0);
    const uint32_t dram_xy = get_arg_val<uint32_t>(1);  // bank A's NOC1 worker endpoint
    const uint32_t dram_base = get_arg_val<uint32_t>(2);
    const uint32_t a_xy = get_arg_val<uint32_t>(3);
    const uint32_t a_drain_flag_addr = get_arg_val<uint32_t>(4);

    Noc noc;
    UnicastEndpoint src;

    // A separate program already put this NIU in stream mode so the socket's config write could land in
    // L1; this kernel must not disturb it, and must not restore NOC2AXI while the socket is live.
    volatile tt_l1_ptr uint32_t* fill_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kFillFlagAddr);
    *fill_flag = 0;
    const uint64_t a_flag_noc_addr = get_noc_addr(a_xy & 0xFFFFu, a_xy >> 16, a_drain_flag_addr);

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    uint64_t t_block = 0;
    uint64_t t_dram = 0;
    uint64_t t_push = 0;
    uint32_t drain_idx = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        const uint64_t b0 = get_timestamp();
        while (*fill_flag <= drain_idx) {
            invalidate_l1_cache();
        }
        t_block += get_timestamp() - b0;

        const uint32_t sweep_off = (drain_idx & 1u) * kSweepBytes;
        for (uint32_t page = 0; page < kPagesPerSweep; page++) {
            const uint32_t page_off = sweep_off + page * kPageBytes;

            const uint64_t r0 = get_timestamp();
            for (uint32_t i = 0; i < kNocReadsPerPage; i++) {
                CoreLocalMem<uint32_t> dst(kL1Buf + i * kNocChunk);
                noc.async_read<NocOptions::DEFAULT, kNocChunk>(
                    src,
                    dst,
                    kNocChunk,
                    {.noc_x = dram_xy & 0xFFFFu, .noc_y = dram_xy >> 16, .addr = dram_base + page_off + i * kNocChunk},
                    {});
            }
            noc.async_read_barrier();
            const uint64_t r1 = get_timestamp();

            socket_reserve_pages(sender, 1);
            noc_write_page_chunked(pcie_xy_enc, kL1Buf, pcie_base + sender.write_ptr, kPageBytes);
            socket_push_pages(sender, 1);
            socket_notify_receiver(sender);
            const uint64_t r2 = get_timestamp();

            t_dram += r1 - r0;
            t_push += r2 - r1;
        }

        drain_idx++;
        noc_inline_dw_write(a_flag_noc_addr, drain_idx);
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(t_block & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(t_block >> 32);
    out[4] = drain_idx;
    out[5] = static_cast<uint32_t>(t_dram & 0xFFFFFFFFu);
    out[6] = static_cast<uint32_t>(t_dram >> 32);
    out[7] = static_cast<uint32_t>(t_push & 0xFFFFFFFFu);
    out[8] = static_cast<uint32_t>(t_push >> 32);

    update_socket_config(sender);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel is used for two tests:
// 1. Multicast test: DRISC L1 reads from DRAM over DMA and mcasts to Tensix L1
// 2. Double buffered DRISC L1 reads from DRAM over DMA and unicast to Tensix L1 (using TRIDs).
//    We measure max possible bandwidth using this test

#include "experimental/noc.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    const uint32_t src_gddr_addr = get_arg_val<uint32_t>(0);
    const uint32_t drisc_l1_addr = get_arg_val<uint32_t>(1);
    const uint32_t tensix_l1_dst_addr = get_arg_val<uint32_t>(2);
    const uint32_t tensix_noc_x_start = get_arg_val<uint32_t>(3);
    const uint32_t tensix_noc_y_start = get_arg_val<uint32_t>(4);
    const uint32_t tensix_noc_x_end = get_arg_val<uint32_t>(5);
    const uint32_t tensix_noc_y_end = get_arg_val<uint32_t>(6);
    const uint32_t num_bytes = get_arg_val<uint32_t>(7);
    const uint32_t num_subordinates = get_arg_val<uint32_t>(8);

    experimental::Noc noc;
    experimental::UnicastEndpoint src;
    // Stream mode: required for DRISC to initiate NOC traffic and for
    // remote cores to reach DRISC L1 over NOC.
    experimental::drisc_set_stream_mode();

#if defined(MULTICAST)
    experimental::MulticastEndpoint dst;
    // Read chunk from GDDR into DRISC L1
    constexpr uint8_t tx_stream = 0;
    experimental::dma_async_read(tx_stream, src_gddr_addr, drisc_l1_addr, num_bytes);
    experimental::dma_async_read_barrier(tx_stream);
    noc.async_write_multicast(
        src,
        dst,
        num_bytes,
        num_subordinates,
        {.addr = drisc_l1_addr},
        {
            .noc_x_start = tensix_noc_x_start,
            .noc_y_start = tensix_noc_y_start,
            .noc_x_end = tensix_noc_x_end,
            .noc_y_end = tensix_noc_y_end,
            .addr = tensix_l1_dst_addr,
        });
    noc.async_write_barrier();
#else
    const uint32_t total_iters = get_arg_val<uint32_t>(9);
    experimental::UnicastEndpoint dst;
    // Single-stream double-buffered: both halves share stream 0 and are
    // pipelined via two pre-fired DMA reads
    constexpr uint8_t tx_stream = 0;
    const uint32_t half_buffer_bytes = num_bytes / 2;
    uint32_t src_A = src_gddr_addr;
    uint32_t src_B = src_gddr_addr + half_buffer_bytes;
    const uint32_t dst_A = drisc_l1_addr;
    const uint32_t dst_B = drisc_l1_addr + half_buffer_bytes;
    const uint32_t tensix_dst_A = tensix_l1_dst_addr;
    const uint32_t tensix_dst_B = tensix_l1_dst_addr + half_buffer_bytes;
    constexpr uint32_t trid_A = 10;
    constexpr uint32_t trid_B = 11;
    uint64_t start = get_timestamp();

    // Prime both halves so the steady-state loop always has two reads in
    // flight on the shared stream. dma_async_read_wait_n(stream, 1) then
    // waits for the oldest of the two to land (FIFO completion order).
    experimental::dma_async_read(tx_stream, src_A, dst_A, half_buffer_bytes);
    experimental::dma_async_read(tx_stream, src_B, dst_B, half_buffer_bytes);
    src_A += num_bytes;
    src_B += num_bytes;

    // Steady-state interleave (per iteration):
    // Each side: wait DMA -> NOC write -> flushed (NIU released the L1 buffer) -> refill DMA.
    // Refilling strictly after consume avoids a race condition and avoids overwriting unconsumed data;
    // the two sides still overlap because NOC writes are async (issue then move on).
    // Maintains 2 outstanding DMAs throughout (interleaved consume/refill)
    for (uint32_t i = 0; i < total_iters - 1; i++) {
        experimental::dma_async_read_wait_n(tx_stream, 1);                        // A[i] ready (oldest)
        noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);  // prev A acked
        noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_A},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
            NOC_UNICAST_WRITE_VC,
            trid_A);
        noc.async_writes_flushed<experimental::Noc::ResponseMode::NON_POSTED, experimental::Noc::BarrierMode::TXN_ID>(
            trid_A);
        experimental::dma_async_read(tx_stream, src_A, dst_A, half_buffer_bytes);  // refill A

        experimental::dma_async_read_wait_n(tx_stream, 1);                        // B[i] ready (oldest now)
        noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);  // prev B acked
        noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_B},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
            NOC_UNICAST_WRITE_VC,
            trid_B);
        noc.async_writes_flushed<experimental::Noc::ResponseMode::NON_POSTED, experimental::Noc::BarrierMode::TXN_ID>(
            trid_B);
        experimental::dma_async_read(tx_stream, src_B, dst_B, half_buffer_bytes);  // refill B
        src_A += num_bytes;
        src_B += num_bytes;
    }

    // Final A: oldest of the two outstanding reads is A[last].
    experimental::dma_async_read_wait_n(tx_stream, 1);                        // A[last] ready
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);  // prev A acked
    noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_A},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
        NOC_UNICAST_WRITE_VC,
        trid_A);

    // Final B: drain the remaining outstanding read.
    experimental::dma_async_read_wait_n(tx_stream, 0);                        // B[last] ready
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);  // prev B acked
    noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_B},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
        NOC_UNICAST_WRITE_VC,
        trid_B);

    // Drain: wait for the final two NOC writes to be acked.
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);

    uint64_t end = get_timestamp();
    uint64_t total_time = end - start;
    // Timing stored immediately after data buffer. Host reads at drisc_l1_addr + num_bytes
    experimental::CoreLocalMem<uint64_t> total_time_res(drisc_l1_addr);
    uint32_t offset = num_bytes / sizeof(uint64_t);
    total_time_res[offset] = total_time;
#endif
    // Always restore NOC2AXI so subsequent context observes the default.
    experimental::drisc_set_noc2axi_mode();
}

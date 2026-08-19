// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC kernel: reads GDDR over DMA into DRISC L1, then sends to Tensix L1.
//   MULTICAST defined   -> multicast each chunk to a Tensix grid (mcast / NOC-mode tests).
//   MULTICAST not defined -> double-buffered unicast with TRIDs (bandwidth measurement).

#include "api/dataflow/noc.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

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
    const uint32_t total_iters = get_arg_val<uint32_t>(9);

    Noc noc;
    UnicastEndpoint src;
    // Stream mode: required for DRISC to initiate NOC traffic and for
    // remote cores to reach DRISC L1 over NOC.
    experimental::drisc_set_stream_mode();

#if defined(MULTICAST)
    MulticastEndpoint dst;
    constexpr uint8_t tx_stream = 0;
    // The two NOCs are opposing tori: pick the multicast entry corner to match the NOC's flow
    // direction (NOC0 enters at the min corner, NOC1 at the max corner).
    const uint32_t noc_x_start = noc_index == 0 ? tensix_noc_x_start : tensix_noc_x_end;
    const uint32_t noc_y_start = noc_index == 0 ? tensix_noc_y_start : tensix_noc_y_end;
    const uint32_t noc_x_end = noc_index == 0 ? tensix_noc_x_end : tensix_noc_x_start;
    const uint32_t noc_y_end = noc_index == 0 ? tensix_noc_y_end : tensix_noc_y_start;
    uint32_t gddr_addr = src_gddr_addr;
    // Per iter: DMA the next GDDR chunk into DRISC L1, then multicast it to the Tensix grid
    for (uint32_t i = 0; i < total_iters; i++) {
        experimental::dma_async_read(tx_stream, gddr_addr, drisc_l1_addr, num_bytes);
        experimental::dma_async_read_barrier(tx_stream);
        noc.async_write_multicast(
            src,
            dst,
            num_bytes,
            num_subordinates,
            {.addr = drisc_l1_addr},
            {
                .noc_x_start = noc_x_start,
                .noc_y_start = noc_y_start,
                .noc_x_end = noc_x_end,
                .noc_y_end = noc_y_end,
                .addr = tensix_l1_dst_addr,
            });
        gddr_addr += num_bytes;
        noc.async_write_barrier();
    }
#else
    UnicastEndpoint dst;
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
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_A});  // prev A acked
        noc.async_write<NocOptions::TXN_ID>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_A},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
            NocOptVals{.trid = trid_A});
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = trid_A});
        experimental::dma_async_read(tx_stream, src_A, dst_A, half_buffer_bytes);  // refill A

        experimental::dma_async_read_wait_n(tx_stream, 1);                        // B[i] ready (oldest now)
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_B});  // prev B acked
        noc.async_write<NocOptions::TXN_ID>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_B},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
            NocOptVals{.trid = trid_B});
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = trid_B});
        experimental::dma_async_read(tx_stream, src_B, dst_B, half_buffer_bytes);  // refill B
        src_A += num_bytes;
        src_B += num_bytes;
    }

    // Final A: oldest of the two outstanding reads is A[last].
    experimental::dma_async_read_wait_n(tx_stream, 1);                        // A[last] ready
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_A});  // prev A acked
    noc.async_write<NocOptions::TXN_ID>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_A},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
        NocOptVals{.trid = trid_A});

    // Final B: drain the remaining outstanding read.
    experimental::dma_async_read_wait_n(tx_stream, 0);                        // B[last] ready
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_B});  // prev B acked
    noc.async_write<NocOptions::TXN_ID>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_B},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
        NocOptVals{.trid = trid_B});

    // Drain: wait for the final two NOC writes to be acked.
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_A});
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = trid_B});

    uint64_t end = get_timestamp();
    uint64_t total_time = end - start;
    // Timing stored immediately after data buffer. Host reads at drisc_l1_addr + num_bytes
    CoreLocalMem<uint64_t> total_time_res(drisc_l1_addr);
    uint32_t offset = num_bytes / sizeof(uint64_t);
    total_time_res[offset] = total_time;
#endif
    // Always restore NOC2AXI so subsequent context observes the default.
    experimental::drisc_set_noc2axi_mode();
}

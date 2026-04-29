// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC Test Kernel
// Tests DRISC DRAN reads over DMA and mcasts to Tensix L1

#include "api/compile_time_args.h"
#include "experimental/noc.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint64_t src_gddr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t drisc_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t tensix_l1_dst_addr = get_compile_time_arg_val(2);
    constexpr uint32_t tensix_noc_x_start = get_compile_time_arg_val(3);
    constexpr uint32_t tensix_noc_y_start = get_compile_time_arg_val(4);
    constexpr uint32_t tensix_noc_x_end = get_compile_time_arg_val(5);
    constexpr uint32_t tensix_noc_y_end = get_compile_time_arg_val(6);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(8);

    experimental::Noc noc;
    experimental::UnicastEndpoint src;
    // Stream mode: required for DRISC to initiate NOC traffic and for
    // remote cores to reach DRISC L1 over NOC.
    experimental::drisc_set_stream_mode_all();

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
    constexpr uint32_t total_iters = get_compile_time_arg_val(9);
    experimental::UnicastEndpoint dst;
    constexpr uint8_t tx_stream = 0;
    constexpr uint8_t num_reads_outstanding = 1;
    constexpr uint32_t half_buffer_bytes = num_bytes / 2;
    constexpr uint32_t src_A = src_gddr_addr;
    constexpr uint32_t src_B = src_gddr_addr + half_buffer_bytes;
    constexpr uint32_t dst_A = drisc_l1_addr;
    constexpr uint32_t dst_B = drisc_l1_addr + half_buffer_bytes;
    constexpr uint32_t tensix_dst_A = tensix_l1_dst_addr;
    constexpr uint32_t tensix_dst_B = tensix_l1_dst_addr + half_buffer_bytes;
    constexpr uint32_t trid_A = 0;
    constexpr uint32_t trid_B = 1;
    uint64_t start = get_timestamp();
    experimental::dma_async_read(tx_stream, src_A, dst_A, half_buffer_bytes);
    experimental::dma_async_read(tx_stream, src_B, dst_B, half_buffer_bytes);
    for (uint32_t i = 0; i < total_iters - 2; i++) {
        experimental::dma_async_read_wait_n(tx_stream, num_reads_outstanding);
        // noc async write from dst_A
        noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);
        noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_A},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
            NOC_UNICAST_WRITE_VC,
            trid_A);
        experimental::dma_async_read(tx_stream, src_A, dst_A, half_buffer_bytes);
        experimental::dma_async_read_wait_n(tx_stream, num_reads_outstanding);
        // noc async write from dst_B
        noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);
        noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
            src,
            dst,
            half_buffer_bytes,
            {.addr = dst_B},
            {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
            NOC_UNICAST_WRITE_VC,
            trid_B);
        experimental::dma_async_read(tx_stream, src_B, dst_B, half_buffer_bytes);
    }
    // Drain: two reads still in flight - wait for each in turn and flush its NoC write.
    experimental::dma_async_read_wait_n(tx_stream, num_reads_outstanding);
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_A);
    noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_A},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_A},
        NOC_UNICAST_WRITE_VC,
        trid_A);
    experimental::dma_async_read_wait_n(tx_stream, 0);
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(trid_B);
    noc.async_write<experimental::Noc::TxnIdMode::ENABLED>(
        src,
        dst,
        half_buffer_bytes,
        {.addr = dst_B},
        {.noc_x = tensix_noc_x_start, .noc_y = tensix_noc_y_start, .addr = tensix_dst_B},
        NOC_UNICAST_WRITE_VC,
        trid_B);
    noc.async_write_barrier(trid_A);
    noc.async_write_barrier(trid_B);
    uint64_t end = get_timestamp();
    uint64_t total_time = end - start;
    experimental::CoreLocalMem<uint64_t> total_time_res(drisc_l1_addr);
    uint32_t offset = (num_bytes) / sizeof(uint64_t);
    total_time_res[offset] = total_time;
#endif

    // Always restore NOC2AXI so subsequent context observes the default.
    experimental::drisc_set_noc2axi_mode_all();
}

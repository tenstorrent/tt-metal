// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/debug/assert.h"
#include "internal/tt-1xx/blackhole/gddr_dma_regs.h"

namespace experimental {

/*
    GDDR DMA API for DRISC kernels - DRISC L1 <-> GDDR transfers.

    Two independent TX streams (0 and 1). Each stream can concurrently carry:
    - 255 outstanding reads
    - 15 outstanding writes

    Typical read:
        dma_async_read(stream, src_gddr, dst_l1, size_bytes);
        dma_async_read_barrier(stream);

    Typical write:
        dma_async_write(stream, src_l1, dst_gddr, size_bytes);
        dma_async_write_barrier(stream);

    All size and increment parameters are in bytes and must be multiples of 16.
    stream must be 0 or 1.
*/

#ifdef COMPILE_FOR_DRISC

//////////////////////////////////////////////////////////////////
/////////////////// Internal helpers /////////////////////////////
//////////////////////////////////////////////////////////////////

static inline __attribute__((always_inline)) void check_stream_(uint8_t stream) {
    ASSERT(stream == 0 || stream == 1, DebugAssertTripped);
}

static inline __attribute__((always_inline)) void check_transfer_size_(uint32_t size_bytes) {
    ASSERT((size_bytes & 0xF) == 0, DebugAssertTripped);
    ASSERT(size_bytes <= 262128u, DebugAssertTripped);  // 14-bit transfer_size_words field
}

static inline __attribute__((always_inline)) void check_outstanding_reads_(uint32_t n) {
    ASSERT(n <= 255, DebugAssertTripped);
}

static inline __attribute__((always_inline)) void check_outstanding_writes_(uint32_t n) {
    ASSERT(n <= 15, DebugAssertTripped);
}

static inline __attribute__((always_inline)) void program_dma_write_addresses_(
    uint8_t stream, uint32_t src_l1, uint64_t dst_gddr) {
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_TRANSFER_START_ADDR_REG_OFFSET, src_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_DEST_ADDR_LOW_REG_OFFSET, (uint32_t)(dst_gddr & 0xFFFFFFFFu));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_DEST_ADDR_HIGH_REG_OFFSET, (uint32_t)(dst_gddr >> 32));
}

static inline __attribute__((always_inline)) void program_dma_read_addresses_(
    uint8_t stream, uint64_t src_gddr, uint32_t dst_l1) {
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_LOW_REG_OFFSET, (uint32_t)(src_gddr & 0xFFFFFFFFu));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_HIGH_REG_OFFSET, (uint32_t)(src_gddr >> 32));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_DEST_REG_OFFSET, dst_l1);
}

//////////////////////////////////////////////////////////////////
/////////////////// Local API (DRISC only) ///////////////////////
//////////////////////////////////////////////////////////////////

/**
 * @brief Restore DMA global transfer attributes to hardware defaults.
 *
 * The DmaCtrlTransferAttrs register persists across kernel runs. Call this
 * at the end of any kernel that modifies it (e.g. via dma_set_max_burst_size)
 * so subsequent kernels observe the expected default state.
 */
inline __attribute__((always_inline)) void dma_restore_default_transfer_attrs() {
    WRITE_TX_CTRL_REG(TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET, DmaCtrlTransferAttrs_DEFAULT);
}

/**
 * @brief Set AXI burst size for DMA transfers.
 *
 * Default is 16 beats (1 KB at 64 B/beat). Higher values can reduce AXI transaction
 * overhead. Valid range: 1-255.
 *
 * @param burst_size  Beats per AXI burst (1-255).
 */
inline __attribute__((always_inline)) void dma_set_burst_size(uint8_t burst_size = 0x10) {
    ASSERT(burst_size > 0 && burst_size <= 255, DebugAssertTripped);
    DmaCtrlTransferAttrs_u attrs;
    attrs.val = READ_TX_CTRL_REG(TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET);
    attrs.f.max_burst_size = burst_size;
    WRITE_TX_CTRL_REG(TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Non-blocking GDDR to L1 read. Pair with dma_async_read_barrier().
 *
 * @param stream      TX stream (0 or 1).
 * @param src_gddr    GDDR source address (64-bit).
 * @param dst_l1      DRISC L1 destination address (32-bit).
 * @param size_bytes  Transfer size in bytes (must be a multiple of 16).
 */
inline __attribute__((always_inline)) void dma_async_read(
    uint8_t stream, uint64_t src_gddr, uint32_t dst_l1, uint32_t size_bytes) {
    check_stream_(stream);
    check_transfer_size_(size_bytes);

    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.transfer_start_read = 1;

    volatile DmaCtrlReadStatus_u status = {.val = DmaCtrlReadStatus_DEFAULT};
    do {
        status.val = READ_TX_CTRL_REG(TX_CTRL_TX_READ_STATUS_REG_OFFSET);
    } while (status.f.read_ready != 1);

    program_dma_read_addresses_(stream, src_gddr, dst_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Non-blocking L1 to GDDR write. Pair with dma_async_write_barrier().
 *
 * @param stream      TX stream (0 or 1).
 * @param src_l1      DRISC L1 source address (32-bit).
 * @param dst_gddr    GDDR destination address (64-bit).
 * @param size_bytes  Transfer size in bytes (must be a multiple of 16).
 */
inline __attribute__((always_inline)) void dma_async_write(
    uint8_t stream, uint32_t src_l1, uint64_t dst_gddr, uint32_t size_bytes) {
    check_stream_(stream);
    check_transfer_size_(size_bytes);

    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.start_of_packet = 0;
    attrs.f.end_of_packet = 0;
    attrs.f.transfer_start_raw = 1;

    volatile DmaCtrlWriteStatus_u status = {.val = DmaCtrlWriteStatus_DEFAULT};
    do {
        status.val = READ_TX_CTRL_REG(TX_CTRL_TX_WRITE_STATUS_REG_OFFSET);
    } while (status.f.write_ready != 1);

    program_dma_write_addresses_(stream, src_l1, dst_gddr);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Wait for all outstanding reads on the given stream to complete.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) void dma_async_read_barrier(uint8_t stream) {
    check_stream_(stream);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.val & DmaTxqStatus_READ_MASK);
}

/**
 * @brief Wait for all outstanding writes on the given stream to complete.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) void dma_async_write_barrier(uint8_t stream) {
    check_stream_(stream);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.val & DmaTxqStatus_WRITE_MASK);
}

/**
 * @brief Wait until the number of outstanding reads on the given stream drops to n or below.
 *
 * n=0 is equivalent to dma_async_read_barrier. Use n=1 with two in-flight reads
 * to implement a ping-pong double-buffer loop: when outstanding drops from 2 to 1
 * the oldest buffer is ready to consume (assumes FIFO completion order).
 *
 * @param stream  TX stream (0 or 1).
 * @param n       Target outstanding read count to wait for.
 */
inline __attribute__((always_inline)) void dma_async_read_wait_n(uint8_t stream, uint8_t n) {
    check_stream_(stream);
    check_outstanding_reads_(n);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.f.num_reads_outstanding > n);
}

/**
 * @brief Wait until the number of outstanding writes on the given stream drops to n or below.
 *
 * n=0 is equivalent to dma_async_write_barrier. Use n=1 with two in-flight writes
 * to implement a ping-pong double-buffer loop: when outstanding drops from 2 to 1
 * the oldest buffer is empty (assumes FIFO completion order).
 *
 * @param stream  TX stream (0 or 1).
 * @param n       Target outstanding write count to wait for.
 */
inline __attribute__((always_inline)) void dma_async_write_wait_n(uint8_t stream, uint8_t n) {
    check_stream_(stream);
    check_outstanding_writes_(n);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.f.num_writes_outstanding > n);
}

/**
 * @brief Return the number of reads currently outstanding on the given stream.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) uint8_t dma_get_reads_outstanding(uint8_t stream) {
    check_stream_(stream);
    DmaTxqStatus_u status;
    status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    return static_cast<uint8_t>(status.f.num_reads_outstanding);
}

/**
 * @brief Return the number of writes currently outstanding on the given stream.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) uint8_t dma_get_writes_outstanding(uint8_t stream) {
    check_stream_(stream);
    DmaTxqStatus_u status;
    status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    return static_cast<uint8_t>(status.f.num_writes_outstanding);
}

#endif  // COMPILE_FOR_DRISC

}  // namespace experimental

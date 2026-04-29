// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/debug/assert.h"
#include "internal/tt-1xx/blackhole/gddr_dma_regs.h"

namespace experimental {

/*
    GDDR DMA API for DRISC kernels — DRISC L1 <-> GDDR transfers.

    Two independent TX streams (0 and 1). Each stream can carry one
    outstanding read and one outstanding write concurrently.

    Typical read:
        dma_async_read(stream, src_gddr, dst_l1, size_bytes);
        dma_async_read_barrier(stream);

    Typical write:
        dma_async_write(stream, src_l1, dst_gddr, size_bytes);
        dma_async_write_barrier(stream);

    Sequential fixed-stride writes (auto-increment):
        dma_auto_incr_enable(stream, base_l1, base_gddr, stride, stride);
        for (int i = 0; i < n; i++) {
            dma_async_write_auto_incr(stream, stride);
            dma_async_write_barrier(stream);
        }
        dma_auto_incr_disable(stream);

    All size and increment parameters are in bytes and must be multiples of 16.
    stream must be 0 or 1.
*/

#ifdef COMPILE_FOR_DRISC

//////////////////////////////////////////////////////////////////
// Internal helpers — not part of the public contract.
//////////////////////////////////////////////////////////////////

static inline __attribute__((always_inline)) void dma_check_stream_(uint8_t stream) {
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
    ASSERT(stream == 0 || stream == 1, DebugAssertTripped);
#endif
}

static inline __attribute__((always_inline)) void dma_check_alignment_(uint32_t size_bytes) {
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
    ASSERT((size_bytes & 0xF) == 0, DebugAssertTripped);
#endif
}

static inline __attribute__((always_inline)) void dma_write_addr_(uint8_t stream, uint32_t src_l1, uint64_t dst_gddr) {
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_TRANSFER_START_ADDR_REG_OFFSET, src_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_DEST_ADDR_LOW_REG_OFFSET, (uint32_t)(dst_gddr & 0xFFFFFFFFu));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_DEST_ADDR_HIGH_REG_OFFSET, (uint32_t)(dst_gddr >> 32));
}

static inline __attribute__((always_inline)) void dma_read_addr_(uint8_t stream, uint64_t src_gddr, uint32_t dst_l1) {
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_LOW_REG_OFFSET, (uint32_t)(src_gddr & 0xFFFFFFFFu));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_SOURCE_HIGH_REG_OFFSET, (uint32_t)(src_gddr >> 32));
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_READ_TRANSFER_DEST_REG_OFFSET, dst_l1);
}

//////////////////////////////////////////////////////////////////
// Public API
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
 * Default is 16 beats (1 KB at 64 B/beat). Higher values reduce AXI transaction
 * overhead for large transfers. Valid range: 1–255.
 *
 * @param burst_size  Beats per AXI burst (1–255).
 */
inline __attribute__((always_inline)) void dma_set_burst_size(uint8_t burst_size = 0x10) {
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
    ASSERT(burst_size > 0 && burst_size <= 255, DebugAssertTripped);
#endif
    DmaCtrlTransferAttrs_u attrs;
    attrs.val = READ_TX_CTRL_REG(TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET);
    attrs.f.max_burst_size = burst_size;
    WRITE_TX_CTRL_REG(TX_CTRL_TX_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Non-blocking GDDR → L1 read. Pair with dma_async_read_barrier().
 *
 * @param stream      TX stream (0 or 1).
 * @param src_gddr    GDDR source address (64-bit).
 * @param dst_l1      DRISC L1 destination address (32-bit).
 * @param size_bytes  Transfer size in bytes (must be a multiple of 16).
 */
inline __attribute__((always_inline)) void dma_async_read(
    uint8_t stream, uint64_t src_gddr, uint32_t dst_l1, uint32_t size_bytes) {
    dma_check_stream_(stream);
    dma_check_alignment_(size_bytes);

    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.transfer_start_read = 1;

    volatile DmaCtrlReadStatus_u status = {.val = DmaCtrlReadStatus_DEFAULT};
    do {
        status.val = READ_TX_CTRL_REG(TX_CTRL_TX_READ_STATUS_REG_OFFSET);
    } while (status.f.read_ready != 1);

    dma_read_addr_(stream, src_gddr, dst_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Non-blocking L1 → GDDR write. Pair with dma_async_write_barrier().
 *
 * @param stream      TX stream (0 or 1).
 * @param src_l1      DRISC L1 source address (32-bit).
 * @param dst_gddr    GDDR destination address (64-bit).
 * @param size_bytes  Transfer size in bytes (must be a multiple of 16).
 */
inline __attribute__((always_inline)) void dma_async_write(
    uint8_t stream, uint32_t src_l1, uint64_t dst_gddr, uint32_t size_bytes) {
    dma_check_stream_(stream);
    dma_check_alignment_(size_bytes);

    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.start_of_packet = 0;
    attrs.f.end_of_packet = 0;
    attrs.f.transfer_start_raw = 1;

    volatile DmaCtrlWriteStatus_u status = {.val = DmaCtrlWriteStatus_DEFAULT};
    do {
        status.val = READ_TX_CTRL_REG(TX_CTRL_TX_WRITE_STATUS_REG_OFFSET);
    } while (status.f.write_ready != 1);

    dma_write_addr_(stream, src_l1, dst_gddr);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

/**
 * @brief Wait for all outstanding reads on the given stream to complete.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) void dma_async_read_barrier(uint8_t stream) {
    dma_check_stream_(stream);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.val & DmaTxqStatus_READ_MASK);
}

/**
 * @brief Wait until the number of outstanding reads on the given stream drops to n or below.
 *
 * n=0 is equivalent to dma_async_read_barrier. Use n=1 with two pre-fired reads
 * to implement a ping-pong double-buffer loop: when outstanding drops from 2 to 1
 * the oldest buffer is ready to consume (assumes FIFO completion order).
 *
 * @param stream  TX stream (0 or 1).
 * @param n       Target outstanding read count to wait for.
 */
inline __attribute__((always_inline)) void dma_async_read_wait_n(uint8_t stream, uint8_t n) {
    dma_check_stream_(stream);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.f.num_reads_outstanding > n);
}

/**
 * @brief Return the number of reads currently outstanding on the given stream.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) uint8_t dma_reads_outstanding(uint8_t stream) {
    dma_check_stream_(stream);
    DmaTxqStatus_u status;
    status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    return static_cast<uint8_t>(status.f.num_reads_outstanding);
}

/**
 * @brief Wait for all outstanding writes on the given stream to complete.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) void dma_async_write_barrier(uint8_t stream) {
    dma_check_stream_(stream);
    volatile DmaTxqStatus_u status;
    do {
        status.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_STATUS_REG_OFFSET);
    } while (status.val & DmaTxqStatus_WRITE_MASK);
}

/**
 * @brief Enable auto-increment mode for sequential fixed-stride writes.
 *
 * Seeds the initial src/dst addresses and configures the per-transfer increment.
 * After each dma_async_write_auto_incr() call the hardware advances both addresses
 * by the configured increments. Call dma_auto_incr_disable() when done.
 *
 * @param stream          TX stream (0 or 1).
 * @param src_l1          Initial DRISC L1 source address (32-bit).
 * @param dst_gddr        Initial GDDR destination address (64-bit).
 * @param src_incr_bytes  Source address increment per transfer (bytes).
 * @param dst_incr_bytes  Destination address increment per transfer (bytes).
 */
inline __attribute__((always_inline)) void dma_auto_incr_enable(
    uint8_t stream, uint32_t src_l1, uint64_t dst_gddr, uint32_t src_incr_bytes, uint32_t dst_incr_bytes) {
    dma_check_stream_(stream);

    DmaTxqSettings_u settings;
    settings.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_TXQ_SETTINGS_REG_OFFSET);
    settings.f.coarse_grain_auto_incr_en = 1;
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TXQ_SETTINGS_REG_OFFSET, settings.val);

    DmaTxqAutoIncrSrcOffset_u src_incr;
    src_incr.f.src_offset = src_incr_bytes;
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_AUTO_INCR_SRC_OFFSET_REG_OFFSET, src_incr.val);

    DmaTxqAutoIncrDstOffset_u dst_incr;
    dst_incr.f.dest_offset = dst_incr_bytes;
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_WRITE_AUTO_INCR_DEST_OFFSET_REG_OFFSET, dst_incr.val);

    // Seed initial addresses so the first fire knows where to start.
    dma_write_addr_(stream, src_l1, dst_gddr);
}

/**
 * @brief Disable auto-increment mode.
 *
 * @param stream  TX stream (0 or 1).
 */
inline __attribute__((always_inline)) void dma_auto_incr_disable(uint8_t stream) {
    dma_check_stream_(stream);
    DmaTxqSettings_u settings;
    settings.val = READ_TX_STREAM_REG(stream, TX_REG_STREAM_TXQ_SETTINGS_REG_OFFSET);
    settings.f.coarse_grain_auto_incr_en = 0;
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TXQ_SETTINGS_REG_OFFSET, settings.val);
}

/**
 * @brief Fire one auto-increment write. Pair with dma_async_write_barrier().
 *
 * The hardware uses its internally tracked address (seeded by dma_auto_incr_enable
 * and advanced after each call). Does not accept addresses — passing them would
 * reset the hardware's position and defeat the auto-increment.
 *
 * @param stream      TX stream (0 or 1).
 * @param size_bytes  Transfer size in bytes (must be a multiple of 16).
 */
inline __attribute__((always_inline)) void dma_async_write_auto_incr(uint8_t stream, uint32_t size_bytes) {
    dma_check_stream_(stream);
    dma_check_alignment_(size_bytes);

    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.start_of_packet = 0;
    attrs.f.end_of_packet = 0;
    attrs.f.transfer_start_raw = 1;
    attrs.f.write_auto_increment_src_addr = 1;
    attrs.f.write_auto_increment_dst_addr = 1;

    volatile DmaCtrlWriteStatus_u status = {.val = DmaCtrlWriteStatus_DEFAULT};
    do {
        status.val = READ_TX_CTRL_REG(TX_CTRL_TX_WRITE_STATUS_REG_OFFSET);
    } while (status.f.write_ready != 1);

    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

#endif  // COMPILE_FOR_DRISC

}  // namespace experimental

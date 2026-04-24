// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Zero-copy host-side handle for an external CMAC port.
//
// Design overview
// ---------------
// erisc_cmac_simple brings the PCS/FEC link up.  Once train_status == 0 the
// host writes a mode-switch magic word to L1:kModeAddr; the firmware (or
// erisc_cmac_gw after reload) enters the data-path loop described below.
//
// TX (host → CMAC) — zero copy
//   1. Host writes raw Ethernet frame bytes directly to TX_BUF0 or TX_BUF1 in
//      erisc L1 via PCIe MMIO (no erisc memcpy involved).
//   2. Host writes frame size to kTxSizeAddr and buffer selector to kTxBufSelAddr.
//   3. Firmware sees kTxSizeAddr != 0, programs ETH_TXQ_TRANSFER_START_ADDR /
//      ETH_TXQ_TRANSFER_SIZE_BYTES, fires ETH_TXQ_CMD_START_RAW, clears kTxSizeAddr.
//   Erisc is in the control path only; data bytes never pass through erisc registers.
//
// RX (CMAC → host) — zero copy, ping-pong double-buffered
//   1. CMAC DMA writes incoming frame bytes into one of two ping-pong buffers
//      (kPacketBuf at 0x4000 or kPacketBuf1 at 0x6C00).
//   2. When a frame completes, firmware re-arms the RX queue to write into the
//      OTHER buffer, then writes kRxBufSelAddr (0 or 1) and kRxWpAddr (word count).
//   3. Host polls kRxWpAddr; when it changes, reads rx_buf_sel() to find which
//      buffer holds the frame, then reads frame bytes directly from that buffer
//      in erisc L1 via PCIe MMIO.
//   4. No ack from the host is required — firmware already switched buffers before
//      publishing, so CMAC DMA fills the next buffer while the host reads.
//   Data bytes travel CMAC DMA → erisc L1 → host PCIe read.  No erisc copy.
//
// L1 memory map (matches erisc_cmac_gw.cpp and boot_params_t layout)
//   0x1000–0x1F40  existing firmware data (boot_params, node_info, debug, results)
//   0x1F40  kTxSizeAddr   — host writes frame size (bytes); erisc clears after TX arm
//   0x1F44  kTxBufSelAddr — host writes 0 or 1 to select TX_BUF0 / TX_BUF1
//   0x1F48  kRxWpAddr     — erisc writes word count of completed frame
//   0x1F4C  kRxRpAddr     — (unused in ping-pong mode; kept for ABI compatibility)
//   0x1F50  kModeAddr     — host writes kModeMagic to switch firmware to data-path loop
//   0x1F54  kRxBufSelAddr — erisc writes 0 or 1: which RX buffer holds the frame
//   0x4000–0x6000  kPacketBuf   (RX DMA buf 0; host reads directly)
//   0x6000–0x6600  kTxBuf0      (host writes TX frame here)
//   0x6600–0x6C00  kTxBuf1      (TX ping-pong partner)
//   0x6C00–0x7C00  kPacketBuf1  (RX DMA buf 1; 4 KB; host reads directly)

#include <cstdint>
#include <span>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/core_coord.hpp>                     // CoreCoord

namespace tt::llrt {

class ExternalIfaceSender {
public:
    ExternalIfaceSender(ChipId chip_id, CoreCoord virtual_eth_core);

    // Poll RESULTS_BUF train_status until 0 (PCS lock), then write kModeMagic.
    // Returns false on timeout (default 5 s).
    bool wait_for_link(uint32_t timeout_ms = 5000);

    // True if PCS lock has been achieved (train_status == 0 in erisc results).
    bool is_link_up() const;

    // Write a raw Ethernet frame (caller-owned) into the next TX buffer slot
    // directly in erisc L1, then arm TX via the mailbox.
    // buf.size() must be <= kMaxFrameBytes.
    // Returns false if the previous TX has not yet been consumed by firmware.
    bool send(std::span<const uint8_t> buf);

    // Returns the word count of the completed RX frame published by firmware.
    // A value different from the last call means a new frame is available.
    // After calling rx_watermark(), use rx_buf_sel() to determine which buffer
    // holds the frame, then call read_rx(rx_buf_sel(), ...) to read it.
    uint32_t rx_watermark() const;

    // Which PACKET_BUF holds the frame signalled by the last rx_watermark() call.
    // 0 → kPacketBuf (0x4000), 1 → kPacketBuf1 (0x6C00).
    // Read kRxBufSelAddr before kRxWpAddr changes again (i.e. before the next
    // rx_watermark() poll iteration overwrites it).
    uint32_t rx_buf_sel() const;

    // Read received frame bytes directly from the specified buffer in erisc L1.
    // buf_sel selects the buffer: 0 → kPacketBuf, 1 → kPacketBuf1.
    // word_offset is the starting word index within the selected buffer.
    // Call after rx_watermark() returns a new value.
    void read_rx(uint32_t buf_sel, uint32_t word_offset, std::span<uint8_t> out) const;

    // DEPRECATED: No longer needed in ping-pong mode.
    // Firmware switches RX buffers before publishing kRxWpAddr, so CMAC DMA
    // fills the next buffer immediately — no host ack is required.
    // Kept for ABI compatibility; calling this is a no-op in practice.
    [[deprecated("rx_consume() is not needed with ping-pong RX buffering")]]
    void rx_consume(uint32_t consumed_wp);

    ChipId chip_id() const { return chip_id_; }
    CoreCoord virtual_core() const { return virtual_core_; }

private:
    ChipId chip_id_;
    CoreCoord virtual_core_;
    uint8_t tx_buf_sel_{0};   // ping-pong index (0 → kTxBuf0, 1 → kTxBuf1)
    uint32_t last_rx_wp_{0};  // last rx_watermark() value seen by host

    // Mailbox layout in erisc L1 (byte addresses)
    static constexpr uint32_t kTxSizeAddr = 0x1F40;
    static constexpr uint32_t kTxBufSelAddr = 0x1F44;
    static constexpr uint32_t kRxWpAddr = 0x1F48;
    static constexpr uint32_t kRxRpAddr = 0x1F4C;  // unused in ping-pong mode; kept for ABI compat
    static constexpr uint32_t kModeAddr = 0x1F50;
    static constexpr uint32_t kRxBufSelAddr = 0x1F54;  // erisc→host: 0=kPacketBuf, 1=kPacketBuf1
    static constexpr uint32_t kModeMagic = 0xDA7ADA7Au;

    // L1 buffer addresses
    static constexpr uint32_t kPacketBuf = 0x4000;   // CMAC RX DMA buf 0 (8 KB)
    static constexpr uint32_t kPacketBuf1 = 0x6C00;  // CMAC RX DMA buf 1 (4 KB)
    static constexpr uint32_t kTxBuf0 = 0x6000;
    static constexpr uint32_t kTxBuf1 = 0x6600;
    static constexpr uint32_t kMaxFrameBytes = 1500;

    // train_status offset within test_results_t at RESULTS_BUF_ADDR (0x1E00)
    static constexpr uint32_t kTrainStatusAddr = 0x1E04;  // results.train_status
};

}  // namespace tt::llrt

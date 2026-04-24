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
// RX (CMAC → host) — zero copy
//   1. CMAC DMA writes incoming frame bytes into PACKET_BUF (erisc L1:kPacketBuf).
//   2. Firmware polls ETH_RXQ_BUF_PTR; when it advances, writes the new value
//      to kRxWpAddr (single 4-byte write per frame).
//   3. Host polls kRxWpAddr; when it differs from last known value, reads frame
//      bytes directly from kPacketBuf in erisc L1 via PCIe.
//   4. Host signals consumed by writing the consumed pointer to kRxRpAddr.
//   Data bytes travel CMAC DMA → erisc L1 → host PCIe read.  No erisc copy.
//
// L1 memory map (matches erisc_cmac_gw.cpp and boot_params_t layout)
//   0x1000–0x1F40  existing firmware data (boot_params, node_info, debug, results)
//   0x1F40  kTxSizeAddr   — host writes frame size (bytes); erisc clears after TX arm
//   0x1F44  kTxBufSelAddr — host writes 0 or 1 to select TX_BUF0 / TX_BUF1
//   0x1F48  kRxWpAddr     — erisc writes ETH_RXQ_BUF_PTR (word units) after each frame
//   0x1F4C  kRxRpAddr     — host writes consumed pointer back; erisc uses to re-arm RX
//   0x1F50  kModeAddr     — host writes kModeMagic to switch firmware to data-path loop
//   0x4000–0x6000  kPacketBuf   (CMAC RX DMA target; host reads directly)
//   0x6000–0x6600  kTxBuf0      (host writes TX frame here)
//   0x6600–0x6C00  kTxBuf1      (ping-pong partner)

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

    // Returns the current ETH_RXQ_BUF_PTR word value published by firmware.
    // Non-zero (and different from last call) means a new frame is available.
    uint32_t rx_watermark() const;

    // Read received frame bytes directly from kPacketBuf in erisc L1.
    // word_offset is the starting word index within PACKET_BUF.
    // Call after rx_watermark() returns a new value.
    void read_rx(uint32_t word_offset, std::span<uint8_t> out) const;

    // Signal to firmware that the host has finished reading the current RX frame.
    // Writes consumed_wp (the rx_watermark() value just processed) to kRxRpAddr.
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
    static constexpr uint32_t kRxRpAddr = 0x1F4C;
    static constexpr uint32_t kModeAddr = 0x1F50;
    static constexpr uint32_t kModeMagic = 0xDA7ADA7Au;

    // L1 buffer addresses
    static constexpr uint32_t kPacketBuf = 0x4000;  // CMAC RX DMA target
    static constexpr uint32_t kTxBuf0 = 0x6000;
    static constexpr uint32_t kTxBuf1 = 0x6600;
    static constexpr uint32_t kMaxFrameBytes = 1500;

    // train_status offset within test_results_t at RESULTS_BUF_ADDR (0x1E00)
    static constexpr uint32_t kTrainStatusAddr = 0x1E04;  // results.train_status
};

}  // namespace tt::llrt

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet firmware ABI constants used when poking ETH cores directly.
//
// Wormhole constants are taken straight from UMD (`umd/device/types/wormhole_eth.hpp`).
// Blackhole ETH FW is driven through an L1 mailbox; UMD does not yet export those
// constants, so we redeclare them here. These values are part of the published
// Blackhole ETH FW ABI (see `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h`
// in the tt-metal tree and the ETH SW APIs wiki page referenced there). They are
// stable and intentionally duplicated to keep tt-ethtool UMD-only.

#pragma once

#include <cstdint>

#include "umd/device/types/wormhole_eth.hpp"

namespace tt_ethtool::eth_fw {

namespace wormhole {
using tt::umd::wormhole::ETH_RETRAIN_ADDR;
using tt::umd::wormhole::ETH_TRAIN_STATUS_ADDR;
using tt::umd::wormhole::ETH_TRIGGER_RETRAIN_VAL;

// Values written to ETH_TRAIN_STATUS_ADDR by ETH FW. Mirrors
// `tt::umd::EthTrainingStatus` in UMD.
inline constexpr std::uint32_t TRAIN_STATUS_IN_PROGRESS = 0;
inline constexpr std::uint32_t TRAIN_STATUS_SUCCESS = 1;
inline constexpr std::uint32_t TRAIN_STATUS_FAIL = 2;
inline constexpr std::uint32_t TRAIN_STATUS_NOT_CONNECTED = 3;
}  // namespace wormhole

namespace blackhole {

// Per-mailbox entry matches `eth_mailbox_t` in eth_fw_api.h.
inline constexpr std::uint32_t ETH_MAILBOX_NUM_ARGS = 3;
inline constexpr std::uint32_t ETH_MAILBOX_ENTRY_SIZE = sizeof(std::uint32_t) * (1 + ETH_MAILBOX_NUM_ARGS);

// Four mailbox slots (HOST, RISC1, CMFW, OTHER) starting at 0x7D000. Host uses slot 0.
inline constexpr std::uint32_t ETH_MAILBOX_BASE_ADDR = 0x7D000;
inline constexpr std::uint32_t ETH_MAILBOX_HOST_MSG_ADDR = ETH_MAILBOX_BASE_ADDR + 0 * ETH_MAILBOX_ENTRY_SIZE;
inline constexpr std::uint32_t ETH_MAILBOX_HOST_ARG0_ADDR = ETH_MAILBOX_HOST_MSG_ADDR + sizeof(std::uint32_t);

// Message word encodes a status nibble (high 16 bits) and a message type (low 16 bits).
inline constexpr std::uint32_t ETH_MSG_STATUS_MASK = 0xFFFF0000u;
inline constexpr std::uint32_t ETH_MSG_CALL = 0xCA110000u;
inline constexpr std::uint32_t ETH_MSG_DONE = 0xD0E50000u;

// Run a link status check. The FW updates the live status counters in
// boot_results.eth_live_status (MAC TX/RX MIB snapshots, FEC corrected /
// uncorrected codewords, retrain count, rx link up, etc.) and optionally
// copies the live status block to the L1 address provided in arg0.
//   arg0: copy_addr (use ETH_LIVE_STATUS_NO_COPY to skip the copy)
//   arg1: unused
//   arg2: unused
inline constexpr std::uint32_t ETH_MSG_TYPE_LINK_STATUS_CHECK = 0x0001u;

// Older name for the same FW message; kept for callers that only care about
// the fast link-up check side-effect.
inline constexpr std::uint32_t ETH_MSG_TYPE_PORT_UP_CHECK = ETH_MSG_TYPE_LINK_STATUS_CHECK;

inline constexpr std::uint32_t ETH_MSG_TYPE_PORT_REINIT_MACPCS = 0x0006u;
inline constexpr std::uint32_t ETH_MSG_TYPE_PORT_ACTION = 0x0009u;

// Sentinel arg0 value for ETH_MSG_LINK_STATUS_CHECK that tells the FW to
// skip the L1 copy step and just refresh the live status counters in place.
inline constexpr std::uint32_t ETH_LIVE_STATUS_NO_COPY = 0xFFFFFFFFu;

// Arguments for ETH_MSG_PORT_ACTION(arg0, _, _)
// See hal.hpp's `FWMailboxMsg::ETH_MSG_PORT_ACTION` doc comment.
inline constexpr std::uint32_t ETH_PORT_ACTION_LINK_UP = 1;
inline constexpr std::uint32_t ETH_PORT_ACTION_LINK_DOWN = 2;

// Arguments for ETH_MSG_PORT_REINIT_MACPCS(retries, reinit_option, _).
// The FW calls eth_port_reinit(retries, reinit_option).
inline constexpr std::uint32_t ETH_PORT_REINIT_ENABLE = 1;
inline constexpr std::uint32_t ETH_PORT_REINIT_OPT_MAC_ONLY = 0;            // Reinit MAC only (SERDES already trained).
inline constexpr std::uint32_t ETH_PORT_REINIT_OPT_MAC_SERDES_RETRAIN = 1;  // Reinit MAC + SERDES via SERDES retrain.
inline constexpr std::uint32_t ETH_PORT_REINIT_OPT_MAC_SERDES = 2;          // Reinit MAC + SERDES from reset.
inline constexpr std::uint32_t ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER = 3;  // Same as 2 but with TX barrier.

// Selected boot_results_t field addresses in L1. See
// `umd/device/types/blackhole_eth.hpp` for the full layout (BOOT_RESULTS_ADDR = 0x7CC00).
inline constexpr std::uint32_t ETH_BOOT_RESULTS_BASE_ADDR = 0x7CC00;
inline constexpr std::uint32_t ETH_PORT_STATUS_ADDR = 0x7CC04;  // eth_status_t.port_status

// eth_live_status_t base and field addresses. Matches the struct layout in
// `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h`.
inline constexpr std::uint32_t ETH_LIVE_STATUS_BASE_ADDR = 0x7CE00;
inline constexpr std::uint32_t ETH_RETRAIN_COUNT_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x00;  // u32
inline constexpr std::uint32_t ETH_RX_LINK_UP_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x04;     // u32

// Snapshot registers (all u64, little-endian on the device).
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_TXD_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x20;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_TXD_OK_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x28;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_TXD_BADFCS_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x30;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_TXD_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x38;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_TXD_OK_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x40;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_TXD_BADFCS_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x48;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_RXD_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x50;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_RXD_OK_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x58;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_RXD_BADFCS_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x60;
inline constexpr std::uint32_t ETH_LIVE_STATUS_FRAMES_RXD_DROPPED_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x68;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_RXD_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x70;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_RXD_OK_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x78;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_RXD_BADFCS_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x80;
inline constexpr std::uint32_t ETH_LIVE_STATUS_BYTES_RXD_DROPPED_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x88;
inline constexpr std::uint32_t ETH_LIVE_STATUS_CORR_CW_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x90;
inline constexpr std::uint32_t ETH_LIVE_STATUS_UNCORR_CW_ADDR = ETH_LIVE_STATUS_BASE_ADDR + 0x98;

// port_status_e values reported in eth_status_t.port_status.
inline constexpr std::uint32_t PORT_STATUS_UNKNOWN = 0;
inline constexpr std::uint32_t PORT_STATUS_UP = 1;
inline constexpr std::uint32_t PORT_STATUS_DOWN = 2;
inline constexpr std::uint32_t PORT_STATUS_UNUSED = 3;

}  // namespace blackhole

}  // namespace tt_ethtool::eth_fw

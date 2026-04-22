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
using tt::umd::wormhole::ETH_TRIGGER_RETRAIN_VAL;
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

inline constexpr std::uint32_t ETH_MSG_TYPE_PORT_REINIT_MACPCS = 0x0006u;
inline constexpr std::uint32_t ETH_MSG_TYPE_PORT_ACTION = 0x0009u;

// Arguments for ETH_MSG_PORT_ACTION(arg0, _, _)
// See hal.hpp's `FWMailboxMsg::ETH_MSG_PORT_ACTION` doc comment.
inline constexpr std::uint32_t ETH_PORT_ACTION_LINK_UP = 1;
inline constexpr std::uint32_t ETH_PORT_ACTION_LINK_DOWN = 2;

// Arguments for ETH_MSG_PORT_REINIT_MACPCS(enable, reinit_option, _)
// reinit_option 2 => reinit MAC + SERDES from reset.
inline constexpr std::uint32_t ETH_PORT_REINIT_ENABLE = 1;
inline constexpr std::uint32_t ETH_PORT_REINIT_OPT_MAC_SERDES = 2;

}  // namespace blackhole

}  // namespace tt_ethtool::eth_fw

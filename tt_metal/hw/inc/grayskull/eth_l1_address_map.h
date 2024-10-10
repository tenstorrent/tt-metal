// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace eth_l1_mem {


struct address_map {
  static constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 0;
  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 0;
  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0;
  static constexpr std::int32_t ERISC_MEM_MAILBOX_BASE = 16;

  static constexpr std::int32_t COMMAND_Q_BASE = 0;
  static constexpr std::int32_t TILE_HEADER_BUFFER_BASE = 0;
  static constexpr std::int32_t ERISC_APP_ROUTING_INFO_SIZE = 0;
  static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 0;
  static constexpr std::int32_t ERISC_APP_ROUTING_INFO_BASE = 0;
  static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = 0;
  static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_BASE = 0;
  static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_SIZE = 0;

  static constexpr std::int32_t ERISC_FIRMWARE_SIZE = 16;
  static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = 0;
  static constexpr std::uint32_t ISSUE_CQ_CB_BASE = 0;
  static constexpr std::uint32_t COMPLETION_CQ_CB_BASE = 0;
  static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = 0;
  static constexpr std::uint32_t FW_VERSION_ADDR = 0;

  static constexpr std::int32_t ERISC_BARRIER_BASE = 0;
  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1;

  static constexpr std::int32_t ERISC_L1_UNRESERVED_SIZE = 0;
  static constexpr std::int32_t ERISC_L1_TUNNEL_BUFFER_SIZE = 0;

  static constexpr std::uint32_t RETRAIN_COUNT_ADDR = 0x1EDC;
  static constexpr std::uint32_t RETRAIN_FORCE_ADDR = 0x1EFC;
};
}  // namespace llk

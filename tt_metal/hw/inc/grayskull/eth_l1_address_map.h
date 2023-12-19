// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace eth_l1_mem {


struct address_map {
  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 0;
  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0;

  static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 0;
  static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = 0;
  static constexpr std::int32_t ERISC_L1_ARG_BASE = 0;

  static constexpr std::int32_t ERISC_FIRMWARE_SIZE = 16;
  static constexpr std::int32_t ERISC_APP_RESERVED_BASE = 0;
  static constexpr std::int32_t ERISC_APP_RESERVED_SIZE = 16;
  static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = 0;
  static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = 0;
  static constexpr std::uint32_t FW_VERSION_ADDR = 0;
  static constexpr std::int32_t PRINT_BUFFER_ER = 0;

  static constexpr std::int32_t ERISC_BARRIER_BASE = 0;
  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1;
};
}  // namespace llk

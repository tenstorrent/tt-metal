#pragma once

#include <stdint.h>

namespace eth_l1_mem {


struct address_map {
  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 0;
  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0;

  static constexpr std::uint32_t FW_VERSION_ADDR = 0;
};
}  // namespace llk

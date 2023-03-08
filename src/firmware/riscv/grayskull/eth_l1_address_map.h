#pragma once

#include <stdint.h>

namespace eth_l1_mem {


struct address_map {
  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 0;
  static constexpr std::int32_t EPOCH_RUNTIME_CONFIG_SIZE = 0;
  static constexpr std::int32_t OVERLAY_BLOB_SIZE = 0;
  static constexpr std::int32_t TILE_HEADER_BUF_SIZE = 0;
  static constexpr std::int32_t FW_L1_BLOCK_SIZE = OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;
  static constexpr std::int32_t FW_DRAM_BLOCK_SIZE = FIRMWARE_SIZE + OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;
  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0;

  static constexpr std::int32_t OVERLAY_BLOB_BASE = FIRMWARE_BASE + FIRMWARE_SIZE;//TRISC2_BASE + TRISC2_SIZE;
  static constexpr std::int32_t EPOCH_RUNTIME_CONFIG_BASE = OVERLAY_BLOB_BASE + OVERLAY_BLOB_SIZE;
  static constexpr std::int32_t DATA_BUFFER_SPACE_BASE = OVERLAY_BLOB_BASE + OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;

  static constexpr std::int32_t MAX_SIZE = 0;
  static constexpr std::uint32_t FW_VERSION_ADDR = 0;
};
}  // namespace llk

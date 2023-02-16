#pragma once

#include <stdint.h>

namespace eth_l1_mem {


struct address_map {
  
  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 20 * 1024;           // 20KB = 8KB + 12KB perf buffers
  static constexpr std::int32_t EPOCH_RUNTIME_CONFIG_SIZE = 128;     //
  static constexpr std::int32_t OVERLAY_BLOB_SIZE = (32 * 1024) - EPOCH_RUNTIME_CONFIG_SIZE;        // 32KB - KERNEL_SCRATCH_SIZE_BYTES = 20KB blob + 12KB epoch - KERNEL_SCRATCH_SIZE_BYTES
  static constexpr std::int32_t TILE_HEADER_BUF_SIZE = 32 * 1024;     // 
  static constexpr std::int32_t FW_L1_BLOCK_SIZE = OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;
  static constexpr std::int32_t FW_DRAM_BLOCK_SIZE = FIRMWARE_SIZE + /*NCRISC_FIRMWARE_SIZE + TRISC0_SIZE + TRISC1_SIZE + TRISC2_SIZE +*/ OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;
  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0x6020;
  static constexpr std::int32_t OVERLAY_BLOB_BASE = 0x12000;
  static constexpr std::int32_t EPOCH_RUNTIME_CONFIG_BASE = OVERLAY_BLOB_BASE + OVERLAY_BLOB_SIZE;
  static constexpr std::int32_t DATA_BUFFER_SPACE_BASE = OVERLAY_BLOB_BASE + OVERLAY_BLOB_SIZE + EPOCH_RUNTIME_CONFIG_SIZE + TILE_HEADER_BUF_SIZE;

template<std::size_t A, std::size_t B> struct TAssertEquality {
  static_assert(A==B, "Not equal");
  static constexpr bool _cResult = (A==B);
};
static constexpr bool _DATA_BUFFER_SPACE_BASE_CORRECT = TAssertEquality<DATA_BUFFER_SPACE_BASE, 0x22000>::_cResult;

  static constexpr std::int32_t MAX_SIZE = 256*1024;
  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1 * 256 * 1024;  
  
  static constexpr std::int32_t RISC_LOCAL_MEM_BASE = 0xffb00000; // Actaul local memory address as seen from risc firmware
                                                                   // As part of the init risc firmware will copy local memory data from
                                                                   // l1 locations listed above into internal local memory that starts 
                                                                   // at RISC_LOCAL_MEM_BASE address

  static constexpr std::uint32_t FW_VERSION_ADDR = 0x210;
};
}  // namespace llk


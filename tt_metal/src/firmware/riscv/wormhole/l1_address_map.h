#pragma once

#include <cstdint>
namespace l1_mem {

// l1_mem:address_map::TRISC0_BASE

struct address_map {

  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 20 * 1024;          // 20KB = 7KB + 1KB zeros + 12KB perf buffers
  static constexpr std::int32_t BRISC_FIRMWARE_SIZE = 7*1024 + 512;
  static constexpr std::int32_t ZEROS_SIZE = 512;
  static constexpr std::int32_t NCRISC_FIRMWARE_SIZE = 32 * 1024;        // 16KB in L0, 16KB in L1
  static constexpr std::int32_t TRISC0_SIZE = 20 * 1024;        // 20KB = 16KB + 4KB local memory
  static constexpr std::int32_t TRISC1_SIZE = 16 * 1024;        // 16KB = 12KB + 4KB local memory
  static constexpr std::int32_t TRISC2_SIZE = 20 * 1024;        // 20KB = 16KB + 4KB local memory
  static constexpr std::int32_t TRISC_LOCAL_MEM_SIZE = 4 * 1024;      //
  static constexpr std::int32_t NCRISC_LOCAL_MEM_SIZE = 4 * 1024;     //
  static constexpr std::int32_t NCRISC_L1_CODE_SIZE = 16*1024;      // Size of code block that is L1 resident
  static constexpr std::int32_t NCRISC_IRAM_CODE_SIZE = 16*1024;    // Size of code block that is IRAM resident
  static constexpr std::int32_t NCRISC_DATA_SIZE = 4 * 1024;        // 4KB

  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0;
  static constexpr std::int32_t ZEROS_BASE = FIRMWARE_BASE + BRISC_FIRMWARE_SIZE;
  static constexpr std::int32_t NCRISC_FIRMWARE_BASE = FIRMWARE_BASE + FIRMWARE_SIZE;
  static constexpr std::int32_t NCRISC_L1_CODE_BASE =  NCRISC_FIRMWARE_BASE + NCRISC_IRAM_CODE_SIZE;
  static constexpr std::int32_t NCRISC_LOCAL_MEM_BASE = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE - NCRISC_LOCAL_MEM_SIZE; // Copy of the local memory

  static constexpr std::int32_t TRISC_BASE = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE;
  static constexpr std::int32_t TRISC0_BASE = NCRISC_FIRMWARE_BASE + NCRISC_FIRMWARE_SIZE;
  static constexpr std::int32_t TRISC0_LOCAL_MEM_BASE = TRISC0_BASE + TRISC0_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t TRISC1_BASE = TRISC0_BASE + TRISC0_SIZE;
  static constexpr std::int32_t TRISC1_LOCAL_MEM_BASE = TRISC1_BASE + TRISC1_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t TRISC2_BASE = TRISC1_BASE + TRISC1_SIZE;
  static constexpr std::int32_t TRISC2_LOCAL_MEM_BASE = TRISC2_BASE + TRISC2_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t DATA_BUFFER_SPACE_BASE = TRISC2_BASE + TRISC2_SIZE;
  static constexpr std::int32_t BRISC_LOCAL_MEM_BASE = DATA_BUFFER_SPACE_BASE; // Only used during init.

  // Upper 2KB of local space is used as debug buffer
  static constexpr std::int32_t DEBUG_BUFFER_SIZE  = 2 * 1024;
  static constexpr std::int32_t TRISC0_DEBUG_BUFFER_BASE  = TRISC0_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;
  static constexpr std::int32_t TRISC1_DEBUG_BUFFER_BASE  = TRISC1_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;
  static constexpr std::int32_t TRISC2_DEBUG_BUFFER_BASE  = TRISC2_LOCAL_MEM_BASE + DEBUG_BUFFER_SIZE;

  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1464 * 1024; // 1464 KB

  // Perf buffer (FIXME - update once location of the perf data buffer is finalized)
  static constexpr std::int32_t PERF_BUF_SIZE = FIRMWARE_SIZE - BRISC_FIRMWARE_SIZE - ZEROS_SIZE;
  static constexpr std::int32_t TRISC_PERF_BUF_SIZE_LEVEL_0 = 640; // smaller buffer size for limited logging
  static constexpr std::int32_t NCRISC_PERF_BUF_SIZE_LEVEL_0 = 640; // smaller buffer size for limited logging
  static constexpr std::int32_t TRISC_PERF_BUF_SIZE_LEVEL_1 = 4*1024; // PERF_BUF_SIZE/3
  static constexpr std::int32_t NCRISC_PERF_BUF_SIZE_LEVEL_1 = 4*1024; // NCRISC performance buffer
  static constexpr std::int32_t PERF_BUF_BASE_ADDR = FIRMWARE_BASE + BRISC_FIRMWARE_SIZE + ZEROS_SIZE;   // 12KB

};
}  // namespace llk

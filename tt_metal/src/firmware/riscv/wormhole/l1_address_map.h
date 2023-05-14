#pragma once

#include <cstdint>
#include "dev_mem_map.h"

namespace l1_mem {

// l1_mem:address_map::TRISC0_BASE

struct address_map {

  // Sizes
  static constexpr std::int32_t ZEROS_SIZE = 512;
  static constexpr std::int32_t TRISC_LOCAL_MEM_SIZE = 4 * 1024;      //
  static constexpr std::int32_t NCRISC_LOCAL_MEM_SIZE = 4 * 1024;     //
  static constexpr std::int32_t NCRISC_L1_CODE_SIZE = 16*1024;      // Size of code block that is L1 resident
  static constexpr std::int32_t NCRISC_IRAM_CODE_SIZE = 16*1024;    // Size of code block that is IRAM resident
  static constexpr std::int32_t NCRISC_DATA_SIZE = 4 * 1024;        // 4KB

  // Base addresses
  static constexpr std::int32_t ZEROS_BASE = MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_CODE_SIZE;
  static constexpr std::int32_t NCRISC_L1_CODE_BASE =  MEM_NCRISC_FIRMWARE_BASE + NCRISC_IRAM_CODE_SIZE;
  static constexpr std::int32_t NCRISC_LOCAL_MEM_BASE = MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE - NCRISC_LOCAL_MEM_SIZE; // Copy of the local memory

  static constexpr std::int32_t TRISC0_LOCAL_MEM_BASE = MEM_TRISC0_BASE + MEM_TRISC0_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t TRISC1_LOCAL_MEM_BASE = MEM_TRISC1_BASE + MEM_TRISC1_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t TRISC2_LOCAL_MEM_BASE = MEM_TRISC2_BASE + MEM_TRISC2_SIZE - TRISC_LOCAL_MEM_SIZE; // Copy of the local memory
  static constexpr std::int32_t DATA_BUFFER_SPACE_BASE = MEM_TRISC2_BASE + MEM_TRISC2_SIZE;
  static constexpr std::int32_t BRISC_LOCAL_MEM_BASE = DATA_BUFFER_SPACE_BASE; // Only used during init.

  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1464 * 1024; // 1464 KB

  // Perf buffer (FIXME - update once location of the perf data buffer is finalized)
  // XXXX pk - put this in brisc.cc as a global variable with exported symbols for other kernels
  static constexpr std::int32_t PERF_BUF_SIZE = MEM_BRISC_FIRMWARE_SIZE - MEM_BRISC_FIRMWARE_CODE_SIZE - ZEROS_SIZE;
  static constexpr std::int32_t TRISC_PERF_BUF_SIZE_LEVEL_0 = 640; // smaller buffer size for limited logging
  static constexpr std::int32_t NCRISC_PERF_BUF_SIZE_LEVEL_0 = 640; // smaller buffer size for limited logging
  static constexpr std::int32_t TRISC_PERF_BUF_SIZE_LEVEL_1 = 4*1024; // PERF_BUF_SIZE/3
  static constexpr std::int32_t NCRISC_PERF_BUF_SIZE_LEVEL_1 = 4*1024; // NCRISC performance buffer
  static constexpr std::int32_t PERF_BUF_BASE_ADDR = MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_CODE_SIZE + ZEROS_SIZE;   // 12KB

};
}  // namespace llk

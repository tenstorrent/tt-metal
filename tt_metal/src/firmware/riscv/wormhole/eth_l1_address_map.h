#pragma once

#include <stdint.h>

namespace eth_l1_mem {


struct address_map {

  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 20 * 1024;           // 20KB = 8KB + 12KB perf buffers

  // Base addresses
  static constexpr std::int32_t FIRMWARE_BASE = 0x6020;

  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1 * 256 * 1024;

  static constexpr std::int32_t RISC_LOCAL_MEM_BASE = 0xffb00000; // Actaul local memory address as seen from risc firmware
                                                                   // As part of the init risc firmware will copy local memory data from
                                                                   // l1 locations listed above into internal local memory that starts
                                                                   // at RISC_LOCAL_MEM_BASE address

  static constexpr std::uint32_t FW_VERSION_ADDR = 0x210;
};
}  // namespace llk

#pragma once

#include <stdint.h>

#include "l1_address_map.h"

namespace dram_mem {

struct address_map {
    // Sizes

    // Actual memory allocated to each bank for perf is 39896 * 1024.
    // It is reduced below for faster dram perf dump.
    // This can be increased to maximum 39896 * 1024 if more space was needed.
    static constexpr std::int32_t DRAM_EACH_BANK_PERF_BUFFER_SIZE = 4800 * 1024;
    static constexpr std::int32_t FW_DRAM_BLOCK_SIZE =
        l1_mem::address_map::TRISC0_SIZE + l1_mem::address_map::TRISC1_SIZE + l1_mem::address_map::TRISC2_SIZE;

    // Ensure values are in sync until l1_mem::address_map::FW_DRAM_BLOCK_SIZE is retired
    static_assert(l1_mem::address_map::FW_DRAM_BLOCK_SIZE == FW_DRAM_BLOCK_SIZE);

    // Base addresses

    static constexpr std::int32_t DRAM_EACH_BANK_PERF_BUFFER_BASE = 1024 * 1024;

    static constexpr std::int32_t TRISC_BASE = 0;
    static constexpr std::int32_t TRISC0_BASE = TRISC_BASE;
    static constexpr std::int32_t TRISC1_BASE = TRISC0_BASE + l1_mem::address_map::TRISC0_SIZE;
    static constexpr std::int32_t TRISC2_BASE = TRISC1_BASE + l1_mem::address_map::TRISC1_SIZE;
    static constexpr std::int32_t OVERLAY_BLOB_BASE = TRISC2_BASE + l1_mem::address_map::TRISC2_SIZE;
    static constexpr std::int32_t EPOCH_RUNTIME_CONFIG_BASE =
        OVERLAY_BLOB_BASE + l1_mem::address_map::OVERLAY_BLOB_SIZE;

    static_assert((TRISC0_BASE + l1_mem::address_map::TRISC0_SIZE) < FW_DRAM_BLOCK_SIZE);
    static_assert((TRISC1_BASE + l1_mem::address_map::TRISC1_SIZE) < FW_DRAM_BLOCK_SIZE);
    static_assert((TRISC2_BASE + l1_mem::address_map::TRISC2_SIZE) < FW_DRAM_BLOCK_SIZE);
    static_assert((OVERLAY_BLOB_BASE + l1_mem::address_map::OVERLAY_BLOB_SIZE) < FW_DRAM_BLOCK_SIZE);
    static_assert((EPOCH_RUNTIME_CONFIG_BASE + l1_mem::address_map::EPOCH_RUNTIME_CONFIG_SIZE) < FW_DRAM_BLOCK_SIZE);
};
}  // namespace dram_mem

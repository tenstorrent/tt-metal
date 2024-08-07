// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the TT Hardware Abstraction Layer interface
// This layer abstracts which TT chip is running from the higher
// level APIs
//

#include <cstdint>
#include <vector>
#include <memory>
#include "tt_metal/common/assert.hpp"

enum class CoreType;

namespace tt {

enum class ARCH;

namespace tt_metal {

enum class HalDispatchCoreType {
    HAL_CORE_TYPE_TENSIX,
    HAL_CORE_TYPE_ACTIVE_ETH,
    HAL_CORE_TYPE_IDLE_ETH,
};

enum HalMemAddrType : uint8_t {
    HAL_L1_MEM_ADDR_BARRIER = 0,
    HAL_L1_MEM_ADDR_LAUNCH = 1,
    HAL_L1_MEM_ADDR_WATCHER = 2,
    HAL_L1_MEM_ADDR_DPRINT = 3,
    HAL_L1_MEM_ADDR_PROFILER = 4,
    HAL_L1_MEM_ADDR_KERNEL_CONFIG_BASE = 5,
    HAL_L1_MEM_ADDR_UNRESERVED_BASE = 6,
    HAL_L1_MEM_ADDR_COUNT = 7
};

using DeviceAddr = std::uint64_t;

// Core information instanced once per core type
class HalCoreInfoType {
  private:
    HalDispatchCoreType dispatch_core_type_;
    CoreType core_type_;
    std::uint32_t proc_count_; // eventually a vector of attributes?
    std::vector<DeviceAddr> mem_map_;

  public:
    HalCoreInfoType(HalDispatchCoreType dispatch_core_type, CoreType core_type,
        uint32_t core_proc_count, const std::vector<DeviceAddr>& mem_map);

    uint32_t get_processor_count();
    DeviceAddr get_dev_addr(HalMemAddrType addr_type);
};

inline DeviceAddr HalCoreInfoType::get_dev_addr(HalMemAddrType addr_type) {
    uint32_t index = static_cast<std::underlying_type<HalMemAddrType>::type>(addr_type);
    TT_ASSERT(index < this->mem_map_.size());
    return this->mem_map_[index];
}

class Hal {
  private:
    std::mutex lock;
    bool initialized_;
    std::vector<HalCoreInfoType> core_info_;

    void initialize_gs();
    void initialize_wh();
    void initialize_bh();

  public:
    Hal();

    void initialize(tt::ARCH arch);

    uint32_t get_core_type_count();
    HalDispatchCoreType get_dispatch_core_type(uint32_t core_type_index);
    CoreType get_core_type(uint32_t core_type_index);

    uint32_t get_processor_count(uint32_t core_type_index);

    DeviceAddr get_dev_addr(HalDispatchCoreType dispatch_core_type, HalMemAddrType addr_type);
};

inline DeviceAddr Hal::get_dev_addr(HalDispatchCoreType dispatch_core_type, HalMemAddrType addr_type) {
    uint32_t index = static_cast<std::underlying_type<HalDispatchCoreType>::type>(dispatch_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_addr(addr_type);
}

extern Hal hal;

}  // namespace tt_metal
}  // namespace tt

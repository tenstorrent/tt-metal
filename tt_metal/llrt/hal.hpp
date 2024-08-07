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

enum class HalProgrammableCoreType {
    TENSIX,
    ACTIVE_ETH,
    IDLE_ETH,
};

enum class HalMemAddrType : uint8_t {
    BARRIER = 0,
    LAUNCH = 1,
    WATCHER = 2,
    DPRINT = 3,
    PROFILER = 4,
    KERNEL_CONFIG_BASE = 5,
    UNRESERVED_BASE = 6,
    COUNT = 7
};

using DeviceAddr = std::uint64_t;

// Core information instanced once per core type
class HalCoreInfoType {
  private:
    HalProgrammableCoreType programmable_core_type_;
    CoreType core_type_;
    std::uint32_t proc_count_; // eventually a vector of attributes?
    std::vector<DeviceAddr> mem_map_;

  public:
    HalCoreInfoType(HalProgrammableCoreType programmable_core_type, CoreType core_type,
        uint32_t core_proc_count, const std::vector<DeviceAddr>& mem_map);

    uint32_t get_processor_count();
    template <typename T = DeviceAddr>
    T get_dev_addr(HalMemAddrType addr_type);
};

template <typename T>
inline T HalCoreInfoType::get_dev_addr(HalMemAddrType addr_type) {
    uint32_t index = static_cast<std::underlying_type<HalMemAddrType>::type>(addr_type);
    TT_ASSERT(index < this->mem_map_.size());
    return reinterpret_cast<T>(this->mem_map_[index]);
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
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index);
    CoreType get_core_type(uint32_t core_type_index);

    uint32_t get_processor_count(uint32_t core_type_index);

    template <typename T = DeviceAddr>
    T get_dev_addr(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type);
};

template <typename T>
inline T Hal::get_dev_addr(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type) {
    uint32_t index = static_cast<std::underlying_type<HalProgrammableCoreType>::type>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_addr<T>(addr_type);
}

extern Hal hal;

}  // namespace tt_metal
}  // namespace tt

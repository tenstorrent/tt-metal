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
    TENSIX     = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH   = 2,
    COUNT      = 3
};

enum class HalMemAddrType : uint8_t {
    BARRIER = 0,
    LAUNCH = 1,
    WATCHER = 2,
    DPRINT = 3,
    PROFILER = 4,
    KERNEL_CONFIG = 5,
    UNRESERVED = 6,
    COUNT = 7
};

using DeviceAddr = std::uint64_t;

class Hal;

// Core information instanced once per core type
class HalCoreInfoType {
    friend class Hal;

  private:
    HalProgrammableCoreType programmable_core_type_;
    CoreType core_type_;
    std::uint32_t proc_count_; // eventually a vector of attributes?
    std::vector<DeviceAddr> mem_map_bases_;
    std::vector<uint32_t> mem_map_sizes_;

  public:
    HalCoreInfoType(HalProgrammableCoreType programmable_core_type, CoreType core_type, uint32_t core_proc_count,
        const std::vector<DeviceAddr>& mem_map_bases, const std::vector<uint32_t>& mem_map_sizes);

    template <typename T = DeviceAddr>
    T get_dev_addr(HalMemAddrType addr_type) const;
    uint32_t get_dev_size(HalMemAddrType addr_type) const;
};

template <typename T>
inline T HalCoreInfoType::get_dev_addr(HalMemAddrType addr_type) const {
    uint32_t index = static_cast<std::underlying_type<HalMemAddrType>::type>(addr_type);
    TT_ASSERT(index < this->mem_map_bases_.size());
    return reinterpret_cast<T>(this->mem_map_bases_[index]);
}

inline uint32_t HalCoreInfoType::get_dev_size(HalMemAddrType addr_type) const {
    uint32_t index = static_cast<std::underlying_type<HalMemAddrType>::type>(addr_type);
    TT_ASSERT(index < this->mem_map_sizes_.size());
    return this->mem_map_sizes_[index];
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

    uint32_t get_programmable_core_type_count() const;
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index) const;
    uint32_t get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const;
    CoreType get_core_type(uint32_t programmable_core_type_index) const;

    uint32_t get_processor_count(uint32_t core_type_index) const;

    template <typename T = DeviceAddr>
    T get_dev_addr(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type) const;
    template <typename T = DeviceAddr>
    T get_dev_addr(uint32_t programmable_core_type_index, HalMemAddrType addr_type) const;
    uint32_t get_dev_size(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type) const;
};

inline uint32_t Hal::get_programmable_core_type_count() const {
    return core_info_.size();
}

inline HalProgrammableCoreType Hal::get_programmable_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].programmable_core_type_;
}

inline CoreType Hal::get_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].core_type_;
}

template <typename T>
inline T Hal::get_dev_addr(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type) const {
    uint32_t index = static_cast<std::underlying_type<HalProgrammableCoreType>::type>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_addr<T>(addr_type);
}

template <typename T>
inline T Hal::get_dev_addr(uint32_t programmable_core_type_index, HalMemAddrType addr_type) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_dev_addr<T>(addr_type);
}

inline uint32_t Hal::get_dev_size(HalProgrammableCoreType programmable_core_type, HalMemAddrType addr_type) const {
    uint32_t index = static_cast<std::underlying_type<HalProgrammableCoreType>::type>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_size(addr_type);
}

extern Hal hal;

}  // namespace tt_metal
}  // namespace tt

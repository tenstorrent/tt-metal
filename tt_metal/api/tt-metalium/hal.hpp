// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the TT Hardware Abstraction Layer interface
// This layer abstracts which TT chip is running from the higher
// level APIs
//

#include <cstdint>
#include <functional>
#include <variant>
#include <vector>
#include <memory>
#include "assert.hpp"
#include "utils.hpp"

enum class CoreType;

namespace tt {

enum class ARCH;

namespace tt_metal {

enum class HalProgrammableCoreType { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2, COUNT = 3 };

static constexpr uint32_t NumHalProgrammableCoreTypes = static_cast<uint32_t>(HalProgrammableCoreType::COUNT);

enum class HalProcessorClassType : uint8_t {
    DM = 0,
    // Setting this to 2 because we currently treat brisc and ncrisc as two unique processor classes on Tensix
    // TODO: Uplift view of Tensix processor classes to be 1 DM class with 2 processor types
    COMPUTE = 2
};

enum class HalL1MemAddrType : uint8_t {
    BASE,
    BARRIER,
    MAILBOX,
    LAUNCH,
    WATCHER,
    DPRINT,
    PROFILER,
    KERNEL_CONFIG,
    UNRESERVED,
    CORE_INFO,
    GO_MSG,
    LAUNCH_MSG_BUFFER_RD_PTR,
    LOCAL,
    BANK_TO_NOC_SCRATCH,
    APP_SYNC_INFO,
    TILE_HEADER_BUFFER,
    APP_ROUTING_INFO,
    RETRAIN_COUNT,
    FABRIC_ROUTER_CONFIG,
    COUNT  // Keep this last so it always indicates number of enum options
};

enum class HalDramMemAddrType : uint8_t { DRAM_BARRIER = 0, COUNT = 1 };

enum class HalMemType : uint8_t { L1 = 0, DRAM = 1, HOST = 2, COUNT = 3 };

using DeviceAddr = std::uint64_t;

// Note: nsidwell will be removing need for fw_base_addr and local_init_addr
// fw_launch_addr is programmed with fw_launch_addr_value on the master risc
// of a given progammable core to start FW.
// fw_launch_addr_value will be a jump instruction to FW or the address of FW
struct HalJitBuildConfig {
    DeviceAddr fw_base_addr;
    DeviceAddr local_init_addr;
    DeviceAddr fw_launch_addr;
    uint32_t fw_launch_addr_value;
};

class Hal;

// Core information instanced once per core type
class HalCoreInfoType {
    friend class Hal;

private:
    HalProgrammableCoreType programmable_core_type_;
    CoreType core_type_;
    // indices represents processor class and type positions, value is build configuration params
    std::vector<std::vector<HalJitBuildConfig>> processor_classes_;
    std::vector<DeviceAddr> mem_map_bases_;
    std::vector<uint32_t> mem_map_sizes_;
    bool supports_cbs_;

public:
    HalCoreInfoType(
        HalProgrammableCoreType programmable_core_type,
        CoreType core_type,
        const std::vector<std::vector<HalJitBuildConfig>>& processor_classes,
        const std::vector<DeviceAddr>& mem_map_bases,
        const std::vector<uint32_t>& mem_map_sizes,
        bool supports_cbs);

    template <typename T = DeviceAddr>
    T get_dev_addr(HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalL1MemAddrType addr_type) const;
    uint32_t get_processor_classes_count() const;
    uint32_t get_processor_types_count(uint32_t processor_class_idx) const;
    const HalJitBuildConfig &get_jit_build_config(uint32_t processor_class_idx, uint32_t processor_type_idx) const;
};

template <typename T>
inline T HalCoreInfoType::get_dev_addr(HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_bases_.size());
    return reinterpret_cast<T>(this->mem_map_bases_[index]);
}

inline uint32_t HalCoreInfoType::get_dev_size(HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_sizes_.size());
    return this->mem_map_sizes_[index];
}

inline uint32_t HalCoreInfoType::get_processor_classes_count() const { return this->processor_classes_.size(); }

inline uint32_t HalCoreInfoType::get_processor_types_count(uint32_t processor_class_idx) const {
    TT_ASSERT(processor_class_idx < this->processor_classes_.size());
    return this->processor_classes_[processor_class_idx].size();
}

inline const HalJitBuildConfig &HalCoreInfoType::get_jit_build_config(uint32_t processor_class_idx, uint32_t processor_type_idx) const {
    TT_ASSERT(processor_class_idx < this->processor_classes_.size());
    TT_ASSERT(processor_type_idx < this->processor_classes_[processor_class_idx].size());
    return this->processor_classes_[processor_class_idx][processor_type_idx];
}

class Hal {
public:
    using RelocateFunc = std::function<uint64_t(uint64_t, uint64_t)>;
    using IramRelocateFunc = std::function<uint64_t(uint64_t)>;
    using ValidRegAddrFunc = std::function<bool(uint32_t)>;
    using NOCXYEncodingFunc = std::function<uint32_t(uint32_t, uint32_t)>;
    using NOCMulticastEncodingFunc = std::function<uint32_t(uint32_t, uint32_t, uint32_t, uint32_t)>;
    using NOCAddrFunc = std::function<uint64_t(uint64_t)>;
    using StackSizeFunc = std::function<uint32_t(uint32_t)>;

private:
    tt::ARCH arch_;
    std::vector<HalCoreInfoType> core_info_;
    std::vector<DeviceAddr> dram_bases_;
    std::vector<uint32_t> dram_sizes_;
    std::vector<uint32_t> mem_alignments_;
    std::vector<uint32_t> mem_alignments_with_pcie_;
    uint32_t num_nocs_;
    uint32_t noc_addr_node_id_bits_;
    uint32_t noc_coord_reg_offset_;
    uint32_t noc_overlay_start_addr_;
    uint32_t noc_stream_reg_space_size_;
    uint32_t noc_stream_remote_dest_buf_size_reg_index_;
    uint32_t noc_stream_remote_dest_buf_start_reg_index_;
    uint32_t noc_stream_remote_dest_buf_space_available_reg_index_;
    uint32_t noc_stream_remote_dest_buf_space_available_update_reg_index_;
    bool coordinate_virtualization_enabled_;
    uint32_t virtual_worker_start_x_;
    uint32_t virtual_worker_start_y_;
    bool eth_fw_is_cooperative_ = false;  // set when eth riscs have to context switch

    float eps_ = 0.0f;
    float nan_ = 0.0f;
    float inf_ = 0.0f;

    void initialize_wh();
    void initialize_bh();

    // Functions where implementation varies by architecture
    RelocateFunc relocate_func_;
    IramRelocateFunc erisc_iram_relocate_func_;
    ValidRegAddrFunc valid_reg_addr_func_;
    NOCXYEncodingFunc noc_xy_encoding_func_;
    NOCMulticastEncodingFunc noc_multicast_encoding_func_;
    NOCAddrFunc noc_mcast_addr_start_x_func_;
    NOCAddrFunc noc_mcast_addr_start_y_func_;
    NOCAddrFunc noc_mcast_addr_end_x_func_;
    NOCAddrFunc noc_mcast_addr_end_y_func_;
    NOCAddrFunc noc_ucast_addr_x_func_;
    NOCAddrFunc noc_ucast_addr_y_func_;
    NOCAddrFunc noc_local_addr_func_;
    StackSizeFunc stack_size_func_;

public:
    Hal();

    tt::ARCH get_arch() const { return arch_; }

    uint32_t get_num_nocs() const { return num_nocs_; }
    uint32_t get_noc_addr_node_id_bits() const { return noc_addr_node_id_bits_; }
    uint32_t get_noc_coord_reg_offset() const { return noc_coord_reg_offset_; }

    uint32_t get_noc_overlay_start_addr() const { return noc_overlay_start_addr_; }
    uint32_t get_noc_stream_reg_space_size() const { return noc_stream_reg_space_size_; }
    uint32_t get_noc_stream_remote_dest_buf_size_reg_index() const {
        return noc_stream_remote_dest_buf_size_reg_index_;
    }
    uint32_t get_noc_stream_remote_dest_buf_start_reg_index() const {
        return noc_stream_remote_dest_buf_start_reg_index_;
    }
    uint32_t get_noc_stream_remote_dest_buf_space_available_reg_index() const {
        return noc_stream_remote_dest_buf_space_available_reg_index_;
    }
    uint32_t get_noc_stream_remote_dest_buf_space_available_update_reg_index() const {
        return noc_stream_remote_dest_buf_space_available_update_reg_index_;
    }

    float get_eps() const { return eps_; }
    float get_nan() const { return nan_; }
    float get_inf() const { return inf_; }

    template <typename IndexType, typename SizeType, typename CoordType>
    auto noc_coordinate(IndexType noc_index, SizeType noc_size, CoordType coord) const
        -> decltype(noc_size - 1 - coord) {
        return noc_index == 0 ? coord : (noc_size - 1 - coord);
    }

    uint32_t noc_xy_encoding(uint32_t x, uint32_t y) const { return noc_xy_encoding_func_(x, y); }
    uint32_t noc_multicast_encoding(uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) const {
        return noc_multicast_encoding_func_(x_start, y_start, x_end, y_end);
    }

    uint64_t get_noc_mcast_addr_start_x(uint64_t addr) const { return noc_mcast_addr_start_x_func_(addr); }
    uint64_t get_noc_mcast_addr_start_y(uint64_t addr) const { return noc_mcast_addr_start_y_func_(addr); }
    uint64_t get_noc_mcast_addr_end_x(uint64_t addr) const { return noc_mcast_addr_end_x_func_(addr); }
    uint64_t get_noc_mcast_addr_end_y(uint64_t addr) const { return noc_mcast_addr_end_y_func_(addr); }
    uint64_t get_noc_ucast_addr_x(uint64_t addr) const { return noc_ucast_addr_x_func_(addr); }
    uint64_t get_noc_ucast_addr_y(uint64_t addr) const { return noc_ucast_addr_y_func_(addr); }
    uint64_t get_noc_local_addr(uint64_t addr) const { return noc_local_addr_func_(addr); }

    bool is_coordinate_virtualization_enabled() const { return this->coordinate_virtualization_enabled_; };
    std::uint32_t get_virtual_worker_start_x() const { return this->virtual_worker_start_x_; }
    std::uint32_t get_virtual_worker_start_y() const { return this->virtual_worker_start_y_; }
    bool get_eth_fw_is_cooperative() const { return this->eth_fw_is_cooperative_; }
    uint32_t get_programmable_core_type_count() const;
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index) const;
    uint32_t get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const;
    CoreType get_core_type(uint32_t programmable_core_type_index) const;
    uint32_t get_processor_classes_count(std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type) const;
    uint32_t get_processor_class_type_index(HalProcessorClassType processor_class);
    uint32_t get_processor_types_count(
        std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type, uint32_t processor_class_idx) const;

    template <typename T = DeviceAddr>
    T get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;
    template <typename T = DeviceAddr>
    T get_dev_addr(uint32_t programmable_core_type_index, HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;

    // Overloads for Dram Params
    template <typename T = DeviceAddr>
    T get_dev_addr(HalDramMemAddrType addr_type) const;
    uint32_t get_dev_size(HalDramMemAddrType addr_type) const;

    uint32_t get_alignment(HalMemType memory_type) const;
    // Returns an alignment that is aligned with PCIE and the given memory type
    uint32_t get_common_alignment_with_pcie(HalMemType memory_type) const;

    bool get_supports_cbs(uint32_t programmable_core_type_index) const;

    uint32_t get_num_risc_processors() const;

    const HalJitBuildConfig &get_jit_build_config(
        uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const;

    uint64_t relocate_dev_addr(uint64_t addr, uint64_t local_init_addr = 0) {
        return relocate_func_(addr, local_init_addr);
    }

    uint64_t erisc_iram_relocate_dev_addr(uint64_t addr) { return erisc_iram_relocate_func_(addr); }

    uint32_t valid_reg_addr(uint32_t addr) { return valid_reg_addr_func_(addr); }

    uint32_t get_stack_size(uint32_t type) { return stack_size_func_(type); }
};

inline uint32_t Hal::get_programmable_core_type_count() const { return core_info_.size(); }

inline uint32_t Hal::get_processor_classes_count(
    std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type) const {
    return std::visit(
        [&](auto&& core_type_specifier) -> uint32_t {
            using T = std::decay_t<decltype(core_type_specifier)>;
            uint32_t index = this->core_info_.size();
            if constexpr (std::is_same_v<T, HalProgrammableCoreType>) {
                index = utils::underlying_type<HalProgrammableCoreType>(core_type_specifier);
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                index = core_type_specifier;
            }
            TT_ASSERT(index < this->core_info_.size());
            return this->core_info_[index].get_processor_classes_count();
        },
        programmable_core_type);
}

inline uint32_t Hal::get_processor_types_count(
    std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type, uint32_t processor_class_idx) const {
    return std::visit(
        [&](auto&& core_type_specifier) -> uint32_t {
            using T = std::decay_t<decltype(core_type_specifier)>;
            uint32_t index = this->core_info_.size();
            if constexpr (std::is_same_v<T, HalProgrammableCoreType>) {
                index = utils::underlying_type<HalProgrammableCoreType>(core_type_specifier);
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                index = core_type_specifier;
            }
            TT_ASSERT(index < this->core_info_.size());
            return this->core_info_[index].get_processor_types_count(processor_class_idx);
        },
        programmable_core_type);
}

inline HalProgrammableCoreType Hal::get_programmable_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].programmable_core_type_;
}

inline CoreType Hal::get_core_type(uint32_t core_type_index) const { return core_info_[core_type_index].core_type_; }

template <typename T>
inline T Hal::get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_addr<T>(addr_type);
}

template <typename T>
inline T Hal::get_dev_addr(uint32_t programmable_core_type_index, HalL1MemAddrType addr_type) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_dev_addr<T>(addr_type);
}

inline uint32_t Hal::get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_size(addr_type);
}

template <typename T>
inline T Hal::get_dev_addr(HalDramMemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalDramMemAddrType>(addr_type);
    TT_ASSERT(index < this->dram_bases_.size());
    return reinterpret_cast<T>(this->dram_bases_[index]);
}

inline uint32_t Hal::get_dev_size(HalDramMemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalDramMemAddrType>(addr_type);
    TT_ASSERT(index < this->dram_sizes_.size());
    return this->dram_sizes_[index];
}

inline uint32_t Hal::get_alignment(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_alignments_.size());
    return this->mem_alignments_[index];
}

inline uint32_t Hal::get_common_alignment_with_pcie(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_alignments_with_pcie_.size());
    return this->mem_alignments_with_pcie_[index];
}

inline bool Hal::get_supports_cbs(uint32_t programmable_core_type_index) const {
    return this->core_info_[programmable_core_type_index].supports_cbs_;
}

inline const HalJitBuildConfig &Hal::get_jit_build_config(
    uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_jit_build_config(processor_class_idx, processor_type_idx);
}

class HalSingleton : public Hal {
private:
    HalSingleton() = default;
    HalSingleton(const HalSingleton&) = delete;
    HalSingleton(HalSingleton&&) = delete;
    ~HalSingleton() = default;

    HalSingleton& operator=(const HalSingleton&) = delete;
    HalSingleton& operator=(HalSingleton&&) = delete;

public:
    static inline HalSingleton& getInstance() {
        static HalSingleton instance;
        return instance;
    }
};

inline auto& hal = HalSingleton::getInstance();  // inline variable requires C++17

uint32_t generate_risc_startup_addr(uint32_t firmware_base); // used by Tensix initializers to build HalJitBuildConfig

}  // namespace tt_metal
}  // namespace tt

#define HAL_MEM_L1_BASE \
    tt::tt_metal::hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE)
#define HAL_MEM_L1_SIZE \
    tt::tt_metal::hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE)

#define HAL_MEM_ETH_BASE                                   \
    ((tt::tt_metal::hal.get_arch() == tt::ARCH::GRAYSKULL) \
         ? 0                                               \
         : tt::tt_metal::hal.get_dev_addr(                 \
               tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE))
#define HAL_MEM_ETH_SIZE                                   \
    ((tt::tt_metal::hal.get_arch() == tt::ARCH::GRAYSKULL) \
         ? 0                                               \
         : tt::tt_metal::hal.get_dev_size(                 \
               tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE))

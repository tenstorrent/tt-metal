// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the TT Hardware Abstraction Layer interface
// This layer abstracts which TT chip is running from the higher
// level APIs
//

#include <tt-metalium/assert.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/utils.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include "tt_memory.h"

enum class CoreType;
enum class AddressableCoreType : uint8_t;

namespace tt {

enum class ARCH;

namespace tt_metal {

// Note: nsidwell will be removing need for fw_base_addr and local_init_addr
// fw_launch_addr is programmed with fw_launch_addr_value on the master risc
// of a given progammable core to start FW.
// fw_launch_addr_value will be a jump instruction to FW or the address of FW
struct HalJitBuildConfig {
    DeviceAddr fw_base_addr;
    DeviceAddr local_init_addr;
    DeviceAddr fw_launch_addr;
    uint32_t fw_launch_addr_value;
    ll_api::memory::Loading memory_load;
};

// Features that are enabled on a given device. Enablement of the feature
// can be queried using the Hal::get_device_feature_enabled function.
enum class DeviceFeature {
    // Ethernet Firmware supports the usage of the mailbox API
    ETH_FW_API,
    // Dispatch to Active ethernet cores utilize a kernel config buffer
    DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER,
    // Dispatch to Idle ethernet cores utilize a kernel config buffer
    DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER,
    // Dispatch to Tensix cores utilize a kernel config buffer
    DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER,
    // Intermesh routing
    ETH_LINKS_INTERMESH_ROUTING,
};

// Ethernet Firmware mailbox messages
enum class FWMailboxMsg : uint8_t {
    // Message status mask.
    // msg & ETH_MSG_STATUS_MASK != ETH_MSG_CALL means the mailbox is free
    ETH_MSG_STATUS_MASK,
    // Execute message
    ETH_MSG_CALL,
    // Indicates message processed
    ETH_MSG_DONE,
    // Run link status check
    // arg0: copy_addr, arg1: unused, arg2: unused
    ETH_MSG_LINK_STATUS_CHECK,
    // Execute function from the core
    // arg0: L1 addr of function, arg1: unused, arg2: unused
    ETH_MSG_RELEASE_CORE,
    // Number of mailbox message types
    COUNT,
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
    std::vector<uint32_t> eth_fw_mailbox_msgs_;
    bool supports_cbs_ = false;
    bool supports_receiving_multicast_cmds_ = false;

public:
    HalCoreInfoType(
        HalProgrammableCoreType programmable_core_type,
        CoreType core_type,
        const std::vector<std::vector<HalJitBuildConfig>>& processor_classes,
        const std::vector<DeviceAddr>& mem_map_bases,
        const std::vector<uint32_t>& mem_map_sizes,
        const std::vector<uint32_t>& eth_fw_mailbox_msgs,
        bool supports_cbs,
        bool supports_receiving_multicast_cmds);

    template <typename T = DeviceAddr>
    T get_dev_addr(HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalL1MemAddrType addr_type) const;
    uint32_t get_processor_classes_count() const;
    uint32_t get_processor_types_count(uint32_t processor_class_idx) const;
    const HalJitBuildConfig& get_jit_build_config(uint32_t processor_class_idx, uint32_t processor_type_idx) const;
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

inline const HalJitBuildConfig& HalCoreInfoType::get_jit_build_config(
    uint32_t processor_class_idx, uint32_t processor_type_idx) const {
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
    using EthFwArgAddrFunc = std::function<uint32_t(uint32_t)>;
    using DeviceFeatureListFunc = std::function<bool(DeviceFeature)>;

private:
    tt::ARCH arch_;
    std::vector<HalCoreInfoType> core_info_;
    std::vector<DeviceAddr> dram_bases_;
    std::vector<uint32_t> dram_sizes_;
    std::vector<uint32_t> mem_alignments_;
    std::vector<uint32_t> mem_read_alignments_;
    std::vector<uint32_t> mem_write_alignments_;
    std::vector<uint32_t> mem_alignments_with_pcie_;
    uint32_t num_nocs_;
    uint32_t noc_addr_node_id_bits_;
    uint32_t noc_node_id_ = 0;
    uint32_t noc_node_id_mask_ = 0;
    uint32_t noc_encoding_reg_ = 0;
    uint32_t noc_coord_reg_offset_;
    uint32_t noc_overlay_start_addr_;
    uint32_t noc_stream_reg_space_size_;
    uint32_t noc_stream_remote_dest_buf_size_reg_index_;
    uint32_t noc_stream_remote_dest_buf_start_reg_index_;
    uint32_t noc_stream_remote_dest_buf_space_available_reg_index_;
    uint32_t noc_stream_remote_dest_buf_space_available_update_reg_index_;
    std::vector<uint32_t> noc_x_id_translate_table_;
    std::vector<uint32_t> noc_y_id_translate_table_;
    bool coordinate_virtualization_enabled_;
    uint32_t virtual_worker_start_x_;
    uint32_t virtual_worker_start_y_;
    bool eth_fw_is_cooperative_ = false;  // set when eth riscs have to context switch
    std::unordered_set<AddressableCoreType> virtualized_core_types_;
    HalTensixHarvestAxis tensix_harvest_axis_;

    float eps_ = 0.0f;
    float nan_ = 0.0f;
    float inf_ = 0.0f;

    void initialize_wh(bool is_base_routing_fw_enabled);
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
    EthFwArgAddrFunc eth_fw_arg_addr_func_;
    DeviceFeatureListFunc device_features_func_;

public:
    Hal(tt::ARCH arch, bool is_base_routing_fw_enabled);

    tt::ARCH get_arch() const { return arch_; }

    uint32_t get_num_nocs() const { return num_nocs_; }
    uint32_t get_noc_node_id() const { return noc_node_id_; }
    uint32_t get_noc_node_id_mask() const { return noc_node_id_mask_; }
    uint32_t get_noc_addr_node_id_bits() const { return noc_addr_node_id_bits_; }
    uint32_t get_noc_coord_reg_offset() const { return noc_coord_reg_offset_; }
    uint32_t get_noc_encoding_reg() const { return noc_encoding_reg_; }
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
    const std::unordered_set<AddressableCoreType>& get_virtualized_core_types() const {
        return this->virtualized_core_types_;
    }
    uint32_t get_eth_fw_mailbox_val(FWMailboxMsg msg) const;
    uint32_t get_eth_fw_mailbox_arg_addr(uint32_t arg_index) const;
    uint32_t get_eth_fw_mailbox_arg_count() const;
    HalTensixHarvestAxis get_tensix_harvest_axis() const { return tensix_harvest_axis_; }
    uint32_t get_programmable_core_type_count() const;
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index) const;
    uint32_t get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const;
    CoreType get_core_type(uint32_t programmable_core_type_index) const;
    uint32_t get_processor_classes_count(std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type) const;
    uint32_t get_processor_types_count(
        std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type, uint32_t processor_class_idx) const;
    // Query device features. Returns true if the feature is enabled.
    bool get_device_feature_enabled(DeviceFeature feature) const { return this->device_features_func_(feature); }
    // Returns true if the core has a kernel config buffer.
    bool get_core_has_kernel_config_buffer(HalProgrammableCoreType programmable_core_type) const;

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
    uint32_t get_read_alignment(HalMemType memory_type) const;
    uint32_t get_write_alignment(HalMemType memory_type) const;

    // Returns an alignment that is aligned with PCIE and the given memory type
    uint32_t get_common_alignment_with_pcie(HalMemType memory_type) const;

    bool get_supports_cbs(uint32_t programmable_core_type_index) const;

    bool get_supports_receiving_multicasts(uint32_t programmable_core_type_index) const;

    uint32_t get_num_risc_processors(HalProgrammableCoreType programmable_core_type) const;

    uint32_t get_total_num_risc_processors() const;

    const HalJitBuildConfig& get_jit_build_config(
        uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const;

    uint64_t relocate_dev_addr(uint64_t addr, uint64_t local_init_addr = 0) const {
        return relocate_func_(addr, local_init_addr);
    }

    uint64_t erisc_iram_relocate_dev_addr(uint64_t addr) const { return erisc_iram_relocate_func_(addr); }

    uint32_t valid_reg_addr(uint32_t addr) const { return valid_reg_addr_func_(addr); }

    const std::vector<uint32_t>& get_noc_x_id_translate_table() const { return noc_x_id_translate_table_; }
    const std::vector<uint32_t>& get_noc_y_id_translate_table() const { return noc_y_id_translate_table_; }
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
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::UNRESERVED),
        "Attempting to read addr of unreserved memory");
    return this->core_info_[index].get_dev_addr<T>(addr_type);
}

template <typename T>
inline T Hal::get_dev_addr(uint32_t programmable_core_type_index, HalL1MemAddrType addr_type) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    TT_FATAL(
        !(get_programmable_core_type(programmable_core_type_index) == HalProgrammableCoreType::TENSIX &&
          addr_type == HalL1MemAddrType::UNRESERVED),
        "Attempting to read addr of unreserved memory");
    return this->core_info_[programmable_core_type_index].get_dev_addr<T>(addr_type);
}

inline uint32_t Hal::get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::UNRESERVED),
        "Attempting to read size of unreserved memory");
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::KERNEL_CONFIG),
        "Attempting to read size of kernel config memory");
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

inline uint32_t Hal::get_read_alignment(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_read_alignments_.size());
    return this->mem_read_alignments_[index];
}

inline uint32_t Hal::get_write_alignment(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_write_alignments_.size());
    return this->mem_write_alignments_[index];
}

inline uint32_t Hal::get_common_alignment_with_pcie(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_alignments_with_pcie_.size());
    return this->mem_alignments_with_pcie_[index];
}

inline bool Hal::get_supports_cbs(uint32_t programmable_core_type_index) const {
    return this->core_info_[programmable_core_type_index].supports_cbs_;
}

inline bool Hal::get_supports_receiving_multicasts(uint32_t programmable_core_type_index) const {
    return this->core_info_[programmable_core_type_index].supports_receiving_multicast_cmds_;
}

inline uint32_t Hal::get_num_risc_processors(HalProgrammableCoreType programmable_core_type) const {
    const uint32_t num_processor_classes =
        this->core_info_[utils::underlying_type<HalProgrammableCoreType>(programmable_core_type)]
            .get_processor_classes_count();
    uint32_t num_riscs = 0;
    for (uint32_t processor_class_idx = 0; processor_class_idx < num_processor_classes; processor_class_idx++) {
        num_riscs += this->core_info_[utils::underlying_type<HalProgrammableCoreType>(programmable_core_type)]
                         .get_processor_types_count(processor_class_idx);
    }
    return num_riscs;
}

inline const HalJitBuildConfig& Hal::get_jit_build_config(
    uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_jit_build_config(processor_class_idx, processor_type_idx);
}

uint32_t generate_risc_startup_addr(uint32_t firmware_base);  // used by Tensix initializers to build HalJitBuildConfig

inline uint32_t Hal::get_eth_fw_mailbox_val(FWMailboxMsg msg) const {
    const auto index = utils::underlying_type<HalProgrammableCoreType>(HalProgrammableCoreType::ACTIVE_ETH);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].eth_fw_mailbox_msgs_[utils::underlying_type<FWMailboxMsg>(msg)];
}

inline uint32_t Hal::get_eth_fw_mailbox_arg_addr(uint32_t arg_index) const {
    return this->eth_fw_arg_addr_func_(arg_index);
}

inline uint32_t Hal::get_eth_fw_mailbox_arg_count() const {
    const auto index = utils::underlying_type<HalProgrammableCoreType>(HalProgrammableCoreType::ACTIVE_ETH);
    TT_ASSERT(index < this->core_info_.size());
    // -1 for the message
    return (this->core_info_[index].get_dev_size(HalL1MemAddrType::ETH_FW_MAILBOX) / sizeof(uint32_t)) - 1;
}

inline bool Hal::get_core_has_kernel_config_buffer(HalProgrammableCoreType programmable_core_type) const {
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            return get_device_feature_enabled(DeviceFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER);
        case HalProgrammableCoreType::ACTIVE_ETH:
            return get_device_feature_enabled(DeviceFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER);
        case HalProgrammableCoreType::IDLE_ETH:
            return get_device_feature_enabled(DeviceFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER);
        default: TT_THROW("Invalid HalProgrammableCoreType {}", static_cast<int>(programmable_core_type));
    }
}

}  // namespace tt_metal
}  // namespace tt

#define HAL_MEM_L1_BASE                                               \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_addr(      \
        ::tt::tt_metal::HalProgrammableCoreType::TENSIX, ::tt::tt_metal::HalL1MemAddrType::BASE)
#define HAL_MEM_L1_SIZE                                               \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_size(      \
        ::tt::tt_metal::HalProgrammableCoreType::TENSIX, ::tt::tt_metal::HalL1MemAddrType::BASE)

#define HAL_MEM_ETH_BASE                                              \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_addr(      \
        ::tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, ::tt::tt_metal::HalL1MemAddrType::BASE)
#define HAL_MEM_ETH_SIZE                                              \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_size(      \
        ::tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, ::tt::tt_metal::HalL1MemAddrType::BASE)

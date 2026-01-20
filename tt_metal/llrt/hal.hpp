// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the TT Hardware Abstraction Layer interface
// This layer abstracts which TT chip is running from the higher
// level APIs
//

#include <sys/types.h>
#include <cstddef>
#include <tt_stl/assert.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/enum.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_types.hpp>
#include <umd/device/utils/semver.hpp>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "tt_memory.h"
#include "hal/generated/dev_msgs.hpp"          // IWYU pragma: export
#include "hal/generated/fabric_telemetry.hpp"  // IWYU pragma: export

#include <tt_stl/overloaded.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>

enum class AddressableCoreType : uint8_t;

namespace tt::tt_metal {

// Struct of core type, processor class, and processor type to uniquely identify any processor.
struct HalProcessorIdentifier {
    HalProgrammableCoreType core_type = HalProgrammableCoreType::TENSIX;
    HalProcessorClassType processor_class = HalProcessorClassType::DM;
    int processor_type = 0;
};

std::ostream& operator<<(std::ostream&, const HalProcessorIdentifier&);
bool operator<(const HalProcessorIdentifier&, const HalProcessorIdentifier&);
bool operator==(const HalProcessorIdentifier&, const HalProcessorIdentifier&);

enum class HalDramMemAddrType : uint8_t { BARRIER = 0, PROFILER = 1, UNRESERVED = 2, COUNT = 3 };

enum class HalTensixHarvestAxis : uint8_t { ROW = 0x1, COL = 0x2 };

// A set of processors distinguishing programmable core type and index within that core type.
// See get_processor_index and get_processor_class_and_type_from_index.
class HalProcessorSet {
private:
    std::array<uint32_t, NumHalProgrammableCoreTypes> masks_{};

public:
    void add(HalProgrammableCoreType core_type, uint32_t processor_index) {
        masks_[static_cast<size_t>(core_type)] |= (1u << processor_index);
    }
    bool contains(HalProgrammableCoreType core_type, uint32_t processor_index) const {
        return (masks_[static_cast<size_t>(core_type)] & (1u << processor_index)) != 0;
    }
    bool empty() const {
        for (const auto& mask : masks_) {
            if (mask != 0) {
                return false;
            }
        }
        return true;
    }
    // Returns the bitmask of processors for the given core type.
    // Bit i set <=> processor index i is in the set.
    uint32_t get_processor_mask(HalProgrammableCoreType core_type) const {
        return masks_[static_cast<size_t>(core_type)];
    }
};

// Compile-time maximum for processor types count for any arch.  Useful for creating bitsets.
static constexpr int MAX_PROCESSOR_TYPES_COUNT = 8;

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
    // Heartbeat counter
    HEARTBEAT,
    // Retrain Count
    RETRAIN_COUNT,
    // Rx Link Up
    RX_LINK_UP,
    // Port Status
    PORT_STATUS,
    // Number of mailbox message types
    COUNT,
};

// Query dispatch related features depending on the arch
enum class DispatchFeature : uint8_t {
    // Ethernet Firmware supports the usage of the mailbox API
    ETH_MAILBOX_API,
    // Dispatch to Active ethernet cores utilize a kernel config buffer
    DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER,
    // Dispatch to Idle ethernet cores utilize a kernel config buffer
    DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER,
    // Dispatch to Tensix cores utilize a kernel config buffer
    DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER,
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
    // Number of firmware binaries generated for each processor class
    std::vector<uint8_t> processor_classes_num_fw_binaries_;
    // indices represents processor class and type positions, values are abbreviated name and full name pairs
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names_;
    std::vector<DeviceAddr> mem_map_bases_;
    std::vector<uint32_t> mem_map_sizes_;
    std::vector<uint32_t> eth_fw_mailbox_msgs_;
    bool supports_cbs_ = false;
    bool supports_receiving_multicast_cmds_ = false;
    dev_msgs::Factory dev_msgs_factory_;
    tt::tt_fabric::fabric_telemetry::Factory fabric_telemetry_factory_;

public:
    HalCoreInfoType(
        HalProgrammableCoreType programmable_core_type,
        CoreType core_type,
        std::vector<std::vector<HalJitBuildConfig>> processor_classes,
        std::vector<uint8_t> processor_classes_num_fw_binaries,
        std::vector<DeviceAddr> mem_map_bases,
        std::vector<uint32_t> mem_map_sizes,
        std::vector<uint32_t> eth_fw_mailbox_msgs,
        std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names,
        bool supports_cbs,
        bool supports_receiving_multicast_cmds,
        dev_msgs::Factory dev_msgs_factory,
        tt::tt_fabric::fabric_telemetry::Factory fabric_telemetry_factory) :
        programmable_core_type_(programmable_core_type),
        core_type_(core_type),
        processor_classes_(std::move(processor_classes)),
        processor_classes_num_fw_binaries_(std::move(processor_classes_num_fw_binaries)),
        processor_classes_names_(std::move(processor_classes_names)),
        mem_map_bases_(std::move(mem_map_bases)),
        mem_map_sizes_(std::move(mem_map_sizes)),
        eth_fw_mailbox_msgs_{std::move(eth_fw_mailbox_msgs)},
        supports_cbs_(supports_cbs),
        supports_receiving_multicast_cmds_(supports_receiving_multicast_cmds),
        dev_msgs_factory_(dev_msgs_factory),
        fabric_telemetry_factory_(fabric_telemetry_factory) {}

    DeviceAddr get_dev_addr(HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalL1MemAddrType addr_type) const;
    uint32_t get_processor_classes_count() const;
    uint32_t get_processor_types_count(uint32_t processor_class_idx) const;
    uint32_t get_processor_index(HalProcessorClassType processor_class, uint32_t processor_type_idx) const;
    std::pair<HalProcessorClassType, uint32_t> get_processor_class_and_type_from_index(uint32_t processor_index) const;
    const HalJitBuildConfig& get_jit_build_config(uint32_t processor_class_idx, uint32_t processor_type_idx) const;
    const std::string& get_processor_class_name(uint32_t processor_index, bool is_abbreviated) const;
    uint32_t get_processor_class_num_fw_binaries(uint32_t processor_class_idx) const;
    const dev_msgs::Factory& get_dev_msgs_factory() const;
    const tt::tt_fabric::fabric_telemetry::Factory& get_fabric_telemetry_factory() const;
};

inline DeviceAddr HalCoreInfoType::get_dev_addr(HalL1MemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_bases_.size());
    return this->mem_map_bases_[index];
}

inline uint32_t HalCoreInfoType::get_dev_size(HalL1MemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_sizes_.size());
    return this->mem_map_sizes_[index];
}

inline uint32_t HalCoreInfoType::get_processor_classes_count() const { return this->processor_classes_.size(); }

inline uint32_t HalCoreInfoType::get_processor_types_count(uint32_t processor_class_idx) const {
    TT_ASSERT(processor_class_idx < this->processor_classes_.size());
    uint32_t count = this->processor_classes_[processor_class_idx].size();
    TT_ASSERT(count <= MAX_PROCESSOR_TYPES_COUNT);
    return count;
}

inline const HalJitBuildConfig& HalCoreInfoType::get_jit_build_config(
    uint32_t processor_class_idx, uint32_t processor_type_idx) const {
    TT_ASSERT(processor_class_idx < this->processor_classes_.size());
    TT_ASSERT(processor_type_idx < this->processor_classes_[processor_class_idx].size());
    return this->processor_classes_[processor_class_idx][processor_type_idx];
}

inline const dev_msgs::Factory& HalCoreInfoType::get_dev_msgs_factory() const { return this->dev_msgs_factory_; }

inline const tt::tt_fabric::fabric_telemetry::Factory& HalCoreInfoType::get_fabric_telemetry_factory() const {
    return this->fabric_telemetry_factory_;
}

// HalJitBuildQueryInterface is an interface for querying arch-specific build options.
// These are generated on demand instead of stored in HalJitBuildConfig,
// as the options may vary based on fw build or kernel build.
class HalJitBuildQueryInterface {
public:
    // The results from this query interface may vary based on these parameters.
    struct Params {
        bool is_fw;
        HalProgrammableCoreType core_type;
        HalProcessorClassType processor_class;
        uint32_t processor_id;
    };
    virtual ~HalJitBuildQueryInterface() = default;
    // Returns a list of objects to be linked; these were compiled offline.
    // Paths are relative to the tt-metal root.
    virtual std::vector<std::string> link_objs(const Params& params) const = 0;
    // Returns a list of includes paths to be added to compiler command line.
    virtual std::vector<std::string> includes(const Params& params) const = 0;
    // Returns a list of defines to be added to compiler command line.
    virtual std::vector<std::string> defines(const Params& params) const = 0;
    // Returns a list of source files to be added to compiler command line.
    virtual std::vector<std::string> srcs(const Params& params) const = 0;
    // Returns a string of common flags to be added to compiler and linker command lines.
    virtual std::string common_flags(const Params& params) const = 0;
    // Returns the path to the linker script, relative to the tt-metal root.
    virtual std::string linker_script(const Params& params) const = 0;
    // Returns a string of linker flags to be added to linker command line.
    virtual std::string linker_flags(const Params& params) const = 0;
    // Returns true if firmware should be linked into the kernel as an object.
    virtual bool firmware_is_kernel_object(const Params& params) const = 0;
    // Returns the target name for the build.
    // Note: this is added only to keep the target names consistent with the previous
    // implementation of build, to avoid breaking users / tools.
    // We can migrate build to use arch-independent target names, and then this can be removed.
    virtual std::string target_name(const Params& params) const = 0;
};

class Hal {
public:
    using RelocateFunc = std::function<uint64_t(uint64_t, uint64_t, bool)>;
    using IramRelocateFunc = std::function<uint64_t(uint64_t)>;
    using ValidRegAddrFunc = std::function<bool(uint32_t)>;
    using NOCXYEncodingFunc = std::function<uint32_t(uint32_t, uint32_t)>;
    using NOCMulticastEncodingFunc = std::function<uint32_t(uint32_t, uint32_t, uint32_t, uint32_t)>;
    using NOCAddrFunc = std::function<uint64_t(uint64_t)>;
    using StackSizeFunc = std::function<uint32_t(uint32_t)>;
    using EthFwArgAddrFunc = std::function<uint32_t(int, uint32_t)>;
    using DispatchFeatureQueryFunc = std::function<bool(DispatchFeature)>;
    using SetIRAMTextSizeFunc = std::function<void(
        dev_msgs::launch_msg_t::View, HalProgrammableCoreType, HalProcessorClassType, uint32_t, uint32_t)>;
    using VerifyFwVersionFunc = std::function<bool(tt::umd::semver_t)>;

private:
    tt::ARCH arch_;
    std::vector<HalCoreInfoType> core_info_;
    std::vector<DeviceAddr> dram_bases_;
    std::vector<uint32_t> dram_sizes_;
    std::vector<uint32_t> mem_alignments_;
    std::vector<uint32_t> mem_read_alignments_;
    std::vector<uint32_t> mem_write_alignments_;
    std::vector<uint32_t> mem_alignments_with_pcie_;
    uint32_t max_processors_per_core_{};
    // Architecture-defined PCIe address range
    uint64_t pcie_addr_lower_bound_{};
    uint64_t pcie_addr_upper_bound_{};
    uint32_t num_nocs_{};
    uint32_t noc_addr_node_id_bits_{};
    uint32_t noc_node_id_ = 0;
    uint32_t noc_node_id_mask_ = 0;
    uint32_t noc_encoding_reg_ = 0;
    uint32_t noc_coord_reg_offset_{};
    uint32_t noc_overlay_start_addr_{};
    uint32_t noc_stream_reg_space_size_{};
    uint32_t noc_stream_remote_dest_buf_size_reg_index_{};
    uint32_t noc_stream_remote_dest_buf_start_reg_index_{};
    uint32_t noc_stream_remote_dest_buf_space_available_reg_index_{};
    uint32_t noc_stream_remote_dest_buf_space_available_update_reg_index_{};
    std::vector<uint32_t> noc_x_id_translate_table_;
    std::vector<uint32_t> noc_y_id_translate_table_;
    bool coordinate_virtualization_enabled_{};
    bool supports_64_bit_pcie_addressing_{};
    uint32_t virtual_worker_start_x_{};
    uint32_t virtual_worker_start_y_{};
    bool eth_fw_is_cooperative_ = false;  // set when eth riscs have to context switch
    std::unordered_set<dev_msgs::AddressableCoreType> virtualized_core_types_;
    HalTensixHarvestAxis tensix_harvest_axis_{HalTensixHarvestAxis::ROW};
    size_t max_pinned_memory_count_{};
    size_t total_pinned_memory_size_{};

    float eps_ = 0.0f;
    float nan_ = 0.0f;
    float inf_ = 0.0f;

    void initialize_wh(bool is_base_routing_fw_enabled, uint32_t profiler_dram_bank_size_per_risc_bytes);
    void initialize_bh(bool enable_2_erisc_mode, uint32_t profiler_dram_bank_size_per_risc_bytes);
    void initialize_qa(uint32_t profiler_dram_bank_size_per_risc_bytes);

    // Functions where implementation varies by architecture
    RelocateFunc relocate_func_;
    IramRelocateFunc erisc_iram_relocate_func_;
    ValidRegAddrFunc valid_reg_addr_func_;
    NOCXYEncodingFunc noc_xy_encoding_func_;
    NOCXYEncodingFunc noc_xy_pcie64_encoding_func_;
    NOCMulticastEncodingFunc noc_multicast_encoding_func_;
    NOCAddrFunc noc_mcast_addr_start_x_func_;
    NOCAddrFunc noc_mcast_addr_start_y_func_;
    NOCAddrFunc noc_mcast_addr_end_x_func_;
    NOCAddrFunc noc_mcast_addr_end_y_func_;
    NOCAddrFunc noc_ucast_addr_x_func_;
    NOCAddrFunc noc_ucast_addr_y_func_;
    NOCAddrFunc noc_local_addr_func_;
    EthFwArgAddrFunc eth_fw_arg_addr_func_;
    DispatchFeatureQueryFunc device_features_func_;
    std::unique_ptr<HalJitBuildQueryInterface> jit_build_query_;
    SetIRAMTextSizeFunc set_iram_text_size_func_;
    VerifyFwVersionFunc verify_eth_fw_version_func_;

public:
    Hal(tt::ARCH arch,
        bool is_base_routing_fw_enabled,
        bool enable_2_erisc_mode,
        uint32_t profiler_dram_bank_size_per_risc_bytes);

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

    // Returns the NOC addr to be used with 64 bit PCIe address space.
    uint32_t noc_xy_pcie64_encoding(uint32_t x, uint32_t y) const { return noc_xy_pcie64_encoding_func_(x, y); }
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
    const std::unordered_set<dev_msgs::AddressableCoreType>& get_virtualized_core_types() const {
        return this->virtualized_core_types_;
    }

    bool get_supports_eth_fw_mailbox() const;
    uint32_t get_eth_fw_mailbox_val(FWMailboxMsg msg) const;
    uint32_t get_eth_fw_mailbox_arg_addr(int mailbox_index, uint32_t arg_index) const;
    uint32_t get_eth_fw_mailbox_arg_count() const;
    uint32_t get_eth_fw_mailbox_address(int mailbox_index) const;
    HalTensixHarvestAxis get_tensix_harvest_axis() const { return tensix_harvest_axis_; }
    uint32_t get_programmable_core_type_count() const;
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index) const;
    uint32_t get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const;
    CoreType get_core_type(uint32_t programmable_core_type_index) const;
    uint32_t get_processor_classes_count(HalProgrammableCoreType programmable_core_type) const;
    uint32_t get_processor_types_count(
        std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type, uint32_t processor_class_idx) const;
    // Query device features. Returns true if the feature is enabled.
    bool get_dispatch_feature_enabled(DispatchFeature feature) const { return this->device_features_func_(feature); }
    // Returns true if kernel binaries for a given core type are stored in the config buffer
    // Note, binaries which are not stored in the config buffer are written directly to L1 at the text start address.
    // This value can be found in the ELF file.
    bool get_core_kernel_stored_in_config_buffer(HalProgrammableCoreType programmable_core_type) const;

    DeviceAddr get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;

    // Overloads for Dram Params
    DeviceAddr get_dev_addr(HalDramMemAddrType addr_type) const;
    uint32_t get_dev_size(HalDramMemAddrType addr_type) const;

    uint32_t get_alignment(HalMemType memory_type) const;
    uint32_t get_read_alignment(HalMemType memory_type) const;
    uint32_t get_write_alignment(HalMemType memory_type) const;
    uint32_t get_dma_alignment() const;

    // Returns an alignment that is aligned with PCIE and the given memory type
    uint32_t get_common_alignment_with_pcie(HalMemType memory_type) const;

    bool get_supports_cbs(uint32_t programmable_core_type_index) const;

    bool get_supports_receiving_multicasts(uint32_t programmable_core_type_index) const;

    uint32_t get_num_risc_processors(HalProgrammableCoreType programmable_core_type) const;
    // Returns the processor index within a core.  There is a 1-1 mapping between
    // (processor_class, processor_type) and processor_index.  This is useful
    // For indexing data structures on devices (only 1-d arrays are needed).
    // Should only be used internally and not expose this index to the user.
    uint32_t get_processor_index(
        HalProgrammableCoreType programmable_core_type,
        HalProcessorClassType processor_class,
        uint32_t processor_type_idx) const;
    // Inverse function of get_processor_index.
    std::pair<HalProcessorClassType, uint32_t> get_processor_class_and_type_from_index(
        HalProgrammableCoreType programmable_core_type, uint32_t processor_index) const;
    // Parses a string representation of a set of processor names (used by env vars).
    HalProcessorSet parse_processor_set_spec(std::string_view spec) const;

    uint32_t get_total_num_risc_processors() const;
    uint32_t get_max_processors_per_core() const { return max_processors_per_core_; }

    const HalJitBuildConfig& get_jit_build_config(
        uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const;

    const std::string& get_processor_class_name(
        HalProgrammableCoreType programmable_core_type, uint32_t processor_index, bool is_abbreviated) const;

    uint32_t get_processor_class_num_fw_binaries(
        uint32_t programmable_core_type_index, uint32_t processor_class_idx) const;

    uint64_t relocate_dev_addr(uint64_t addr, uint64_t local_init_addr = 0, bool has_shared_local_mem = false) const {
        return relocate_func_(addr, local_init_addr, has_shared_local_mem);
    }

    uint64_t erisc_iram_relocate_dev_addr(uint64_t addr) const { return erisc_iram_relocate_func_(addr); }

    uint32_t valid_reg_addr(uint32_t addr) const { return valid_reg_addr_func_(addr); }

    const std::vector<uint32_t>& get_noc_x_id_translate_table() const { return noc_x_id_translate_table_; }
    const std::vector<uint32_t>& get_noc_y_id_translate_table() const { return noc_y_id_translate_table_; }

    const HalJitBuildQueryInterface& get_jit_build_query() const {
        TT_ASSERT(jit_build_query_ != nullptr);
        return *jit_build_query_;
    }

    const dev_msgs::Factory& get_dev_msgs_factory(HalProgrammableCoreType programmable_core_type) const {
        auto index = get_programmable_core_type_index(programmable_core_type);
        TT_ASSERT(index < this->core_info_.size());
        return this->core_info_[index].get_dev_msgs_factory();
    }

    const tt::tt_fabric::fabric_telemetry::Factory& get_fabric_telemetry_factory(
        HalProgrammableCoreType programmable_core_type) const {
        TT_ASSERT(programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH);
        auto index = get_programmable_core_type_index(programmable_core_type);
        return this->core_info_[index].get_fabric_telemetry_factory();
    }

    // This interface guarantees that go_msg_t is 4B and has the same layout for all core types.
    // Code that assumes that should use this interface to create go_msg_t values,
    // as it is otherwise not guaranteed by the HAL interface.
    uint32_t make_go_msg_u32(uint8_t signal, uint8_t master_x, uint8_t master_y, uint8_t dispatch_message_offset) const;

    // If the specified processor uses IRAM, update the launch message to set the IRAM text size.
    void set_iram_text_size(
        dev_msgs::launch_msg_t::View launch_msg,
        HalProgrammableCoreType programmable_core_type,
        HalProcessorClassType processor_class,
        uint32_t processor_type_idx,
        uint32_t iram_text_size) const {
        if (this->set_iram_text_size_func_) {
            this->set_iram_text_size_func_(
                launch_msg, programmable_core_type, processor_class, processor_type_idx, iram_text_size);
        }
    }

    // Returns the supported PCIe address range for the current architecture
    uint64_t get_pcie_addr_lower_bound() const;
    // Inclusive upper bound
    uint64_t get_pcie_addr_upper_bound() const;
    bool get_supports_64_bit_pcie_addressing() const { return supports_64_bit_pcie_addressing_; }

    // Verify that the eth version is compatible with the HAL capabilities. Throws an exception if version is
    // not compatible.
    bool verify_eth_fw_version(tt::umd::semver_t eth_fw_version) const {
        return this->verify_eth_fw_version_func_(eth_fw_version);
    }

    size_t get_max_pinned_memory_count() const { return max_pinned_memory_count_; }
    size_t get_total_pinned_memory_size() const { return total_pinned_memory_size_; }
};

inline uint32_t Hal::get_programmable_core_type_count() const { return core_info_.size(); }

inline uint32_t Hal::get_processor_classes_count(HalProgrammableCoreType programmable_core_type) const {
    uint32_t index = get_programmable_core_type_index(programmable_core_type);
    return this->core_info_[index].get_processor_classes_count();
}

inline uint32_t Hal::get_processor_types_count(
    std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type, uint32_t processor_class_idx) const {
    uint32_t index = std::visit(
        ttsl::overloaded{
            [](HalProgrammableCoreType core_type_specifier) -> uint32_t {
                return ttsl::as_underlying_type(core_type_specifier);
            },
            [](uint32_t core_type_specifier) { return core_type_specifier; },
        },
        programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_processor_types_count(processor_class_idx);
}

inline HalProgrammableCoreType Hal::get_programmable_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].programmable_core_type_;
}

inline CoreType Hal::get_core_type(uint32_t core_type_index) const { return core_info_[core_type_index].core_type_; }

inline DeviceAddr Hal::get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::UNRESERVED),
        "Attempting to read addr of unreserved memory");
    return this->core_info_[index].get_dev_addr(addr_type);
}

inline uint32_t Hal::get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::UNRESERVED),
        "Attempting to read size of unreserved memory");
    TT_FATAL(
        !(programmable_core_type == HalProgrammableCoreType::TENSIX && addr_type == HalL1MemAddrType::KERNEL_CONFIG),
        "Attempting to read size of kernel config memory");
    return this->core_info_[index].get_dev_size(addr_type);
}

inline DeviceAddr Hal::get_dev_addr(HalDramMemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalDramMemAddrType>(addr_type);
    TT_ASSERT(index < this->dram_bases_.size());
    return this->dram_bases_[index];
}

inline uint32_t Hal::get_dev_size(HalDramMemAddrType addr_type) const {
    uint32_t index = ttsl::as_underlying_type<HalDramMemAddrType>(addr_type);
    TT_ASSERT(index < this->dram_sizes_.size());
    return this->dram_sizes_[index];
}

inline uint32_t Hal::get_alignment(HalMemType memory_type) const {
    uint32_t index = ttsl::as_underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_alignments_.size());
    return this->mem_alignments_[index];
}

inline uint32_t Hal::get_read_alignment(HalMemType memory_type) const {
    uint32_t index = ttsl::as_underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_read_alignments_.size());
    return this->mem_read_alignments_[index];
}

inline uint32_t Hal::get_write_alignment(HalMemType memory_type) const {
    uint32_t index = ttsl::as_underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_write_alignments_.size());
    return this->mem_write_alignments_[index];
}

inline uint32_t Hal::get_dma_alignment() const {
    switch (arch_) {
        case tt::ARCH::WORMHOLE_B0: return 4;
        // Only Wormhole B0 devices support DMA transfers today.
        default: return 1;
    }
}

inline uint32_t Hal::get_common_alignment_with_pcie(HalMemType memory_type) const {
    uint32_t index = ttsl::as_underlying_type<HalMemType>(memory_type);
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
        this->core_info_[ttsl::as_underlying_type<HalProgrammableCoreType>(programmable_core_type)]
            .get_processor_classes_count();
    uint32_t num_riscs = 0;
    for (uint32_t processor_class_idx = 0; processor_class_idx < num_processor_classes; processor_class_idx++) {
        num_riscs += this->core_info_[ttsl::as_underlying_type<HalProgrammableCoreType>(programmable_core_type)]
                         .get_processor_types_count(processor_class_idx);
    }
    TT_ASSERT(num_riscs <= max_processors_per_core_);
    return num_riscs;
}
inline uint32_t Hal::get_processor_index(
    HalProgrammableCoreType programmable_core_type,
    HalProcessorClassType processor_class,
    uint32_t processor_type_idx) const {
    auto idx = get_programmable_core_type_index(programmable_core_type);
    return this->core_info_[idx].get_processor_index(processor_class, processor_type_idx);
}

inline std::pair<HalProcessorClassType, uint32_t> Hal::get_processor_class_and_type_from_index(
    HalProgrammableCoreType programmable_core_type, uint32_t processor_index) const {
    auto idx = get_programmable_core_type_index(programmable_core_type);
    return this->core_info_[idx].get_processor_class_and_type_from_index(processor_index);
}

inline const HalJitBuildConfig& Hal::get_jit_build_config(
    uint32_t programmable_core_type_index, uint32_t processor_class_idx, uint32_t processor_type_idx) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_jit_build_config(processor_class_idx, processor_type_idx);
}

inline const std::string& Hal::get_processor_class_name(
    HalProgrammableCoreType programmable_core_type, uint32_t processor_index, bool is_abbreviated) const {
    auto idx = get_programmable_core_type_index(programmable_core_type);
    return this->core_info_[idx].get_processor_class_name(processor_index, is_abbreviated);
}

inline uint32_t Hal::get_processor_class_num_fw_binaries(
    uint32_t programmable_core_type_index, uint32_t processor_class_idx) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_processor_class_num_fw_binaries(processor_class_idx);
}

uint32_t generate_risc_startup_addr(uint32_t firmware_base);  // used by Tensix initializers to build HalJitBuildConfig

inline bool Hal::get_supports_eth_fw_mailbox() const {
    return this->get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ETH_FW_MAILBOX) != 0;
}

inline uint32_t Hal::get_eth_fw_mailbox_val(FWMailboxMsg msg) const {
    const auto index = ttsl::as_underlying_type<HalProgrammableCoreType>(HalProgrammableCoreType::ACTIVE_ETH);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].eth_fw_mailbox_msgs_[ttsl::as_underlying_type<FWMailboxMsg>(msg)];
}

inline uint32_t Hal::get_eth_fw_mailbox_arg_addr(int mailbox_index, uint32_t arg_index) const {
    return this->eth_fw_arg_addr_func_(mailbox_index, arg_index);
}

inline uint32_t Hal::get_eth_fw_mailbox_arg_count() const {
    const auto index = ttsl::as_underlying_type<HalProgrammableCoreType>(HalProgrammableCoreType::ACTIVE_ETH);
    TT_ASSERT(index < this->core_info_.size());
    // -1 for the message
    return (this->core_info_[index].get_dev_size(HalL1MemAddrType::ETH_FW_MAILBOX) / sizeof(uint32_t)) - 1;
}

inline uint32_t Hal::get_eth_fw_mailbox_address(int mailbox_index) const {
    const auto index = ttsl::as_underlying_type<HalProgrammableCoreType>(HalProgrammableCoreType::ACTIVE_ETH);
    TT_ASSERT(index < this->core_info_.size());
    // Index 0 is the offset of the mailbox message
    return get_eth_fw_mailbox_arg_addr(mailbox_index, 0) - sizeof(uint32_t);
}

inline bool Hal::get_core_kernel_stored_in_config_buffer(HalProgrammableCoreType programmable_core_type) const {
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            return get_dispatch_feature_enabled(DispatchFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER);
        case HalProgrammableCoreType::ACTIVE_ETH:
            return get_dispatch_feature_enabled(DispatchFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER);
        case HalProgrammableCoreType::IDLE_ETH:
            return get_dispatch_feature_enabled(DispatchFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER);
        default: TT_THROW("Invalid HalProgrammableCoreType {}", static_cast<int>(programmable_core_type));
    }
}

// Core type conversion utility
constexpr HalProgrammableCoreType hal_programmable_core_type_from_core_type(CoreType core_type) {
    switch (core_type) {
        case CoreType::WORKER:
        case CoreType::TENSIX: return HalProgrammableCoreType::TENSIX;
        case CoreType::ACTIVE_ETH: return HalProgrammableCoreType::ACTIVE_ETH;
        case CoreType::IDLE_ETH: return HalProgrammableCoreType::IDLE_ETH;
        default: TT_FATAL(false, "CoreType is not recognized by the HAL in {}", __FUNCTION__);
    }
}

}  // namespace tt::tt_metal

template <>
struct std::hash<tt::tt_metal::HalProcessorIdentifier> {
    std::size_t operator()(const tt::tt_metal::HalProcessorIdentifier&) const;
};

#define HAL_MEM_L1_BASE                                          \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_addr( \
        ::tt::tt_metal::HalProgrammableCoreType::TENSIX, ::tt::tt_metal::HalL1MemAddrType::BASE)
#define HAL_MEM_L1_SIZE                                          \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_size( \
        ::tt::tt_metal::HalProgrammableCoreType::TENSIX, ::tt::tt_metal::HalL1MemAddrType::BASE)

#define HAL_MEM_ETH_BASE                                         \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_addr( \
        ::tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, ::tt::tt_metal::HalL1MemAddrType::BASE)
#define HAL_MEM_ETH_SIZE                                         \
    ::tt::tt_metal::MetalContext::instance().hal().get_dev_size( \
        ::tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, ::tt::tt_metal::HalL1MemAddrType::BASE)

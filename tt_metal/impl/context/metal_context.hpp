// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/hal_types.hpp>
#include "dev_msgs.h"
#include <tt-metalium/allocator_types.hpp>
#include <llrt/tt_cluster.hpp>
#include <llrt/hal.hpp>
#include <llrt/rtoptions.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <magic_enum/magic_enum.hpp>

#include <array>
#include <unordered_set>
#include <vector>

namespace tt::tt_fabric {
class GlobalControlPlane;
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

namespace inspector {
class Data;
}

// A class to manage one-time initialization and teardown (FW, dispatch, fabric, cluster) and access to related state.
// Dispatch-independent state (Cluster) is initialized with the creation of MetalContext and accessible right after.
// Dispatch-dependent state (FW, dispatch, fabric) is initialized explicitly with a MetalContext::initialize() call, and
// only accessible after that.
class MetalContext {
public:
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext& operator=(MetalContext&& other) noexcept = delete;
    MetalContext(const MetalContext&) = delete;
    MetalContext(MetalContext&& other) noexcept = delete;
    static MetalContext& instance();

    Cluster& get_cluster();
    llrt::RunTimeOptions& rtoptions();
    const Cluster& get_cluster() const;
    const llrt::RunTimeOptions& rtoptions() const;
    const Hal& hal() const;
    dispatch_core_manager& get_dispatch_core_manager();
    DispatchQueryManager& get_dispatch_query_manager();
    const DispatchMemMap& dispatch_mem_map() const;  // DispatchMemMap for the core type we're dispatching on.
    const DispatchMemMap& dispatch_mem_map(const CoreType& core_type) const;  // DispatchMemMap for specific core type.
    inspector::Data* get_inspector_data() const {
        return inspector_data_.get();
    }

    void initialize(
        const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, const BankMapping& l1_bank_remap);

    // Control plane accessors
    tt::tt_fabric::ControlPlane& get_control_plane();
    void set_custom_control_plane_mesh_graph(
        const std::string& mesh_graph_desc_file,
        const std::map<tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    void set_default_control_plane_mesh_graph();
    void set_fabric_config(
        tt_metal::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt);
    void initialize_fabric_config();
    tt_metal::FabricConfig get_fabric_config() const;

private:
    friend class tt::stl::Indestructible<MetalContext>;
    MetalContext();
    ~MetalContext();
    void teardown();

    void clear_l1_state(chip_id_t device_id);
    void clear_dram_state(chip_id_t device_id);
    void clear_launch_messages_on_eth_cores(chip_id_t device_id);
    void initialize_control_plane();
    void teardown_fabric_config();

    bool initialized_ = false;
    bool teardown_registered_ = false;

    uint8_t num_hw_cqs_ = 0;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;
    size_t fw_compile_hash_ = 0;  // To check if FW recompilation is needed

    // Used to track which FW has been built already
    std::unordered_set<uint32_t> firmware_built_keys_;

    llrt::RunTimeOptions rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;
    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;
    std::unique_ptr<inspector::Data> inspector_data_;
    std::array<std::unique_ptr<DispatchMemMap>, magic_enum::enum_count<CoreType>()> dispatch_mem_map_;
    std::unique_ptr<tt::tt_fabric::GlobalControlPlane> global_control_plane_;
    tt_metal::FabricConfig fabric_config_ = tt_metal::FabricConfig::DISABLED;
    uint8_t num_fabric_active_routing_planes_ = 0;
};

}  // namespace tt::tt_metal

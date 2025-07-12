// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed_context.hpp>
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
#include <impl/debug/dprint_server.hpp>
#include <impl/debug/watcher_server.hpp>

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
    std::unique_ptr<DPrintServer>& dprint_server() { return dprint_server_; }
    std::unique_ptr<WatcherServer>& watcher_server() { return watcher_server_; }

    void initialize(
        const DispatchCoreConfig& dispatch_core_config,
        uint8_t num_hw_cqs,
        const BankMapping& l1_bank_remap,
        size_t worker_l1_size,
        bool minimal = false);
    void reinitialize();
    void teardown();

    // Control plane accessors
    tt::tt_fabric::ControlPlane& get_control_plane();
    void set_custom_control_plane_mesh_graph(
        const std::string& mesh_graph_desc_file,
        const std::map<tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    void set_default_control_plane_mesh_graph();
    void set_fabric_config(
        tt_fabric::FabricConfig fabric_config,
        tt_fabric::FabricReliabilityMode reliability_mode =
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::optional<uint8_t> num_routing_planes = std::nullopt);
    void initialize_fabric_config();
    tt_fabric::FabricConfig get_fabric_config() const;

    distributed::multihost::DistributedContext& get_distributed_context();

private:
    friend class tt::stl::Indestructible<MetalContext>;
    MetalContext();
    ~MetalContext();

    void clear_l1_state(chip_id_t device_id);
    void clear_dram_state(chip_id_t device_id);
    void clear_launch_messages_on_eth_cores(chip_id_t device_id);
    void initialize_control_plane();
    void teardown_fabric_config();

    void reset_cores(chip_id_t device_id);
    void assert_cores(chip_id_t device_id);

    // Returns the ERISC Launch Flag address
    uint32_t get_active_erisc_launch_flag_addr();
    // Returns true if metal firmware or a kernel is running on the virtual ethernet core
    bool erisc_app_still_running(chip_id_t device_id, CoreCoord virtual_core);
    // Send a message to exit the erisc app
    void erisc_send_exit_signal(chip_id_t device_id, CoreCoord virtual_core, bool is_idle_eth);

    // Functions used to init/run firmware on devices
    CoreCoord virtual_noc0_coordinate(chip_id_t device_id, uint8_t noc_index, CoreCoord coord);
    void generate_device_bank_to_noc_tables(chip_id_t device_id);
    void initialize_device_bank_to_noc_tables(
        chip_id_t device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core);
    void initialize_firmware(
        chip_id_t device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        launch_msg_t* launch_msg,
        go_msg_t* go_msg);
    void initialize_and_launch_firmware(chip_id_t device_id);

    bool initialized_ = false;
    bool teardown_registered_ = false;
    bool force_reinit_ = false;

    uint8_t num_hw_cqs_ = 0;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;
    size_t worker_l1_size_ = 0;
    size_t worker_l1_unreserved_start_ = 0;
    size_t fw_compile_hash_ = 0;  // To check if FW recompilation is needed

    // Used to track which FW has been built already
    std::unordered_set<uint32_t> firmware_built_keys_;

    // Written to device as part of FW init, device-specific
    std::unordered_map<chip_id_t, std::vector<int32_t>> dram_bank_offset_map_;
    std::unordered_map<chip_id_t, std::vector<int32_t>> l1_bank_offset_map_;
    std::unordered_map<chip_id_t, std::vector<uint16_t>> dram_bank_to_noc_xy_;
    std::unordered_map<chip_id_t, std::vector<uint16_t>> l1_bank_to_noc_xy_;

    llrt::RunTimeOptions rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;
    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;
    std::unique_ptr<inspector::Data> inspector_data_;
    std::unique_ptr<DPrintServer> dprint_server_;
    std::unique_ptr<WatcherServer> watcher_server_;
    std::array<std::unique_ptr<DispatchMemMap>, static_cast<size_t>(CoreType::COUNT)> dispatch_mem_map_;
    std::unique_ptr<tt::tt_fabric::GlobalControlPlane> global_control_plane_;
    tt_fabric::FabricConfig fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;

    // Strict system health mode requires (expects) all links/devices to be live. When enabled, it
    // is expected that any downed devices/links will result in some sort of error condition being
    // reported. When set to false, the control plane is free to instantiate fewer routing planes
    // according to which links are available.
    tt_fabric::FabricReliabilityMode fabric_reliability_mode_ = tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    uint8_t num_fabric_active_routing_planes_ = 0;
};

}  // namespace tt::tt_metal

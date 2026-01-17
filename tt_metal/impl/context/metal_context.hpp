// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <llrt/rtoptions.hpp>
#include <impl/allocator/allocator_types.hpp>
#include "experimental/fabric/routing_table_generator.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "hostdevcommon/api/hostdevcommon/common_values.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt::tt_metal {
struct ProfilerStateManager;

namespace inspector {
class Data;
}

class DataCollector;
class DeviceManager;
class Hal;
class dispatch_core_manager;
class DispatchQueryManager;
class DPrintServer;
class WatcherServer;
class DispatchMemMap;

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

    std::unique_ptr<ProfilerStateManager>& profiler_state_manager() { return profiler_state_manager_; }
    std::unique_ptr<DataCollector>& data_collector() { return data_collector_; }
    std::unique_ptr<DeviceManager>& device_manager() { return device_manager_; }

    void initialize_device_manager(
        const std::vector<ChipId>& device_ids,
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        const tt_metal::DispatchCoreConfig& dispatch_core_config,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
        bool init_profiler = true,
        bool initialize_fabric_and_dispatch_fw = true);

    void initialize(
        const DispatchCoreConfig& dispatch_core_config,
        uint8_t num_hw_cqs,
        const BankMapping& l1_bank_remap,
        size_t worker_l1_size,
        bool minimal = false);
    void teardown();

    // Control plane accessors
    void initialize_control_plane();
    tt::tt_fabric::ControlPlane& get_control_plane();
    void set_custom_fabric_topology(
        const std::string& mesh_graph_desc_file,
        const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    void set_default_fabric_topology();
    void set_fabric_config(
        tt_fabric::FabricConfig fabric_config,
        tt_fabric::FabricReliabilityMode reliability_mode =
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::optional<uint8_t> num_routing_planes = std::nullopt,
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED,
        tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED,
        tt_fabric::FabricManagerMode fabric_manager = tt_fabric::FabricManagerMode::DEFAULT,
        tt_fabric::FabricRouterConfig router_config = tt_fabric::FabricRouterConfig{});
    void initialize_fabric_config();
    void initialize_fabric_tensix_datamover_config();
    tt_fabric::FabricConfig get_fabric_config() const;
    tt_fabric::FabricReliabilityMode get_fabric_reliability_mode() const;
    const tt_fabric::FabricRouterConfig& get_fabric_router_config() const;

    const distributed::multihost::DistributedContext& global_distributed_context();
    const distributed::multihost::DistributedContext& full_world_distributed_context() const;
    std::shared_ptr<distributed::multihost::DistributedContext> get_distributed_context_ptr();

    // Fabric tensix configuration
    void set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config);
    tt_fabric::FabricTensixConfig get_fabric_tensix_config() const;

    // Fabric UDM mode configuration
    tt_fabric::FabricUDMMode get_fabric_udm_mode() const;

    // Fabric manager mode configuration
    tt_fabric::FabricManagerMode get_fabric_manager() const;

    // This is used to track the current thread's command queue id stack
    using CommandQueueIdStack = std::vector<uint8_t>;
    CommandQueueIdStack& get_command_queue_id_stack_for_thread();
    const CommandQueueIdStack& get_command_queue_id_stack_for_thread() const;

    // Utilities
    bool is_coord_in_range(CoreCoord coord, CoreType core_type);

    // Hang detection
    void on_dispatch_timeout_detected();

private:
    friend class tt::stl::Indestructible<MetalContext>;
    MetalContext();
    ~MetalContext();

    void clear_l1_state(ChipId device_id);
    void clear_dram_state(ChipId device_id);
    void clear_launch_messages_on_eth_cores(ChipId device_id);
    void construct_control_plane(const std::filesystem::path& mesh_graph_desc_path);
    void construct_control_plane();
    void initialize_control_plane_impl();  // Private implementation without mutex
    void teardown_fabric_config();
    void teardown_base_objects();

    void reset_cores(ChipId device_id);
    void assert_cores(ChipId device_id);

    // Returns the ERISC Launch Flag address
    uint32_t get_active_erisc_launch_flag_addr();
    // Returns true if metal firmware or a kernel is running on the virtual ethernet core
    bool erisc_app_still_running(ChipId device_id, CoreCoord virtual_core);
    // Send a message to exit the erisc app
    void erisc_send_exit_signal(ChipId device_id, CoreCoord virtual_core, bool is_idle_eth);

    // Functions used to init/run firmware on devices
    CoreCoord virtual_noc0_coordinate(ChipId device_id, uint8_t noc_index, CoreCoord coord);
    void generate_device_bank_to_noc_tables(ChipId device_id);
    void generate_worker_logical_to_virtual_map(ChipId device_id);
    void initialize_device_bank_to_noc_tables(
        ChipId device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        std::optional<CoreCoord> end_core);
    void initialize_worker_logical_to_virtual_tables(
        ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord start_core, CoreCoord end_core);
    void initialize_firmware(
        ChipId device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        dev_msgs::launch_msg_t::View launch_msg,
        dev_msgs::go_msg_t::ConstView go_msg,
        std::optional<CoreCoord> end_core = std::nullopt);
    void initialize_and_launch_firmware(ChipId device_id);
    dev_msgs::core_info_msg_t populate_core_info_msg(
        ChipId device_id, HalProgrammableCoreType programmable_core_type) const;

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
    std::unordered_set<uint64_t> firmware_built_keys_;
    std::mutex firmware_built_keys_mutex_;

    // Mutex to protect control_plane_ for thread-safe access
    std::mutex control_plane_mutex_;

    // Mutex to protect timeout detection for thread-safe access
    std::mutex dispatch_timeout_detection_mutex_;
    bool dispatch_timeout_detection_processed_ = false;

    // Written to device as part of FW init, device-specific
    std::unordered_map<ChipId, std::vector<int32_t>> dram_bank_offset_map_;
    std::unordered_map<ChipId, std::vector<int32_t>> l1_bank_offset_map_;
    std::unordered_map<ChipId, std::vector<uint16_t>> dram_bank_to_noc_xy_;
    std::unordered_map<ChipId, std::vector<uint16_t>> l1_bank_to_noc_xy_;

    std::unordered_map<ChipId, std::vector<uint8_t>> worker_logical_col_to_virtual_col_;
    std::unordered_map<ChipId, std::vector<uint8_t>> worker_logical_row_to_virtual_row_;

    llrt::RunTimeOptions rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;
    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;
    std::unique_ptr<inspector::Data> inspector_data_;
    std::unique_ptr<DPrintServer> dprint_server_;
    std::unique_ptr<WatcherServer> watcher_server_;
    std::unique_ptr<ProfilerStateManager> profiler_state_manager_;
    std::unique_ptr<DataCollector> data_collector_;
    std::unique_ptr<DeviceManager> device_manager_;

    std::array<std::unique_ptr<DispatchMemMap>, static_cast<size_t>(CoreType::COUNT)> dispatch_mem_map_;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;
    tt_fabric::FabricConfig fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    tt_fabric::FabricTensixConfig fabric_tensix_config_ = tt_fabric::FabricTensixConfig::DISABLED;
    tt_fabric::FabricUDMMode fabric_udm_mode_ = tt_fabric::FabricUDMMode::DISABLED;
    tt_fabric::FabricRouterConfig fabric_router_config_ = tt_fabric::FabricRouterConfig{};
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;
    std::shared_ptr<distributed::multihost::DistributedContext> compute_only_distributed_context_;

    // We are using a thread_local to allow each thread to have its own command queue id stack.
    // This not only allows consumers to set active command queue for a thread
    // but to also easily push/pop ids to temporarily change the current cq id.
    static thread_local CommandQueueIdStack command_queue_id_stack_for_thread_;

    // Strict system health mode requires (expects) all links/devices to be live. When enabled, it
    // is expected that any downed devices/links will result in some sort of error condition being
    // reported. When set to false, the control plane is free to instantiate fewer routing planes
    // according to which links are available.
    tt_fabric::FabricReliabilityMode fabric_reliability_mode_ = tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    uint8_t num_fabric_active_routing_planes_ = 0;
    std::map<tt_fabric::FabricNodeId, ChipId> logical_mesh_chip_id_to_physical_chip_id_mapping_;
    std::optional<std::string> custom_mesh_graph_desc_path_ = std::nullopt;
    tt_fabric::FabricManagerMode fabric_manager_ = tt_fabric::FabricManagerMode::DEFAULT;
};

}  // namespace tt::tt_metal

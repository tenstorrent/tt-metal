// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>
#include <vector>
#include <llrt/hal.hpp>  // Hal — full definition needed to call hal().get_*() via MetalContext
#include <llrt/rtoptions.hpp>
#include <impl/allocator/allocator_types.hpp>
#include <tt-metalium/allocator.hpp>
#include "impl/device/firmware/firmware_initializer.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "hostdevcommon/api/hostdevcommon/common_values.hpp"
#include "context_types.hpp"

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

class ContextDescriptor;
class DataCollector;
class DeviceManager;
class RiscFirmwareInitializer;
class dispatch_core_manager;
class DispatchQueryManager;
class DPrintServer;
class WatcherServer;
class DispatchMemMap;
class NOCDebugState;

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

    // Access the MetalContext for a given context id.
    // The context can be created beforehand using MetalContext::create_instance(). Otherwise an exception is thrown.
    // NOTE: To maintain legacy behavior, the default context id is automatically created if not already initialized
    static MetalContext& instance(ContextId context_id = DEFAULT_CONTEXT_ID);

    // Create a MetalContext instance which will use the given MetalEnv to facilitate runtime.
    static ContextId create_instance(MetalEnv& env_to_use);

    // Destroy the MetalContext for a given context id.
    static void destroy_instance(bool check_device_count = true, ContextId context_id = DEFAULT_CONTEXT_ID);

    // Destroy all MetalContext instances.
    static void destroy_all_instances(bool check_device_count = true);

    // Check if a MetalContext for a given context id exists.
    static bool instance_exists(ContextId context_id = DEFAULT_CONTEXT_ID);

    [[deprecated("Use MetalEnv instead")]] Cluster& get_cluster();
    [[deprecated("Use MetalEnv instead")]] llrt::RunTimeOptions& rtoptions();
    [[deprecated("Use MetalEnv instead")]] const Cluster& get_cluster() const;
    [[deprecated("Use MetalEnv instead")]] const llrt::RunTimeOptions& rtoptions() const;
    [[deprecated("Use MetalEnv instead")]] const Hal& hal() const;

    // Returns the MetalEnv instance assigned to this context.
    [[deprecated(
        "Use MetalEnv directly instead. This is a temporary workaround until all code is migrated to use "
        "MetalEnv.")]] tt::tt_metal::MetalEnv&
    get_env();

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
    bool is_device_manager_initialized() const { return device_manager_ != nullptr; }

    std::unique_ptr<NOCDebugState>& noc_debug_state() { return noc_debug_state_; }

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

    // Set fast dispatch mode and automatically reinitialize dispatch managers
    // This ensures dispatch/compute core allocations stay in sync with the mode
    void set_fast_dispatch_mode(bool enable);

    // Delegates to MetalEnv assigned to this context
    void initialize_control_plane();
    tt::tt_fabric::ControlPlane& get_control_plane();
    distributed::SystemMesh& get_system_mesh();

    const distributed::multihost::DistributedContext& global_distributed_context();
    const distributed::multihost::DistributedContext& full_world_distributed_context() const;
    std::shared_ptr<distributed::multihost::DistributedContext> get_distributed_context_ptr();

    // Fabric Settings
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

    void set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config);
    tt_fabric::FabricTensixConfig get_fabric_tensix_config() const;

    tt_fabric::FabricUDMMode get_fabric_udm_mode() const;

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

    // Construct MetalContext to use the given MetalEnv and assign it context id. The MetalEnv must not be
    // destroyed while its associated MetalContext instance is alive.
    MetalContext(ContextId context_id, tt::tt_metal::MetalEnv& metal_env);
    ~MetalContext();

    // This will create a MetalContext instance and create a default MetalEnv owned by the context.
    // Usually the MetalEnv is owned by the user, but in this case of legacy behaviour, the context will own it.
    // Caller holds the g_instance mutex.
    static ContextId create_default_instance_implicit_locked();

    // Register handlers -- caller already holds the instance lock
    static void register_handlers_locked();

    // Reinitialize dispatch managers when transitioning dispatch modes (SD<->FD)
    // This updates cached dispatch/compute core allocations to match current dispatch mode
    void reinitialize_dispatch_managers();
    void teardown_dispatch_state();

    void init_context_descriptor(int num_hw_cqs, size_t l1_small_size, size_t trace_region_size, size_t worker_l1_size);
    void init_risc_fw_context_descriptor(int num_hw_cqs, size_t worker_l1_size);

    bool initialized_ = false;
    bool force_reinit_ = false;

    uint8_t num_hw_cqs_ = 0;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;
    size_t worker_l1_size_ = 0;
    size_t worker_l1_unreserved_start_ = 0;
    size_t fw_compile_hash_ = 0;  // To check if FW recompilation is needed

    // Mutex to protect control_plane_ for thread-safe access
    std::mutex control_plane_mutex_;

    // Mutex to protect timeout detection for thread-safe access
    std::mutex dispatch_timeout_detection_mutex_;
    bool dispatch_timeout_detection_processed_ = false;

    // The MetalEnv is normally owned by the user
    // For the legacy code, we will initialize it in the MetalContext constructor and own the env
    // This means MetalContext will delete the env in the MetalContext destructor.
    tt::tt_metal::MetalEnv* env_;
    bool env_owned_ = false;
    ContextId context_id_;

    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;
    std::unique_ptr<inspector::Data> inspector_data_;
    std::unique_ptr<DPrintServer> dprint_server_;
    std::unique_ptr<WatcherServer> watcher_server_;
    std::unique_ptr<ProfilerStateManager> profiler_state_manager_;
    std::unique_ptr<DataCollector> data_collector_;
    std::unique_ptr<DeviceManager> device_manager_;
    std::unique_ptr<NOCDebugState> noc_debug_state_;
    // The context descriptor used for runtime components.
    std::shared_ptr<ContextDescriptor> context_descriptor_;
    // The context descriptor used for risc firmware only. L1/trace size/fabric settings were not known
    // at the time of creating this descriptor.
    std::shared_ptr<ContextDescriptor> risc_fw_context_descriptor_;
    std::unique_ptr<RiscFirmwareInitializer> risc_firmware_initializer_;
    std::unordered_set<InitializerKey> risc_fw_init_done_;

    std::array<std::unique_ptr<DispatchMemMap>, static_cast<size_t>(CoreType::COUNT)> dispatch_mem_map_;

    // We are using a thread_local to allow each thread to have its own command queue id stack.
    // This not only allows consumers to set active command queue for a thread
    // but to also easily push/pop ids to temporarily change the current cq id.
    static thread_local CommandQueueIdStack command_queue_id_stack_for_thread_;
};

}  // namespace tt::tt_metal

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <mutex>
#include <future>
#include <set>
#include <vector>
#include <unordered_set>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>

#include "metal_context.hpp"
#include "core_coord.hpp"
#include "device/firmware/risc_firmware_initializer.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "firmware_capability.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "fabric/channel_trimming_export.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "allocator/l1_banking_allocator.hpp"
#include "debug/dprint_server.hpp"
#include "debug/inspector/inspector.hpp"

#include <umd/device/types/xy_pair.hpp>
#include "debug/inspector/data.hpp"
#include "debug/noc_logging.hpp"
#include "debug/watcher_server.hpp"
#include "debug/noc_debugging.hpp"
#include "dispatch/topology.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "profiler/profiler_state_manager.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/llrt.hpp"
#include <experimental/fabric/control_plane.hpp>
#include <experimental/mock_device.hpp>
#include "context/context_descriptor.hpp"
#include "device/device_manager.hpp"
#include <distributed_context.hpp>
#include <experimental/fabric/fabric.hpp>
#include <system_mesh.hpp>

#include <tt_metal.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "dispatch/data_collector.hpp"

#include <dispatch/dispatch_query_manager.hpp>
#include <dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include <dispatch/dispatch_mem_map.hpp>
#include "common/executor.hpp"

namespace tt::tt_metal {

namespace {
// Helper function to validate worker_l1_size, also updates it if it's 0.
void validate_worker_l1_size(size_t& worker_l1_size, Hal& hal) {
    if (worker_l1_size == 0) {
        worker_l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    }
    size_t max_worker_l1_size = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                                hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) -
                                hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    TT_FATAL(
        worker_l1_size <= max_worker_l1_size,
        "Worker L1 size {} is larger than max size {}",
        worker_l1_size,
        max_worker_l1_size);
}

// Construct compute-only distributed context by filtering out switch meshes
std::shared_ptr<distributed::multihost::DistributedContext> construct_compute_only_distributed_context(
    MetalContext& metal_context) {
    const auto& global_context = distributed::multihost::DistributedContext::get_current_world();
    if (*global_context->size() == 1) {
        return global_context;
    }

    // Get all compute mesh IDs (excludes switches) from control plane mesh graph
    const auto& mesh_graph = metal_context.get_control_plane().get_mesh_graph();

    // If there are no switch meshes, return the global context directly
    if (mesh_graph.get_switch_ids().empty()) {
        return global_context;
    }

    const auto& compute_mesh_ids = mesh_graph.get_mesh_ids();

    // Get global logical bindings to map ranks to mesh IDs
    const auto& global_logical_bindings = metal_context.get_control_plane().get_global_logical_bindings();

    // Collect all MPI ranks for compute meshes only
    std::unordered_set<int> compute_mpi_ranks;
    for (const auto& [rank, mesh_binding] : global_logical_bindings) {
        const auto& [mesh_id, _] = mesh_binding;
        // Check if this mesh_id is a compute mesh (not a switch)
        if (std::find(compute_mesh_ids.begin(), compute_mesh_ids.end(), mesh_id) != compute_mesh_ids.end()) {
            compute_mpi_ranks.insert(rank.get());
        }
    }

    // If no compute meshes found, fall back to host_local_context
    if (compute_mpi_ranks.empty()) {
        TT_THROW("No compute meshes found in mesh graph.");
    }

    // Convert to sorted vector for create_sub_context
    std::vector<int> compute_ranks_vec(compute_mpi_ranks.begin(), compute_mpi_ranks.end());
    std::sort(compute_ranks_vec.begin(), compute_ranks_vec.end());

    // Check if current rank is in compute ranks
    int current_rank = *global_context->rank();
    bool is_current_rank_in_compute =
        std::find(compute_ranks_vec.begin(), compute_ranks_vec.end(), current_rank) != compute_ranks_vec.end();

    // If current rank is not in compute ranks (e.g., host only has switches), return host_local_context
    if (!is_current_rank_in_compute) {
        return metal_context.get_control_plane().get_host_local_context();
    }

    // Create sub-context with only compute mesh ranks
    return global_context->create_sub_context(compute_ranks_vec);
}

}  // namespace

void MetalContext::initialize_device_manager(
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    bool init_profiler,
    bool initialize_fabric_and_dispatch_fw) {
    initialize(dispatch_core_config, num_hw_cqs, {l1_bank_remap.begin(), l1_bank_remap.end()}, worker_l1_size);
    init_context_descriptor(num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size);
    device_manager_->initialize(device_ids, init_profiler, initialize_fabric_and_dispatch_fw, context_descriptor_);
}

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config,
    uint8_t num_hw_cqs,
    const BankMapping& l1_bank_remap,
    size_t worker_l1_size,
    bool minimal) {
    ZoneScoped;

    // Workaround for galaxy, need to always re-init
    if (rtoptions_.get_force_context_reinit() or cluster_->is_galaxy_cluster()) {
        force_reinit_ = true;
    }
    // Settings that affect FW build can also trigger a re-initialization
    const size_t fw_compile_hash = std::hash<std::string>{}(rtoptions_.get_compile_hash_string());
    validate_worker_l1_size(worker_l1_size, *hal_);
    if (initialized_) {
        if (dispatch_core_config_ != dispatch_core_config or num_hw_cqs != num_hw_cqs_ or
            worker_l1_size_ != worker_l1_size or l1_bank_remap != l1_bank_remap_ or
            fw_compile_hash != fw_compile_hash_) {
            log_warning(tt::LogAlways, "Closing and re-initializing MetalContext with new parameters.");
            teardown();
        } else {
            // Re-init request with the same parameters, do nothing unless force re-init requested.
            if (force_reinit_) {
                force_reinit_ = false;
                log_debug(
                    tt::LogAlways,
                    "Closing and re-initializing MetalContext with same parameters due to force_reinit flag.");
                teardown();
            } else {
                return;
            }
        }
    }

    initialized_ = true;
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;
    worker_l1_size_ = worker_l1_size;
    l1_bank_remap_ = l1_bank_remap;
    fw_compile_hash_ = fw_compile_hash;
    std::uint32_t max_alignment = std::max(hal_->get_alignment(HalMemType::DRAM), hal_->get_alignment(HalMemType::L1));
    worker_l1_unreserved_start_ = tt::align(
        hal_->get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal_->get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size_,
        max_alignment);

    // Initialize inspector
    inspector_data_ = Inspector::initialize();
    // Set fw_compile_hash for Inspector RPC build environment info
    Inspector::set_build_env_fw_compile_hash(fw_compile_hash);

    // Reset timeout detection state
    dispatch_timeout_detection_processed_ = false;

    // Initialize dispatch state
    bool is_galaxy_cluster = cluster_->is_galaxy_cluster();
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs, hal(), is_galaxy_cluster);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs, hal(), is_galaxy_cluster);
    // Initialize debug servers. Attaching individual devices done below
    rtoptions_.resolve_fabric_node_ids_to_chip_ids(this->get_control_plane());
    if (rtoptions_.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        TT_FATAL(!rtoptions_.get_profiler_enabled(), "Both DPRINT and Profiler cannot be enabled at the same time.");
        rtoptions_.set_disable_dma_ops(true);  // DMA is not thread-safe
        dprint_server_ = std::make_unique<DPrintServer>(rtoptions_);
    }
    watcher_server_ =
        std::make_unique<WatcherServer>();  // Watcher server always created, since we use it to register kernels
    noc_debug_state_ = std::make_unique<NOCDebugState>();

    if (rtoptions_.get_experimental_noc_debug_dump_enabled()) {
        TT_FATAL(
            !rtoptions_.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint),
            "Both DPRINT and NOC debug dump cannot be enabled at the same time.");
        TT_FATAL(
            !rtoptions_.get_watcher_enabled(), "Both Watcher and NOC debug dump cannot be enabled at the same time.");
    }

    if (rtoptions_.get_profiler_enabled()) {
        TT_FATAL(cluster_->arch() != ARCH::QUASAR, "Device profiler is not yet supported on Quasar.");
        profiler_state_manager_ = std::make_unique<ProfilerStateManager>();
    }

    data_collector_ = std::make_unique<DataCollector>();

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // Clear state, build FW
    auto all_devices = cluster_->all_chip_ids();
    std::set<ChipId> device_ids(all_devices.begin(), all_devices.end());

    auto get_dispatch_ignore_cores = [this](ChipId device_id) {
        std::unordered_set<CoreCoord> out;
        if (device_manager_ && device_manager_->is_initialized()) {
            const auto& dc = device_manager_->get_virtual_dispatch_cores(device_id);
            const auto& rc = device_manager_->get_virtual_dispatch_routing_cores(device_id);
            out.insert(dc.begin(), dc.end());
            out.insert(rc.begin(), rc.end());
        }
        return out;
    };
    init_risc_fw_context_descriptor(num_hw_cqs_, worker_l1_size_);
    risc_firmware_initializer_ = std::make_unique<RiscFirmwareInitializer>(
        risc_fw_context_descriptor_,
        std::bind(&MetalContext::get_control_plane, this),
        *dispatch_core_manager_,
        get_dispatch_ignore_cores);

    risc_firmware_initializer_->run_async_build_phase(device_ids);

    // Set internal routing for active ethernet cores, this is required for our FW to run
    if (has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC) &&
        cluster_->get_target_device_type() != tt::TargetDevice::Mock) {
        cluster_->set_internal_routing_info_for_ethernet_cores(get_control_plane(), true);
    }

    // Initialize debug tools, reset cores, init FW
    if (dprint_server_) {
        dprint_server_->attach_devices();
    }
    watcher_server_->init_devices();

    risc_firmware_initializer_->run_launch_phase(device_ids);

    // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
    // starts since it also writes to watcher mailboxes.
    watcher_server_->attach_devices();
}

// IMPORTANT: This function is registered as an atexit handler. Creating threads during program termination may cause
// undefined behavior. Do not create threads in this function or any functions it calls.
void MetalContext::reinitialize_dispatch_managers() {
    // Reinitialize dispatch core manager and query manager to pick up current dispatch mode
    // This refreshes cached dispatch/compute core allocations when transitioning SD<->FD
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config_, num_hw_cqs_);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs_);
}

void MetalContext::set_fast_dispatch_mode(bool enable) {
    rtoptions().set_fast_dispatch(enable);
    reinitialize_dispatch_managers();
}

void MetalContext::teardown() {
    ZoneScoped;

    if (!initialized_) {
        return;
    }
    initialized_ = false;

    auto all_devices = cluster_->all_chip_ids();

    if (data_collector_) {
        data_collector_->DumpData();
        data_collector_.reset();
    }

    if (dprint_server_) {
        if (cluster_->get_target_device_type() != tt::TargetDevice::Mock) {
            dprint_server_->detach_devices();
        }
        dprint_server_.reset();
        rtoptions_.set_disable_dma_ops(false);
    }

    if (cluster_->get_target_device_type() != tt::TargetDevice::Mock) {
        watcher_server_->detach_devices();
    }
    watcher_server_.reset();

    risc_firmware_initializer_->teardown(risc_fw_init_done_);
    risc_firmware_initializer_.reset();
    risc_fw_context_descriptor_.reset();

    if (profiler_state_manager_) {
        profiler_state_manager_.reset();
    }

    teardown_dispatch_state();

    // Clear dispatch, dispatch_s and prefetcher core info in inspector data
    Inspector::clear_all_core_info();
    // Deinitialize inspector
    inspector_data_.reset();

    system_mesh_.reset();
    control_plane_.reset();

    noc_debug_state_.reset();

    // Clear mock mode configuration if it was enabled
    if (experimental::is_mock_mode_registered()) {
        experimental::disable_mock_mode();
    }
}

// MetalContext destructor is private, so we can't use a unique_ptr to manage the instance.
std::atomic<MetalContext*> g_instance{nullptr};
std::mutex g_instance_mutex;
bool registered_atexit = false;

MetalContext& MetalContext::instance() {
    MetalContext* instance = g_instance.load(std::memory_order_acquire);
    if (instance) {
        // There is a potential race condition here if the instance is being torn down while this call is running or
        // while the caller is using the instance. We assume that if teardown is in progress, this call must be coming
        // from the teardown process (maybe on one of several threads) and is synchronized with the teardown.
        return *instance;
    }
    std::lock_guard lock(g_instance_mutex);
    // Check again in case another thread created the instance while we were waiting for the lock.
    instance = g_instance.load(std::memory_order_acquire);
    if (!instance) {
        instance = new MetalContext();
        g_instance.store(instance, std::memory_order_release);
        if (!registered_atexit) {
            std::atexit([]() {
                // Don't check device count because the destruction order is complicated and we can't guarantee that the
                // client isn't holding onto devices on process exit.
                MetalContext::destroy_instance(false);
            });
            registered_atexit = true;
        }
    }
    return *instance;
}

void MetalContext::destroy_instance(bool check_device_count) {
    // Don't lock g_instance_mutex to avoid deadlocking with instance() calls. Teardown should only ever be called from
    // one thread while no work is being done on the MetalContext.
    MetalContext* instance = g_instance.load(std::memory_order_acquire);
    if (!instance) {
        return;
    }
    if (check_device_count && instance->device_manager() && instance->device_manager()->is_initialized() &&
        !instance->device_manager()->get_all_active_devices().empty()) {
        TT_THROW("Cannot destroy MetalContext while devices are still open. Close all devices first.");
    }
    delete instance;
    // Only store to g_instance after the instance is deleted to allow MetalContext::instance() calls during teardown to
    // return the old instance.
    g_instance.store(nullptr, std::memory_order_release);
}

// Switch from mock mode to real hardware (requires all devices to be closed).
//
// This function is needed because MetalContext is a singleton with process lifetime, but mock mode
// configuration can be changed at runtime. The sequence of events is:
// 1. User calls API to enable mock device
// 2. MetalContext initialized in mock mode
// 3. User runs in mock mode
// 4. User calls API to disable mock device
// 5. Without this function, MetalContext would remain stuck in mock mode because the cluster/HAL
//    objects were already initialized with mock configuration.
//
// This function won't be needed when MetalContext has explicit lifetime management.
void MetalContext::reinitialize_for_real_hardware() {
    std::lock_guard<std::mutex> lock(reinitialization_mutex_);

    // Check if device_manager_ is initialized (MetalContext must be fully constructed)
    TT_FATAL(device_manager_ != nullptr, "Cannot reinitialize MetalContext before it is fully initialized");

    // Check if any devices are actually active (not just if MetalContext was initialized)
    auto active_devices = device_manager_->get_all_active_devices();
    TT_FATAL(
        active_devices.empty(),
        "Cannot switch to real hardware while {} device(s) are active. Close all devices first by calling "
        "MeshDevice::close() or letting the device go out of scope.",
        active_devices.size());

    log_info(tt::LogMetal, "Reinitializing MetalContext for real hardware (switching from mock mode)");

    rtoptions_.clear_mock_cluster_desc();
    teardown_base_objects();
    initialize_base_objects();

    teardown_dispatch_state();
    initialized_ = false;

    log_info(tt::LogMetal, "MetalContext reinitialized with real hardware");
}

void MetalContext::teardown_base_objects() {
    // Teardown in backward order of dependencies to avoid dereferencing uninitialized objects
    system_mesh_.reset();
    control_plane_.reset();
    distributed_context_.reset();
    // Destroy inspector before cluster to prevent RPC handlers from accessing destroyed cluster
    inspector_data_.reset();
    cluster_.reset();
    hal_.reset();
}

void MetalContext::teardown_dispatch_state() {
    for (auto& mem_map : dispatch_mem_map_) {
        if (mem_map) {
            mem_map.reset();
        }
    }
    device_manager_->reset_dispatch_topology();
    dispatch_query_manager_.reset();
    dispatch_core_manager_.reset();
}

void MetalContext::initialize_base_objects() {
    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    const auto platform_arch = get_platform_architecture(rtoptions_);

    cluster_ = std::make_unique<Cluster>(rtoptions_);

    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = rtoptions_.get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;

    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        rtoptions_.set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }

    distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
    hal_ = std::make_unique<Hal>(
        platform_arch,
        is_base_routing_fw_enabled,
        rtoptions_.get_enable_2_erisc_mode(),
        get_profiler_dram_bank_size_for_hal_allocation(rtoptions_));

    rtoptions_.ParseAllFeatureEnv(*hal_);
    cluster_->set_hal(hal_.get());
}

MetalContext::MetalContext() {
    // Check if mock mode was configured via API (before env vars take effect)
    if (auto mock_cluster_desc = experimental::get_mock_cluster_desc()) {
        rtoptions_.set_mock_cluster_desc(*mock_cluster_desc);
        log_info(tt::LogMetal, "Using programmatically configured mock mode: {}", *mock_cluster_desc);
    }

    // If a custom fabric mesh graph descriptor is specified as an RT Option, use it by default
    // to initialize the control plane.
    if (rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_.get_custom_fabric_mesh_graph_desc_path();
    }

    initialize_base_objects();

    device_manager_ = std::make_unique<DeviceManager>();
}

const distributed::multihost::DistributedContext& MetalContext::full_world_distributed_context() const {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return *distributed_context_;
}

const distributed::multihost::DistributedContext& MetalContext::global_distributed_context() {
    // If control plane is not initilazed, return the global distributed context
    if (!control_plane_) {
        return *distributed_context_;
    }
    // Lazy initilazation of compute only distributed context
    if (!compute_only_distributed_context_) {
        compute_only_distributed_context_ = construct_compute_only_distributed_context(*this);
    }
    return *compute_only_distributed_context_;
}

std::shared_ptr<distributed::multihost::DistributedContext> MetalContext::get_distributed_context_ptr() {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return distributed_context_;
}

MetalContext::~MetalContext() {
    teardown();
    device_manager_.reset();
    teardown_base_objects();
}

llrt::RunTimeOptions& MetalContext::rtoptions() { return rtoptions_; }

Cluster& MetalContext::get_cluster() {
    TT_FATAL(cluster_, "Trying to get cluster before initializing it.");
    return *cluster_;
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const { return rtoptions_; }

const Cluster& MetalContext::get_cluster() const {
    TT_FATAL(cluster_, "Trying to get cluster before initializing it.");
    return *cluster_;
}

const Hal& MetalContext::hal() const {
    TT_FATAL(hal_, "Trying to get hal before initializing it.");
    return *hal_;
}

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before initializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before initializing it.");
    return *dispatch_query_manager_;
}

const DispatchMemMap& MetalContext::dispatch_mem_map() const {
    return dispatch_mem_map(get_core_type_from_config(dispatch_core_config_));
}

const DispatchMemMap& MetalContext::dispatch_mem_map(const CoreType& core_type) const {
    const auto& mem_map = dispatch_mem_map_[enchantum::to_underlying(core_type)];
    TT_FATAL(mem_map, "Tried to get dispatch_mem_map for {} before initializing it.", core_type);
    return *mem_map;
}

tt::tt_fabric::ControlPlane& MetalContext::get_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!control_plane_) {
        // Initialize control plane (creates stub for mock devices)
        this->initialize_control_plane_impl();
    }
    return *control_plane_;
}

distributed::SystemMesh& MetalContext::get_system_mesh() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!system_mesh_) {
        if (!control_plane_) {
            this->initialize_control_plane_impl();
        }
        system_mesh_ = std::unique_ptr<distributed::SystemMesh>(new distributed::SystemMesh(*control_plane_));
    }
    return *system_mesh_;
}

void MetalContext::set_custom_fabric_topology(
    const std::string& mesh_graph_desc_file,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    // Set the user specified mesh graph descriptor file and FabricNodeID to physical chip mapping.
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    custom_mesh_graph_desc_path_ = mesh_graph_desc_file;
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalContext::set_default_fabric_topology() {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    // Reset the system mesh and control plane, since they were initialized with custom parameters.
    system_mesh_.reset();
    control_plane_.reset();
    // Set the mesh graph descriptor file to the default value and clear the custom FabricNodeId to physical chip
    // mapping.
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.clear();

    if (rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_.get_custom_fabric_mesh_graph_desc_path();
    } else {
        custom_mesh_graph_desc_path_ = std::nullopt;
    }
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalContext::teardown_fabric_config() {
    this->fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    this->cluster_->configure_ethernet_cores_for_fabric_routers(this->fabric_config_);
    this->num_fabric_active_routing_planes_ = 0;
    // if (!rtoptions_.get_erisc_iram_env_var_enabled()) {
    //     rtoptions_.set_erisc_iram_enabled(false);
    // }
    // Stub control plane for mock devices will make this a no-op
    this->get_control_plane().clear_fabric_context();
}

void MetalContext::set_fabric_config(
    const tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    tt_fabric::FabricUDMMode fabric_udm_mode,
    tt_fabric::FabricManagerMode fabric_manager,
    tt_fabric::FabricRouterConfig router_config) {
    // Changes to fabric force a re-init. TODO: We should supply the fabric config in the same way as the dispatch
    // config, not through this function exposed in the detail API.
    force_reinit_ = true;

    // Export channel trimming capture data before fabric config changes.
    // Must happen while fabric_config_ is still active and fabric context is alive.
    bool is_tearing_down_fabric = fabric_config == tt_fabric::FabricConfig::DISABLED &&
        this->fabric_config_ != tt_fabric::FabricConfig::DISABLED;
    if (is_tearing_down_fabric) {
        tt::tt_fabric::export_channel_trimming_capture();
    }

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED ||
        fabric_config == tt_fabric::FabricConfig::DISABLED) {
        this->fabric_config_ = fabric_config;
        this->fabric_reliability_mode_ = reliability_mode;
    } else {
        TT_FATAL(
            this->fabric_config_ == fabric_config,
            "Tried to override previous value of fabric config: {}, with: {}",
            this->fabric_config_,
            fabric_config);
    }

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        if (num_routing_planes.has_value()) {
            log_warning(
                tt::LogMetal,
                "Got num_routing_planes while disabling fabric, ignoring it and disabling all active routing planes");
        }

        this->teardown_fabric_config();
        return;
    }

    bool enable_erisc_iram =
        !rtoptions_.get_erisc_iram_env_var_enabled() || !rtoptions_.get_erisc_iram_env_var_disabled();
    rtoptions_.set_erisc_iram_enabled(enable_erisc_iram);

    if (num_routing_planes.has_value() && num_routing_planes.value() < this->num_fabric_active_routing_planes_) {
        log_warning(
            tt::LogMetal,
            "Got num_routing_planes: {}, which is less than current value: {}, ignoring the override",
            num_routing_planes.value(),
            this->num_fabric_active_routing_planes_);
        return;
    }

    // if num_routing_planes is not specified, use max available number of routing planes
    // ideally the highest value should be the maximum number of eth cores in a direction across all chips
    const auto new_val = std::max(
        this->num_fabric_active_routing_planes_, num_routing_planes.value_or(std::numeric_limits<uint8_t>::max()));
    if (new_val != this->num_fabric_active_routing_planes_ && this->num_fabric_active_routing_planes_ > 0) {
        log_info(
            tt::LogMetal,
            "Overriding the number of routing planes to activate from {} to {}",
            this->num_fabric_active_routing_planes_,
            new_val);
    }
    this->num_fabric_active_routing_planes_ = new_val;

    // Set the fabric tensix config
    this->set_fabric_tensix_config(fabric_tensix_config);
    this->fabric_udm_mode_ = fabric_udm_mode;
    this->fabric_manager_ = fabric_manager;
    this->fabric_router_config_ = router_config;

    // Reinitialize control plane with updated fabric settings
    if (control_plane_ != nullptr) {
        log_info(
            tt::LogMetal,
            "Fabric config changed from {} to {}, reinitializing control plane",
            this->get_control_plane().get_fabric_config(),
            this->fabric_config_);
        system_mesh_.reset();
        this->initialize_control_plane_impl();
    }
}

void MetalContext::initialize_fabric_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->cluster_->configure_ethernet_cores_for_fabric_routers(
        this->fabric_config_, this->num_fabric_active_routing_planes_);
    auto& control_plane = this->get_control_plane();
    control_plane.configure_routing_tables_for_fabric_ethernet_channels();
}

void MetalContext::initialize_fabric_tensix_datamover_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    if (cluster_->get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }

    // Initialize fabric tensix config after routing tables are configured and devices are available
    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        auto& control_plane = this->get_control_plane();
        control_plane.initialize_fabric_tensix_datamover_config();
    }
}

tt_fabric::FabricConfig MetalContext::get_fabric_config() const { return fabric_config_; }

tt_fabric::FabricReliabilityMode MetalContext::get_fabric_reliability_mode() const { return fabric_reliability_mode_; }

const tt_fabric::FabricRouterConfig& MetalContext::get_fabric_router_config() const { return fabric_router_config_; }

void MetalContext::set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config) {
    fabric_tensix_config_ = fabric_tensix_config;
}

tt_fabric::FabricTensixConfig MetalContext::get_fabric_tensix_config() const { return fabric_tensix_config_; }

tt_fabric::FabricUDMMode MetalContext::get_fabric_udm_mode() const { return fabric_udm_mode_; }

tt_fabric::FabricManagerMode MetalContext::get_fabric_manager() const { return fabric_manager_; }

void MetalContext::init_context_descriptor(
    int num_hw_cqs, size_t l1_small_size, size_t trace_region_size, size_t worker_l1_size) {
    context_descriptor_ = std::shared_ptr<ContextDescriptor>(new ContextDescriptor(
        *hal_,
        *cluster_,
        rtoptions_,
        fabric_config_,
        fabric_reliability_mode_,
        fabric_tensix_config_,
        fabric_udm_mode_,
        fabric_manager_,
        fabric_router_config_,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        worker_l1_size,
        dispatch_core_config_,
        l1_bank_remap_,
        rtoptions_.get_mock_cluster_desc_path()));
}

void MetalContext::init_risc_fw_context_descriptor(int num_hw_cqs, size_t worker_l1_size) {
    // Various settings are not known and not relevant for risc firmware
    risc_fw_context_descriptor_ = std::shared_ptr<ContextDescriptor>(new ContextDescriptor(
        *hal_,
        *cluster_,
        rtoptions_,
        tt::tt_fabric::FabricConfig::DISABLED,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricManagerMode::DEFAULT,
        tt::tt_fabric::FabricRouterConfig{},
        num_hw_cqs,
        /*l1_small_size=*/0,
        /*trace_region_size=*/0,
        worker_l1_size,
        {},
        {},
        rtoptions_.get_mock_cluster_desc_path()));
}

void MetalContext::construct_control_plane(const std::filesystem::path& mesh_graph_desc_path) {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_info(tt::LogDistributed, "Using custom Fabric Node Id to physical chip mapping.");
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
            *this->cluster_,
            this->rtoptions_,
            *hal_,
            *distributed_context_,
            mesh_graph_desc_path.string(),
            logical_mesh_chip_id_to_physical_chip_id_mapping_,
            this->fabric_config_,
            this->fabric_reliability_mode_,
            this->fabric_tensix_config_,
            this->fabric_udm_mode_,
            this->fabric_router_config_,
            this->fabric_manager_);
    } else {
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
            *this->cluster_,
            this->rtoptions_,
            *hal_,
            *distributed_context_,
            mesh_graph_desc_path.string(),
            this->fabric_config_,
            this->fabric_reliability_mode_,
            this->fabric_tensix_config_,
            this->fabric_udm_mode_,
            this->fabric_router_config_,
            this->fabric_manager_);
    }
}

void MetalContext::construct_control_plane() {
    // Use auto-discovery to generate mesh graph from physical system descriptor
    // This uses MeshGraph::generate_from_physical_system_descriptor which internally
    // uses map_mesh_to_physical to find a valid mapping
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_warning(
            tt::LogDistributed,
            "Custom Fabric Node Id to physical chip mapping provided but no mesh graph descriptor path. "
            "Mapping will be ignored. Please provide a custom mesh graph descriptor path for custom logical to "
            "physical mapping.");
    }
    log_info(tt::LogDistributed, "Constructing control plane using auto-discovery (no mesh graph descriptor).");
    control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
        *this->cluster_,
        this->rtoptions_,
        *hal_,
        *distributed_context_,
        this->fabric_config_,
        this->fabric_reliability_mode_,
        this->fabric_tensix_config_,
        this->fabric_udm_mode_,
        this->fabric_router_config_,
        this->fabric_manager_);
}

void MetalContext::initialize_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    initialize_control_plane_impl();
}

void MetalContext::initialize_control_plane_impl() {
    if (custom_mesh_graph_desc_path_.has_value()) {
        log_debug(tt::LogDistributed, "Using custom mesh graph descriptor: {}", custom_mesh_graph_desc_path_.value());
        std::filesystem::path mesh_graph_desc_path = std::filesystem::path(custom_mesh_graph_desc_path_.value());
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Custom mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());

        log_info(tt::LogDistributed, "Using custom mesh graph descriptor: {}", mesh_graph_desc_path.string());
        this->construct_control_plane(mesh_graph_desc_path);
        return;
    }
    // If no custom mesh graph descriptor use auto discovery to generate mesh graph
    log_info(tt::LogDistributed, "Using auto discovery to generate mesh graph.");

    if (*distributed_context_->size() == 1) {
        this->construct_control_plane();
    } else {
        auto cluster_type = cluster_->get_cluster_type();
        auto fabric_type = tt::tt_fabric::get_fabric_type(this->fabric_config_, cluster_->is_ubb_galaxy());
        std::filesystem::path mesh_graph_desc_path =
            tt::tt_fabric::MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
                cluster_type, rtoptions_.get_root_dir(), fabric_type);

        log_debug(tt::LogMetal, "Using mesh graph descriptor: {}", mesh_graph_desc_path);

        TT_FATAL(!mesh_graph_desc_path.empty(), "No mesh graph descriptor found for cluster type");
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());
        this->construct_control_plane(mesh_graph_desc_path);
    }
}

// Command queue id stack for thread
thread_local MetalContext::CommandQueueIdStack MetalContext::command_queue_id_stack_for_thread_;

MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() {
    return MetalContext::command_queue_id_stack_for_thread_;
}
const MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() const {
    return MetalContext::command_queue_id_stack_for_thread_;
}

bool MetalContext::is_coord_in_range(CoreCoord coord, CoreType core_type) {
    ChipId id = *cluster_->all_chip_ids().begin();
    if (core_type == CoreType::ACTIVE_ETH || core_type == CoreType::IDLE_ETH) {
        core_type = CoreType::ETH;
    }

    CoreCoord virtual_coord = cluster_->get_virtual_coordinate_from_logical_coordinates(id, coord, core_type);
    return cluster_->is_ethernet_core(virtual_coord, id) || cluster_->is_worker_core(virtual_coord, id);
}

void MetalContext::on_dispatch_timeout_detected() {
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);

    if (!dispatch_timeout_detection_processed_) {
        dispatch_timeout_detection_processed_ = true;
        log_error(tt::LogMetal, "Timeout detected");
        // Serialize Inspector RPC data if enabled
        if (rtoptions_.get_serialize_inspector_on_dispatch_timeout()) {
            log_info(tt::LogMetal, "Serializing Inspector RPC data");
            Inspector::serialize_rpc();
        }

        // Execute command if specified (mostly used to call tt-triage when a timeout occurs)
        std::string command = rtoptions_.get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            log_info(tt::LogMetal, "Executing command: {}", command);

            int result = std::system(command.c_str());

            if (result != 0) {
                log_warning(
                    tt::LogMetal, "Timeout command '{}' returned non-zero exit code: {}", command, WEXITSTATUS(result));
            }
        }
    }
}

}  // namespace tt::tt_metal

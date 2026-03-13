// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <mutex>
#include <future>
#include <set>
#include <vector>
#include <unordered_set>
#include <sys/wait.h>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>

#include "metal_context.hpp"
#include "context/metal_env_accessor.hpp"
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "context_descriptor.hpp"
#include "core_coord.hpp"
#include "device/firmware/risc_firmware_initializer.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal_types.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "debug/dprint_server.hpp"
#include "debug/inspector/inspector.hpp"

#include <umd/device/types/xy_pair.hpp>
#include "debug/inspector/data.hpp"
#include "debug/noc_logging.hpp"
#include "debug/watcher_server.hpp"
#include "debug/noc_debugging.hpp"
#include "common/filesystem_utils.hpp"
#include "dispatch/topology.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "profiler/profiler_state_manager.hpp"
#include <experimental/fabric/control_plane.hpp>
#include <experimental/mock_device.hpp>
#include "device/device_manager.hpp"
#include "device/firmware/risc_firmware_initializer.hpp"
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

namespace tt::tt_metal {

// MetalContext destructor is private, so we can't use a unique_ptr to manage the instance.
std::array<std::atomic<MetalContext*>, MAX_CONTEXT_COUNT> g_instances{};
std::atomic<MetalEnv*> g_default_env =
    nullptr;  // used for implicit creation of the default context -- legacy behaviour
std::mutex g_instance_mutex;
bool registered_handlers = false;

namespace {

void check_context_id(ContextId context_id) {
    TT_FATAL(context_id.get() >= 0, "context_id {} is invalid.", context_id);
    TT_FATAL(
        context_id.get() < MAX_CONTEXT_COUNT,
        "context_id {} is out of range (max {}).",
        context_id.get(),
        MAX_CONTEXT_COUNT);
}

ContextId find_free_context_id_locked() {
    // Slot 0 is reserved for the silicon context.
    for (int index = DEFAULT_CONTEXT_ID.get() + 1; index < MAX_CONTEXT_COUNT; ++index) {
        if (g_instances[index] == nullptr) {
            return ContextId{index};
        }
    }
    TT_THROW("Maximum MetalContext count ({}) reached.", MAX_CONTEXT_COUNT);
}

// Helper function to validate worker_l1_size, also updates it if it's 0.
void validate_worker_l1_size(size_t& worker_l1_size, const Hal& hal) {
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
    if (rtoptions().get_force_context_reinit() or get_cluster().is_galaxy_cluster()) {
        force_reinit_ = true;
    }
    // Fabric config changes (e.g. legacy DeviceManager enabling dispatch fabric) also force a re-init.
    if (MetalEnvAccessor(*env_).impl().consume_force_reinit()) {
        force_reinit_ = true;
    }

    const size_t fw_compile_hash = std::hash<std::string>{}(rtoptions().get_compile_hash_string());
    validate_worker_l1_size(worker_l1_size, hal());
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
    std::uint32_t max_alignment = std::max(hal().get_alignment(HalMemType::DRAM), hal().get_alignment(HalMemType::L1));
    worker_l1_unreserved_start_ = tt::align(
        hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size_,
        max_alignment);

    // Initialize inspector
    inspector_data_ = Inspector::initialize();
    // Set fw_compile_hash for Inspector RPC build environment info
    Inspector::set_build_env_fw_compile_hash(fw_compile_hash);

    // Reset timeout detection state
    dispatch_timeout_detection_processed_ = false;

    // Initialize dispatch state
    bool is_galaxy_cluster = get_cluster().is_galaxy_cluster();
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs, hal(), is_galaxy_cluster);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs, hal(), is_galaxy_cluster);
    // Initialize debug servers. Attaching individual devices done below
    rtoptions().resolve_fabric_node_ids_to_chip_ids(this->get_control_plane());
    rtoptions().resolve_mesh_coords_to_chip_ids(this->get_system_mesh());
    if (rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        TT_FATAL(!rtoptions().get_profiler_enabled(), "Both DPRINT and Profiler cannot be enabled at the same time.");
        rtoptions().set_disable_dma_ops(true);  // DMA is not thread-safe
        dprint_server_ = std::make_unique<DPrintServer>(rtoptions());
    }
    watcher_server_ =
        std::make_unique<WatcherServer>();  // Watcher server always created, since we use it to register kernels
    noc_debug_state_ = std::make_unique<NOCDebugState>();

    if (rtoptions().get_experimental_noc_debug_dump_enabled()) {
        TT_FATAL(
            !rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint),
            "Both DPRINT and NOC debug dump cannot be enabled at the same time.");
        TT_FATAL(
            !rtoptions().get_watcher_enabled(), "Both Watcher and NOC debug dump cannot be enabled at the same time.");
    }

    if (rtoptions().get_profiler_enabled()) {
        TT_FATAL(hal().get_arch() != ARCH::QUASAR, "Device profiler is not yet supported on Quasar.");
        profiler_state_manager_ = std::make_unique<ProfilerStateManager>();
    }

    data_collector_ = std::make_unique<DataCollector>();

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // Clear state, build FW
    auto all_devices = get_cluster().all_chip_ids();
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
        get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
        get_cluster().set_internal_routing_info_for_ethernet_cores(this->get_control_plane(), true);
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

    if (data_collector_) {
        data_collector_->DumpData();
        data_collector_.reset();
    }

    if (dprint_server_) {
        if (get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
            dprint_server_->detach_devices();
        }
        dprint_server_.reset();
        rtoptions().set_disable_dma_ops(false);
    }

    if (get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
        watcher_server_->detach_devices();
    }
    watcher_server_.reset();

    if (risc_firmware_initializer_) {
        risc_firmware_initializer_->teardown(risc_fw_init_done_);
        risc_firmware_initializer_.reset();
    }
    risc_fw_context_descriptor_.reset();

    if (profiler_state_manager_) {
        profiler_state_manager_.reset();
    }

    teardown_dispatch_state();

    // Clear dispatch, dispatch_s and prefetcher core info in inspector data
    Inspector::clear_all_core_info();
    // Destroy inspector before cluster to prevent RPC handlers from accessing destroyed cluster
    inspector_data_.reset();

    noc_debug_state_.reset();
}

bool MetalContext::instance_exists(ContextId context_id) {
    check_context_id(context_id);
    return g_instances[context_id.get()].load(std::memory_order_acquire) != nullptr;
}

MetalContext& MetalContext::instance(ContextId context_id) {
    check_context_id(context_id);
    int index = context_id.get();
    MetalContext* instance = g_instances[index].load(std::memory_order_acquire);
    if (instance) {
        // There is a potential race condition here if the instance is being torn down while this call is running or
        // while the caller is using the instance. We assume that if teardown is in progress, this call must be coming
        // from the teardown process (maybe on one of several threads) and is synchronized with the teardown.
        return *instance;
    }
    std::lock_guard lock(g_instance_mutex);
    // Check again in case another thread created the instance while we were waiting for the lock.
    instance = g_instances[index].load(std::memory_order_acquire);
    if (!instance) {
        // SILICON_CONTEXT_ID is implicitly created to match legacy behaviour
        TT_FATAL(
            context_id == DEFAULT_CONTEXT_ID,
            "No MetalContext instance for context_id {}. Create one via create_instance().",
            context_id);
        create_default_instance_implicit_locked();
        register_handlers_locked();
        instance = g_instances[DEFAULT_CONTEXT_ID.get()].load(std::memory_order_acquire);
    }
    return *instance;
}

ContextId MetalContext::create_default_instance_implicit_locked() {
    if (g_instances[DEFAULT_CONTEXT_ID.get()].load(std::memory_order_acquire) != nullptr) {
        TT_THROW("Only one silicon MetalContext instance may exist; context_id 0 is already in use.");
    }

    MetalEnvDescriptor desc{};
    if (auto mock_cluster_desc = experimental::get_mock_cluster_desc()) {
        log_info(tt::LogMetal, "Using programmatically configured mock mode: {}", *mock_cluster_desc);
        desc = MetalEnvDescriptor(*mock_cluster_desc);
    }
    g_default_env = new MetalEnv(std::move(desc));
    MetalContext* instance = new MetalContext(DEFAULT_CONTEXT_ID, *g_default_env);
    // Set the env_owned_ to true so the MetalContext destructor will delete the env_
    instance->env_owned_ = true;

    g_instances[DEFAULT_CONTEXT_ID.get()].store(instance, std::memory_order_release);
    return DEFAULT_CONTEXT_ID;
}

ContextId MetalContext::create_instance(MetalEnv& env_to_use) {
    std::lock_guard lock(g_instance_mutex);
    register_handlers_locked();

    // Allow only one instance connected to a silicon cluster
    if (!MetalEnvAccessor(env_to_use).impl().get_rtoptions().get_mock_enabled()) {
        if (g_instances[DEFAULT_CONTEXT_ID.get()].load(std::memory_order_acquire) != nullptr) {
            TT_THROW("Only one silicon MetalContext instance may exist; context_id 0 is already in use.");
        }
        MetalContext* instance = new MetalContext(DEFAULT_CONTEXT_ID, env_to_use);
        g_instances[DEFAULT_CONTEXT_ID.get()].store(instance, std::memory_order_release);
        return DEFAULT_CONTEXT_ID;
    }

    ContextId context_id = find_free_context_id_locked();
    MetalContext* instance = new MetalContext(context_id, env_to_use);
    g_instances[context_id.get()].store(instance, std::memory_order_release);
    return context_id;
}

void MetalContext::destroy_instance(bool check_device_count, ContextId context_id) {
    check_context_id(context_id);
    // Don't lock g_instance_mutex to avoid deadlocking with instance() calls. Teardown should only ever be called from
    // one thread while no work is being done on the MetalContext.
    int index = context_id.get();
    MetalContext* instance = g_instances[index].load(std::memory_order_acquire);
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
    g_instances[index].store(nullptr, std::memory_order_release);
}

void MetalContext::destroy_all_instances(bool check_device_count) {
    for (int index = 0; index < MAX_CONTEXT_COUNT; ++index) {
        destroy_instance(check_device_count, ContextId{index});
    }
}

void MetalContext::register_handlers_locked() {
    if (!registered_handlers) {
        std::atexit([]() {
            // Don't check device count because the destruction order is complicated and we can't guarantee that the
            // client isn't holding onto devices on process exit.
            MetalContext::destroy_all_instances(false);
        });
        registered_handlers = true;
    }
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

MetalContext::MetalContext(ContextId context_id, tt::tt_metal::MetalEnv& metal_env) :
    env_(&metal_env), context_id_(context_id) {
    check_context_id(context_id);
    device_manager_ = std::make_unique<DeviceManager>();
}

MetalContext::~MetalContext() {
    teardown();
    device_manager_.reset();
    if (env_owned_) {
        delete env_;
    }
    env_ = nullptr;
}

tt::tt_metal::MetalEnv& MetalContext::get_env() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return *env_;
}

// ─── Delegated accessors (Cluster / HAL / rtoptions) ─────────────────────────

llrt::RunTimeOptions& MetalContext::rtoptions() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_rtoptions();
}

Cluster& MetalContext::get_cluster() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_cluster();
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_rtoptions();
}

const Cluster& MetalContext::get_cluster() const {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_cluster();
}

const Hal& MetalContext::hal() const {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_hal();
}

// ─── Dispatch managers ────────────────────────────────────────────────────────

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

// ─── Fabric / control plane / system mesh — delegated to MetalEnv ─────────────

tt::tt_fabric::ControlPlane& MetalContext::get_control_plane() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_control_plane();
}

void MetalContext::initialize_control_plane() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().initialize_control_plane();
}

distributed::SystemMesh& MetalContext::get_system_mesh() {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_system_mesh();
}

void MetalContext::set_custom_fabric_topology(
    const std::string& mesh_graph_desc_file,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().set_custom_fabric_topology(
        mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void MetalContext::set_default_fabric_topology() {
    TT_FATAL(
        !device_manager_->is_initialized() || device_manager_->get_all_active_devices().empty(),
        "Modifying control plane requires no devices to be active");
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().set_default_fabric_topology();
}

void MetalContext::set_fabric_config(
    tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    tt_fabric::FabricUDMMode fabric_udm_mode,
    tt_fabric::FabricManagerMode fabric_manager,
    tt_fabric::FabricRouterConfig router_config) {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    bool updated = MetalEnvAccessor(*env_).impl().set_fabric_config(
        fabric_config,
        reliability_mode,
        num_routing_planes,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_manager,
        router_config);
    if (updated) {
        // Update the risc firmware context descriptor with the new fabric settings
        // as well due to transient state between the descriptor creation and the fabric config update
        if (risc_fw_context_descriptor_) {
            risc_fw_context_descriptor_->fabric_config_ = fabric_config;
            risc_fw_context_descriptor_->reliability_mode_ = reliability_mode;
            risc_fw_context_descriptor_->num_routing_planes_ = num_routing_planes;
            risc_fw_context_descriptor_->fabric_tensix_config_ = fabric_tensix_config;
            risc_fw_context_descriptor_->fabric_udm_mode_ = fabric_udm_mode;
            risc_fw_context_descriptor_->fabric_manager_ = fabric_manager;
        }
    }
}

void MetalContext::initialize_fabric_config() {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().initialize_fabric_config();
}

void MetalContext::initialize_fabric_tensix_datamover_config() {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().initialize_fabric_tensix_datamover_config();
}

tt_fabric::FabricConfig MetalContext::get_fabric_config() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_config();
}

tt_fabric::FabricReliabilityMode MetalContext::get_fabric_reliability_mode() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_reliability_mode();
}

const tt_fabric::FabricRouterConfig& MetalContext::get_fabric_router_config() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_router_config();
}

void MetalContext::set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config) {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvAccessor(*env_).impl().set_fabric_tensix_config(fabric_tensix_config);
}

tt_fabric::FabricTensixConfig MetalContext::get_fabric_tensix_config() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_tensix_config();
}

tt_fabric::FabricUDMMode MetalContext::get_fabric_udm_mode() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_udm_mode();
}

tt_fabric::FabricManagerMode MetalContext::get_fabric_manager() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_fabric_manager();
}

// ─── Distributed context — delegated to MetalEnv ─────────────────────────────

const distributed::multihost::DistributedContext& MetalContext::full_world_distributed_context() const {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().full_world_distributed_context();
}

const distributed::multihost::DistributedContext& MetalContext::global_distributed_context() {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().global_distributed_context();
}

std::shared_ptr<distributed::multihost::DistributedContext> MetalContext::get_distributed_context_ptr() {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    return MetalEnvAccessor(*env_).impl().get_distributed_context_ptr();
}

void MetalContext::init_context_descriptor(
    int num_hw_cqs, size_t l1_small_size, size_t trace_region_size, size_t worker_l1_size) {
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");
    MetalEnvImpl& env_accessor = MetalEnvAccessor(*env_).impl();
    std::string mock_cluster_desc_path =
        env_->get_descriptor().is_mock_device() ? env_->get_descriptor().mock_cluster_desc_path() : "";
    context_descriptor_ = std::make_shared<ContextDescriptor>(
        hal(),
        get_cluster(),
        rtoptions(),
        env_accessor.get_fabric_config(),
        env_accessor.get_fabric_reliability_mode(),
        env_accessor.get_fabric_tensix_config(),
        env_accessor.get_fabric_udm_mode(),
        env_accessor.get_fabric_manager(),
        env_accessor.get_fabric_router_config(),
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        worker_l1_size,
        dispatch_core_config_,
        l1_bank_remap_,
        mock_cluster_desc_path);
}

void MetalContext::init_risc_fw_context_descriptor(int num_hw_cqs, size_t worker_l1_size) {
    // env_ is assigned in MetalContext constructor. It should never be null at this point
    TT_FATAL(env_ != nullptr, "Missing MetalEnv for this MetalContext");

    // Fabric settings are used during risc init. In some cases, fabric is already running
    // and we don't want to reset the cores
    MetalEnvImpl& env_accessor = MetalEnvAccessor(*env_).impl();
    risc_fw_context_descriptor_ = std::make_shared<ContextDescriptor>(
        hal(),
        get_cluster(),
        rtoptions(),
        env_accessor.get_fabric_config(),
        env_accessor.get_fabric_reliability_mode(),
        env_accessor.get_fabric_tensix_config(),
        env_accessor.get_fabric_udm_mode(),
        env_accessor.get_fabric_manager(),
        env_accessor.get_fabric_router_config(),
        num_hw_cqs,
        /*l1_small_size=*/0,
        /*trace_region_size=*/0,
        worker_l1_size,
        DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t>{},
        rtoptions().get_mock_cluster_desc_path());
}

// ─── Command queue id stack ──────────────────────────────────────────────────

thread_local MetalContext::CommandQueueIdStack MetalContext::command_queue_id_stack_for_thread_;

MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() {
    return MetalContext::command_queue_id_stack_for_thread_;
}
const MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() const {
    return MetalContext::command_queue_id_stack_for_thread_;
}

bool MetalContext::is_coord_in_range(CoreCoord coord, CoreType core_type) {
    ChipId id = *get_cluster().all_chip_ids().begin();
    if (core_type == CoreType::ACTIVE_ETH || core_type == CoreType::IDLE_ETH) {
        core_type = CoreType::ETH;
    }

    CoreCoord virtual_coord = get_cluster().get_virtual_coordinate_from_logical_coordinates(id, coord, core_type);
    return get_cluster().is_ethernet_core(virtual_coord, id) || get_cluster().is_worker_core(virtual_coord, id);
}

void MetalContext::on_dispatch_timeout_detected() {
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);

    if (!dispatch_timeout_detection_processed_) {
        dispatch_timeout_detection_processed_ = true;
        log_error(tt::LogMetal, "Timeout detected");
        // Serialize Inspector RPC data if enabled
        if (rtoptions().get_serialize_inspector_on_dispatch_timeout()) {
            log_info(tt::LogMetal, "Serializing Inspector RPC data");
            Inspector::serialize_rpc();
        }

        // Execute command if specified (mostly used to call tt-triage when a timeout occurs)
        std::string command = rtoptions().get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            log_info(tt::LogMetal, "Executing command: {}", command);

            // std::system() passes the command through /bin/sh, which is required
            // because timeout commands may contain shell features (redirections,
            // pipes, etc.).
            int result = std::system(command.c_str());

            if (result != 0) {
                log_warning(
                    tt::LogMetal, "Timeout command '{}' returned non-zero exit code: {}", command, WEXITSTATUS(result));
            }
        }
    }
}

}  // namespace tt::tt_metal

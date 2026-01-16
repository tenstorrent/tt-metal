// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <mutex>
#include <future>
#include <vector>
#include <unordered_set>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>

#include "metal_context.hpp"
#include "core_coord.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "allocator/l1_banking_allocator.hpp"
#include "debug/dprint_server.hpp"
#include "debug/inspector/inspector.hpp"

#include <umd/device/types/xy_pair.hpp>
#include "debug/inspector/data.hpp"
#include "debug/noc_logging.hpp"
#include "debug/watcher_server.hpp"
#include "dispatch/topology.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "profiler/profiler_state_manager.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/llrt.hpp"
#include <experimental/fabric/control_plane.hpp>
#include "device/device_manager.hpp"
#include <distributed_context.hpp>
#include <experimental/fabric/fabric.hpp>

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
    device_manager_->initialize(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        l1_bank_remap,
        worker_l1_size,
        init_profiler,
        initialize_fabric_and_dispatch_fw);
}

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config,
    uint8_t num_hw_cqs,
    const BankMapping& l1_bank_remap,
    size_t worker_l1_size,
    bool minimal) {
    ZoneScoped;

    if (cluster_->get_target_device_type() == tt::TargetDevice::Mock) {
        TT_THROW(
            "Mock cluster cannot be initialized because there is no device. "
            "Mock clusters are only supported for testing control plane initialization without a device."
            "Please unset the TT_METAL_MOCK_CLUSTER_DESC_PATH environment variable.");
    }

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
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    // Need DispatchMemMap for both dispatch core types
    tt_metal::DispatchSettings::initialize(*cluster_);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs);
    dispatch_mem_map_[enchantum::to_underlying(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs);
    // Initialize debug servers. Attaching individual devices done below
    if (rtoptions_.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        TT_FATAL(!rtoptions_.get_profiler_enabled(), "Both DPRINT and Profiler cannot be enabled at the same time.");
        rtoptions_.set_disable_dma_ops(true);  // DMA is not thread-safe
        dprint_server_ = std::make_unique<DPrintServer>(rtoptions_);
    }
    watcher_server_ =
        std::make_unique<WatcherServer>();  // Watcher server always created, since we use it to register kernels

    if (rtoptions_.get_profiler_enabled()) {
        profiler_state_manager_ = std::make_unique<ProfilerStateManager>();
    }

    data_collector_ = std::make_unique<DataCollector>();

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // Clear state, build FW
    auto all_devices = cluster_->all_chip_ids();

    std::vector<std::shared_future<void>> futures;
    {
        ZoneScopedN("FW builds and Device Inits");

        futures.reserve(all_devices.size());

        // Launch async tasks for each device
        for (ChipId device_id : all_devices) {
            futures.emplace_back(detail::async([this, device_id, fw_compile_hash]() {
                // Clear L1/DRAM if requested
                if (rtoptions_.get_clear_l1()) {
                    clear_l1_state(device_id);
                }
                if (rtoptions_.get_clear_dram()) {
                    clear_dram_state(device_id);
                }
                [[maybe_unused]] int ai_clk = cluster_->get_device_aiclk(device_id);
                log_debug(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
                generate_device_bank_to_noc_tables(device_id);
                generate_worker_logical_to_virtual_map(device_id);

                // Create build env for this device, and build FW if it's not built already
                BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
                // fw_build_key is a combination of build_key and fw_compile_hash
                // If fw_compile_hash changes, the fw_build_key will change and FW will be rebuilt
                // if it's not already in firmware_built_keys_
                // Combine build_key and fw_compile_hash using XOR to create unique firmware build key
                // Uses full 64-bit fw_compile_hash for proper change detection
                uint64_t fw_build_key =
                    BuildEnvManager::get_instance().get_device_build_env(device_id).build_key() ^ fw_compile_hash;

                {
                    std::lock_guard<std::mutex> lock(firmware_built_keys_mutex_);
                    if (!firmware_built_keys_.contains(fw_build_key)) {
                        BuildEnvManager::get_instance().build_firmware(device_id);
                        firmware_built_keys_.insert(fw_build_key);
                    }
                }

                // Clear the entire launch message ring buffer on ethernet cores before application firmware is
                // activated. This is required since ethernet cores context switch between application and routing
                // firmware. If ERISC application firmware is activated before the launch messages are cleared, it can
                // enter an undefined state by reading a corrupted launch message. Routing firmware will never run in
                // this case, causing UMD issued transactions to hang.
                clear_launch_messages_on_eth_cores(device_id);
            }));
        }

        // Wait for all async tasks to complete
        for (auto& fut : futures) {
            fut.wait();
        }
    }

    // Populate FD topology across all devices
    if (rtoptions_.get_fast_dispatch()) {
        std::set<ChipId> all_devices_set(all_devices.begin(), all_devices.end());
        // TODO: enable this when dispatch init/teardown moves to MetalContext
        // populate_fd_kernels(all_devices_set, num_hw_cqs);
    }

    // Set internal routing for active ethernet cores, this is required for our FW to run
    if (has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        cluster_->set_internal_routing_info_for_ethernet_cores(true);
    }

    // Initialize debug tools, reset cores, init FW
    if (dprint_server_) {
        dprint_server_->attach_devices();
    }
    watcher_server_->init_devices();

    // Launch FW on each device sequentially, since a multithreaded launch leads to initialization hangs.
    // See https://github.com/tenstorrent/tt-metal/issues/35701
    {
        ZoneScopedN("Resets and FW Launch");

        // Launch async tasks for each device
        for (ChipId device_id : all_devices) {
            ClearNocData(device_id);

            reset_cores(device_id);

            initialize_and_launch_firmware(device_id);
        }
    }
    // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
    // starts since it also writes to watcher mailboxes.
    watcher_server_->attach_devices();

    // Register teardown function, but only once.
    if (not teardown_registered_) {
        std::atexit([]() { MetalContext::instance().teardown(); });
        teardown_registered_ = true;
    }
}

// IMPORTANT: This function is registered as an atexit handler. Creating threads during program termination may cause
// undefined behavior. Do not create threads in this function or any functions it calls.
void MetalContext::teardown() {
    ZoneScoped;

    if (!initialized_) {
        return;
    }
    initialized_ = false;

    auto all_devices = cluster_->all_chip_ids();
    // If simulator is enabled, force a teardown of active ethernet cores for WH
    if (rtoptions_.get_simulator_enabled()) {
        if (hal_->get_eth_fw_is_cooperative()) {
            for (ChipId device_id : all_devices) {
                for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                    CoreCoord virtual_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::ETH);
                    erisc_send_exit_signal(device_id, virtual_core, false);
                    while (erisc_app_still_running(device_id, virtual_core)) {
                    }
                }
            }
        }
    }

    // Set internal routing to false to exit active ethernet FW & go back to base FW
    cluster_->set_internal_routing_info_for_ethernet_cores(false);

    if (data_collector_) {
        data_collector_->DumpData();
        data_collector_.reset();
    }

    if (dprint_server_) {
        dprint_server_->detach_devices();
        dprint_server_.reset();
        rtoptions_.set_disable_dma_ops(false);
    }

    watcher_server_->detach_devices();
    watcher_server_.reset();
    for (ChipId device_id : all_devices) {
        assert_cores(device_id);

        cluster_->l1_barrier(device_id);
    }

    if (profiler_state_manager_) {
        profiler_state_manager_.reset();
    }

    for (auto& mem_map : dispatch_mem_map_) {
        if (mem_map) {
            mem_map.reset();
        }
    }

    dispatch_query_manager_.reset();
    dispatch_core_manager_.reset();
    tt::tt_metal::reset_topology_state();

    // Clear dispatch, dispatch_s and prefetcher core info in inspector data
    Inspector::clear_all_core_info();
    // Deinitialize inspector
    inspector_data_.reset();

    control_plane_.reset();
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

void MetalContext::teardown_base_objects() {
    // Teardown in backward order of dependencies to avoid dereferencing uninitialized objects
    distributed_context_.reset();
    // Destroy inspector before cluster to prevent RPC handlers from accessing destroyed cluster
    inspector_data_.reset();
    cluster_.reset();
    hal_.reset();
}

MetalContext::MetalContext() {
    // If a custom fabric mesh graph descriptor is specified as an RT Option, use it by default
    // to initialize the control plane.
    if (rtoptions_.is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_.get_custom_fabric_mesh_graph_desc_path();
    }

    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    const auto platform_arch = get_platform_architecture(rtoptions_);

    const auto initialize_objects = [&]() {
        hal_ = std::make_unique<Hal>(
            platform_arch,
            is_base_routing_fw_enabled,
            rtoptions_.get_enable_2_erisc_mode(),
            get_profiler_dram_bank_size_for_hal_allocation(rtoptions_));
        rtoptions_.ParseAllFeatureEnv(*hal_);
        cluster_ = std::make_unique<Cluster>(rtoptions_, *hal_);
        distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
    };

    initialize_objects();

    // Requires reinit with features disabled
    // This will maintain backward compatibility with clusters that have legacy firmware but it will cause a slowdown
    // during the first init
    if (!cluster_->verify_eth_fw_capability()) {
        rtoptions_.set_enable_2_erisc_mode(false);
        teardown_base_objects();
        initialize_objects();
    }

    // Initialize some container members to allow threadsafe operations on them later
    dram_bank_offset_map_.reserve(cluster_->all_chip_ids().size());
    l1_bank_offset_map_.reserve(cluster_->all_chip_ids().size());
    dram_bank_to_noc_xy_.reserve(cluster_->all_chip_ids().size());
    l1_bank_to_noc_xy_.reserve(cluster_->all_chip_ids().size());
    worker_logical_col_to_virtual_col_.reserve(cluster_->all_chip_ids().size());
    worker_logical_row_to_virtual_row_.reserve(cluster_->all_chip_ids().size());
    for (ChipId device_id : cluster_->all_chip_ids()) {
        dram_bank_offset_map_.emplace(device_id, std::vector<int32_t>{});
        l1_bank_offset_map_.emplace(device_id, std::vector<int32_t>{});
        dram_bank_to_noc_xy_.emplace(device_id, std::vector<uint16_t>{});
        l1_bank_to_noc_xy_.emplace(device_id, std::vector<uint16_t>{});
        worker_logical_col_to_virtual_col_.emplace(device_id, std::vector<uint8_t>{});
        worker_logical_row_to_virtual_row_.emplace(device_id, std::vector<uint8_t>{});
    }

    device_manager_ = std::make_unique<DeviceManager>();

    // We do need to call Cluster teardown at the end of the program, use atexit temporarily until we have clarity on
    // how MetalContext lifetime will work through the API.
    std::atexit([]() { MetalContext::instance().~MetalContext(); });
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

void MetalContext::clear_l1_state(ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = cluster_->get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_->write_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    // Clear erisc unreserved L1
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        static uint32_t zero_vec_size = hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = hal::get_erisc_l1_unreserved_base();

        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        cluster_->write_core(device_id, virtual_core, zero_vec, zero_vec_addr);
    }
    // TODO: clear idle eriscs as well
    cluster_->l1_barrier(device_id);
}

void MetalContext::clear_dram_state(ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", device_id);

    auto dram_size_per_channel = cluster_->get_soc_desc(device_id).dram_view_size;
    auto num_dram_channels = cluster_->get_soc_desc(device_id).get_num_dram_views();
    constexpr uint32_t start_address = 0;
    std::vector<uint8_t> zero_vec(dram_size_per_channel, 0);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        cluster_->write_dram_vec(zero_vec.data(), zero_vec.size(), device_id, channel, start_address);

        cluster_->dram_barrier(device_id);
    }
}

void MetalContext::clear_launch_messages_on_eth_cores(ChipId device_id) {
    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core, HalProgrammableCoreType programmable_core_type) {
        auto factory = hal_->get_dev_msgs_factory(programmable_core_type);
        std::vector<std::byte> init_launch_msg_data(
            dev_msgs::launch_msg_buffer_num_entries * factory.size_of<dev_msgs::launch_msg_t>(), std::byte{0});
        dev_msgs::go_msg_t go_msg = factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

        CoreCoord virtual_eth_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        cluster_->write_core(
            init_launch_msg_data.data(),
            init_launch_msg_data.size(),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        cluster_->write_core(
            go_msg.data(),
            go_msg.size(),
            {static_cast<size_t>(device_id), virtual_eth_core},
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG));
    };

    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        clear_ethernet_core(eth_core, HalProgrammableCoreType::ACTIVE_ETH);
    }
    for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
            continue;
        }
        clear_ethernet_core(eth_core, HalProgrammableCoreType::IDLE_ETH);
    }

    cluster_->l1_barrier(device_id);
}

tt::tt_fabric::ControlPlane& MetalContext::get_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!control_plane_) {
        this->initialize_control_plane_impl();
    }
    return *control_plane_;
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
    // Reset the control plane, since it was initialized with custom parameters.
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
}

void MetalContext::initialize_fabric_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->cluster_->configure_ethernet_cores_for_fabric_routers(
        this->fabric_config_, this->num_fabric_active_routing_planes_);
    auto& control_plane = this->get_control_plane();
    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        control_plane.initialize_fabric_context(this->fabric_config_, this->fabric_router_config_);
    }
    control_plane.configure_routing_tables_for_fabric_ethernet_channels(
        this->fabric_config_, this->fabric_reliability_mode_);
}

void MetalContext::initialize_fabric_tensix_datamover_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
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

void MetalContext::construct_control_plane(const std::filesystem::path& mesh_graph_desc_path) {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_info(tt::LogDistributed, "Using custom Fabric Node Id to physical chip mapping.");
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
            mesh_graph_desc_path.string(), logical_mesh_chip_id_to_physical_chip_id_mapping_);
    } else {
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
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
    control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>();
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
        auto fabric_type = tt::tt_fabric::get_fabric_type(this->fabric_config_);
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

void MetalContext::reset_cores(ChipId device_id) {
    ZoneScoped;
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<ChipId, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    if (has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        // Active ethernet
        if (hal_->get_eth_fw_is_cooperative()) {
            for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                CoreCoord virtual_core =
                    cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                if (erisc_app_still_running(device_id, virtual_core)) {
                    log_info(
                        tt::LogMetal,
                        "While initializing device {}, active ethernet dispatch core {} detected as still "
                        "running, issuing exit signal.",
                        device_id,
                        virtual_core.str());
                    erisc_send_exit_signal(device_id, virtual_core, false /* is_idle_eth */);
                    device_to_early_exit_cores[device_id].insert(virtual_core);
                }
            }
        } else {
            for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
                // Ensure exit to base firmware. Send this before assertion subordinate cores otherwise if we stop the
                // subordinates we could hang waiting for subordinates to finish
                CoreCoord virtual_core =
                    cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                if (rtoptions_.get_enable_2_erisc_mode()) {
                    erisc_send_exit_signal(
                        device_id, virtual_core, false /* is_idle_eth */);  // Stop any running erisc kernels
                    llrt::internal_::return_to_base_firmware_and_wait_for_heartbeat(device_id, virtual_core);
                }
                // Only send reset to subordinate cores
                // Assert all cores except ERISC0, which is running base firmware.
                tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
        }
    }
    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto& id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000;  // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(
                    id_and_cores.first, dev_msgs::RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error& e) {
                log_warning(
                    tt::LogAlways,
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW initialization...");
            }
        }
    }

    // Reset Tensix cores
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
        }
    }

    if (has_flag(
            tt::tt_metal::MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        // Reset idle ethernet cores
        for (const auto& logical_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::ALL);
        }
    }
    cluster_->l1_barrier(device_id);
}

void MetalContext::assert_cores(ChipId device_id) {
    auto dispatch_cores = get_virtual_dispatch_cores(device_id);
    auto routing_cores = get_virtual_dispatch_routing_cores(device_id);

    // Assert riscs on Tensix
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);

            if (!dispatch_cores.contains(worker_core) && !routing_cores.contains(worker_core)) {
                if (!hal_->get_eth_fw_is_cooperative() &&
                    this->get_control_plane().get_active_ethernet_cores(device_id, false).contains(logical_core)) {
                    // Cannot put these cores into reset because they are running base FW
                    // Below will return to base FW
                    continue;
                }
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), device_id);
            }
        }
    }

    if (!hal_->get_eth_fw_is_cooperative()) {
        // Assert riscs on active eth
        const auto assert_eth_core = [&](const CoreCoord& logical_eth_core) {
            CoreCoord virtual_eth_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
            // Ensure that the core has returned to base fw
            if (rtoptions_.get_enable_2_erisc_mode()) {
                llrt::internal_::return_to_base_firmware_and_wait_for_heartbeat(device_id, virtual_eth_core);
            }
            // Stop subordinate
            // Assert all cores except ERISC0, which is running base firmware.
            tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_eth_core), reset_val);
        };

        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            assert_eth_core(eth_core);
        }
    }
}

CoreCoord MetalContext::virtual_noc0_coordinate(ChipId device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = cluster_->get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y || cluster_->arch() == ARCH::BLACKHOLE) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    }  // Coordinate in Physical NOC0 Space. Convert to Virtual.
    coord = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, coord);
    // Derive virtual coord in noc_index space.
    CoreCoord virtual_coord = {
        hal_->noc_coordinate(noc_index, grid_size.x, coord.x), hal_->noc_coordinate(noc_index, grid_size.y, coord.y)};
    return virtual_coord;
}

void MetalContext::generate_device_bank_to_noc_tables(ChipId device_id) {
    // Create a dummp allocator to generatoe the bank/noc tables. Specifically, these depend on l1_bank_remap.
    auto config = L1BankingAllocator::generate_config(
        device_id,
        num_hw_cqs_,
        DEFAULT_L1_SMALL_SIZE,      // Not required for noc table gen
        DEFAULT_TRACE_REGION_SIZE,  // Not required for noc table gen
        worker_l1_unreserved_start_,
        l1_bank_remap_);
    const auto allocator = L1BankingAllocator(config);
    const auto& soc_d = cluster_->get_soc_desc(device_id);
    const size_t num_dram_banks = allocator.get_num_banks(BufferType::DRAM);
    dram_bank_offset_map_[device_id].clear();
    dram_bank_offset_map_[device_id].resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator.get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map_[device_id].clear();
    l1_bank_offset_map_[device_id].resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = cluster_->get_virtual_coordinate_from_logical_coordinates(
            device_id, allocator.get_logical_core_from_bank_id(bank_id), CoreType::WORKER);
        l1_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::L1, bank_id);
    }

    dram_bank_to_noc_xy_[device_id].clear();
    dram_bank_to_noc_xy_[device_id].reserve(hal_->get_num_nocs() * num_dram_banks);
    bool noc_translation_enabled = cluster_->get_cluster_desc()->get_noc_translation_table_en().at(device_id);
    bool dram_is_virtualized =
        noc_translation_enabled && (hal_->get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
    for (unsigned int noc = 0; noc < hal_->get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
            uint16_t noc_x, noc_y;
            CoreCoord dram_noc_coord =
                soc_d.get_preferred_worker_core_for_dram_view(allocator.get_dram_channel_from_bank_id(bank_id), noc);
            if (dram_is_virtualized) {
                noc_x = dram_noc_coord.x;
                noc_y = dram_noc_coord.y;
            } else {
                noc_x = hal_->noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord.x);
                noc_y = hal_->noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord.y);
            }
            uint16_t xy = ((noc_y << hal_->get_noc_addr_node_id_bits()) | noc_x) << hal_->get_noc_coord_reg_offset();
            dram_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }

    l1_bank_to_noc_xy_[device_id].clear();
    l1_bank_to_noc_xy_[device_id].reserve(hal_->get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < hal_->get_num_nocs(); noc++) {
        for (const auto& noc_coord : l1_noc_coord_per_bank) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, noc_coord);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal_->get_noc_addr_node_id_bits()) | noc_x) << hal_->get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }
}

void MetalContext::generate_worker_logical_to_virtual_map(ChipId device_id) {
    // Generate logical to virtual map for DRAM and L1 banks
    const auto& soc_desc = cluster_->get_soc_desc(device_id);
    auto tensix_grid_size = soc_desc.get_grid_size(CoreType::TENSIX);

    worker_logical_col_to_virtual_col_[device_id].clear();
    worker_logical_row_to_virtual_row_[device_id].clear();
    worker_logical_col_to_virtual_col_[device_id].reserve(tensix_grid_size.x);
    worker_logical_row_to_virtual_row_[device_id].reserve(tensix_grid_size.y);

    for (size_t x = 0; x < tensix_grid_size.x; x++) {
        worker_logical_col_to_virtual_col_[device_id].push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{x, 0}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .x);
    }
    for (size_t y = 0; y < tensix_grid_size.y; y++) {
        worker_logical_row_to_virtual_row_[device_id].push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{0, y}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .y);
    }
}

void MetalContext::initialize_device_bank_to_noc_tables(
    ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    std::optional<CoreCoord> end_core) {
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_[device_id].size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_[device_id].size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal_->get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT(
        (dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <=
            mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    if (end_core.has_value()) {
        // Multicast to all tensix cores in the range [virtual_core, end_core]
        auto start_core = virtual_core;
        cluster_->noc_multicast_write(
            dram_bank_to_noc_xy_[device_id].data(),
            dram_to_noc_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_->noc_multicast_write(
            l1_bank_to_noc_xy_[device_id].data(),
            l1_to_noc_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_->noc_multicast_write(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_->noc_multicast_write(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            l1_offset_addr);
    } else {
        // Unicast to single core
        cluster_->write_core(
            dram_bank_to_noc_xy_[device_id].data(),
            dram_to_noc_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_->write_core(
            l1_bank_to_noc_xy_[device_id].data(),
            l1_to_noc_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_->write_core(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_->write_core(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            l1_offset_addr);
    }
}

void MetalContext::initialize_worker_logical_to_virtual_tables(
    ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord start_core, CoreCoord end_core) {
    // Generate logical to virtual map for DRAM and L1 banks
    const auto& soc_desc = cluster_->get_soc_desc(device_id);
    const uint32_t logical_col_to_virtual_col_sz_in_bytes =
        worker_logical_col_to_virtual_col_[device_id].size() * sizeof(uint8_t);
    const uint8_t firmware_grid_size_x =
        tt::round_up(soc_desc.grid_size.x, 4);  // Ensure multiple of 4 for uint32_t alignment
    const uint32_t logical_row_to_virtual_row_sz_in_bytes =
        worker_logical_row_to_virtual_row_[device_id].size() * sizeof(uint8_t);
    const uint64_t logical_to_virtual_map_addr =
        hal_->get_dev_addr(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t logical_to_virtual_map_size =
        hal_->get_dev_size(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);

    TT_ASSERT(
        (firmware_grid_size_x + logical_row_to_virtual_row_sz_in_bytes) <= logical_to_virtual_map_size,
        "Size of logical to virtual map is greater than available space");

    uint64_t logical_col_to_virtual_col_addr = logical_to_virtual_map_addr;
    cluster_->noc_multicast_write(
        worker_logical_col_to_virtual_col_[device_id].data(),
        logical_col_to_virtual_col_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_col_to_virtual_col_addr);

    // Size of the data in the firmware is the full size of the grid, not the harvested size.
    // Therefore, we must adjust the address to account for the full grid size.
    uint64_t logical_row_to_virtual_row_addr = logical_to_virtual_map_addr + (firmware_grid_size_x * sizeof(uint8_t));
    cluster_->noc_multicast_write(
        worker_logical_row_to_virtual_row_[device_id].data(),
        logical_row_to_virtual_row_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_row_to_virtual_row_addr);
}

void MetalContext::initialize_firmware(
    ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    dev_msgs::launch_msg_t::View launch_msg,
    dev_msgs::go_msg_t::ConstView go_msg,
    std::optional<CoreCoord> end_core) {
    ZoneScoped;

    TT_FATAL(
        core_type != HalProgrammableCoreType::TENSIX or end_core.has_value(),
        "Tensix cores require end_core to be specified for bank to noc table initialization.");

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core, end_core);
    if (core_type == HalProgrammableCoreType::TENSIX) {
        // Only need to generate logical to virtual tables for Tensix cores, as only they run the firmware that
        // requires it.
        initialize_worker_logical_to_virtual_tables(device_id, core_type, virtual_core, end_core.value());
    }

    uint32_t core_type_idx = hal_->get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal_->get_processor_classes_count(core_type);
    auto jit_build_config =
        hal_->get_jit_build_config(core_type_idx, 0, 0);  // Only the first risc needs to be programmed

    const auto start_core = virtual_core;

    // Initialize each entry in the launch_msg ring buffer with the correct dispatch mode - Cores that don't get a valid
    // launch_message during program execution need to at least have the correct dispatch mode.
    // When using Fast Dispatch on Tensix:
    // dispatch cores (Tensix) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // Idle Eth cores configured with DISPATCH_MODE_HOST but not used
    // When using Fast Dispatch on Idle Eth:
    // dispatch cores (Idle Eth) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // When using Slow Dispatch, all cores initialized with DISPATCH_MODE_HOST
    size_t launch_msg_size = launch_msg.size();
    std::vector<std::byte> init_launch_msg_data(
        dev_msgs::launch_msg_buffer_num_entries * launch_msg_size, std::byte{0});
    auto prepare_initial_launch_msg = [&]() {
        for (size_t i = 0; i < dev_msgs::launch_msg_buffer_num_entries; ++i) {
            std::copy(
                launch_msg.data(),
                launch_msg.data() + launch_msg_size,
                init_launch_msg_data.data() + (i * launch_msg_size));
        }
    };
    const auto write_initial_go_launch_msg = [&]() {
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        uint32_t launch_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);
        uint32_t go_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        uint64_t launch_msg_buffer_read_ptr_addr =
            hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t go_message_index_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        // Unicast only to non-tensix cores
        if (core_type != HalProgrammableCoreType::TENSIX) {
            cluster_->write_core(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                tt_cxy_pair(device_id, virtual_core),
                launch_addr);
            cluster_->write_core(go_msg.data(), go_msg.size(), tt_cxy_pair(device_id, virtual_core), go_addr);
            uint32_t zero = 0;
            cluster_->write_core(
                &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
            cluster_->write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), go_message_index_addr);
        } else {
            // Multicast to all tensix cores in the range [start_core, tensix_end_core]
            cluster_->noc_multicast_write(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                device_id,
                start_core,
                end_core.value(),
                launch_addr);
            cluster_->noc_multicast_write(
                go_msg.data(), go_msg.size(), device_id, start_core, end_core.value(), go_addr);
            uint32_t zero = 0;
            cluster_->noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), launch_msg_buffer_read_ptr_addr);
            cluster_->noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), go_message_index_addr);
        }
    };

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [_, num_build_states] = BuildEnvManager::get_instance().get_build_index_and_state_count(
                    core_type_idx, processor_class, true);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance()
                                       .get_firmware_build_state(device_id, core_type_idx, processor_class, riscv_id)
                                       .get_target_out_path("");
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    hal_->set_iram_text_size(
                        launch_msg, core_type, static_cast<HalProcessorClassType>(processor_class), riscv_id, fw_size);
                    log_debug(
                        tt::LogMetal,
                        "RISC {} DM{} fw {} binary size: {} in bytes",
                        start_core.str(),
                        riscv_id,
                        fw_path,
                        fw_size);

                    if (not rtoptions_.get_skip_loading_fw()) {
                        llrt::test_load_multicast_write_risc_binary(
                            binary_mem,
                            device_id,
                            start_core,
                            end_core.value(),
                            core_type_idx,
                            processor_class,
                            riscv_id);
                    }
                }
            }

            if (!rtoptions_.get_fast_dispatch()) {
                // Host always writes launch messages
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
            } else {
                // Worker cores - Dispatcher will write launch messages
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_DEV;
            }
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (rtoptions_.get_fast_dispatch() and
                dispatch_core_manager_->get_dispatch_core_type() == CoreType::WORKER) {
                // Prepare a new launch message, with updated dispatch mode for dispatch cores
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
                prepare_initial_launch_msg();
                for (const auto& logical_core : dispatch_core_manager_->get_all_logical_dispatch_cores(device_id)) {
                    auto virtual_dispatch_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::WORKER);
                    auto programmable_core_type = llrt::get_core_type(device_id, virtual_dispatch_core);
                    cluster_->write_core(
                        init_launch_msg_data.data(),
                        init_launch_msg_data.size(),
                        tt_cxy_pair(device_id, virtual_dispatch_core),
                        hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
                }
            }

            cluster_->noc_multicast_write(
                &jit_build_config.fw_launch_addr_value,
                sizeof(uint32_t),
                device_id,
                start_core,
                end_core.value(),
                jit_build_config.fw_launch_addr);

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            if (!has_flag(MetalContext::instance().get_fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
                break;
            }
            const bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            const bool is_active_eth = !is_idle_eth;
            tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX;
            if (is_active_eth) {
                // On active eth, don't assert ERISC0, which is running base firmware.
                reset_val &= ~tt::umd::RiscType::ERISC0;
            }
            if (is_idle_eth or !hal_->get_eth_fw_is_cooperative()) {
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
            if (not rtoptions_.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal_->get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path =
                            BuildEnvManager::get_instance()
                                .get_firmware_build_state(device_id, core_type_idx, processor_class, eriscv_id)
                                .get_target_out_path("");
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        [[maybe_unused]] uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(
                            tt::LogMetal,
                            "{} ERISC {} DM{} fw {} binary size: {} in bytes",
                            is_active_eth ? "Active" : "Idle",
                            virtual_core.str(),
                            eriscv_id,
                            fw_path,
                            fw_size);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg.kernel_config().mode() = (!rtoptions_.get_fast_dispatch() or is_idle_eth)
                                                    ? dev_msgs::DISPATCH_MODE_HOST
                                                    : dev_msgs::DISPATCH_MODE_DEV;
            // For eth, write the go and launch message before initializing because when using the ETH FW API
            // it will launch immediately. DM0 is not in a reset state as it is running base FW.
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (core_type == HalProgrammableCoreType::ACTIVE_ETH) {
                // Clear the ncrisc_halt message
                DeviceAddr mailbox_addr = hal_->get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
                auto factory = hal_->get_dev_msgs_factory(core_type);
                DeviceAddr ncrisc_halt_addr =
                    mailbox_addr + factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::ncrisc_halt);
                std::vector<uint8_t> data(factory.size_of<dev_msgs::ncrisc_halt_msg_t>(), 0);
                cluster_->write_core(data.data(), data.size(), tt_cxy_pair(device_id, virtual_core), ncrisc_halt_addr);
            }

            // Write firmware main to primary erisc (DM0)
            // Using classic ASSERT/DEASSERT PC method for 1 erisc mode because erisc1 has no base firmware
            if (hal_->get_eth_fw_is_cooperative() || core_type != HalProgrammableCoreType::ACTIVE_ETH ||
                !rtoptions_.get_enable_2_erisc_mode()) {
                // PC
                cluster_->write_core(
                    &jit_build_config.fw_launch_addr_value,
                    sizeof(uint32_t),
                    tt_cxy_pair(device_id, virtual_core),
                    jit_build_config.fw_launch_addr);
            } else {
                // Active ethernet firmware launched immediately. Set the enable flag to 1 so FW doesn't exit
                // immediately.
                // Wait for ack not required because we wait for the done message
                constexpr uint32_t mailbox_index = 0;
                tt::llrt::internal_::send_msg_to_eth_mailbox(
                    device_id,
                    virtual_core,
                    tt_metal::FWMailboxMsg::ETH_MSG_RELEASE_CORE,
                    mailbox_index,
                    {/*l1 addr to exec*/ jit_build_config.fw_launch_addr_value},
                    false);
            }

            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", enchantum::to_string(core_type));
    }
}

dev_msgs::core_info_msg_t MetalContext::populate_core_info_msg(
    ChipId device_id, HalProgrammableCoreType programmable_core_type) const {
    const metal_SocDescriptor& soc_d = cluster_->get_soc_desc(device_id);
    // Use architecture-defined supported PCIe address bounds
    auto factory = hal_->get_dev_msgs_factory(programmable_core_type);
    dev_msgs::core_info_msg_t buffer = factory.create<dev_msgs::core_info_msg_t>();
    auto core_info = buffer.view();
    core_info.noc_pcie_addr_base() = hal_->get_pcie_addr_lower_bound();
    core_info.noc_pcie_addr_end() = hal_->get_pcie_addr_upper_bound();
    core_info.noc_dram_addr_base() = 0;
    core_info.noc_dram_addr_end() = soc_d.dram_core_size;
    core_info.l1_unreserved_start() = align(worker_l1_unreserved_start_, hal_->get_alignment(HalMemType::DRAM));
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::WORKER;
    } else if (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::ACTIVE_ETH;
    } else {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::IDLE_ETH;
    }
    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    // There are multiple NoC endpoints for DRAM, but not all are exposed through the API. Watcher will flag endpoints
    // that are not exposed as invalid transactions. This helps to avoid BH issue highlighted by SYS-592 where writing
    // to multiple DRAM endpoints can hang the card.
    std::unordered_set<tt::umd::CoreCoord> dram_cores;
    auto num_dram_channels = cluster_->get_soc_desc(device_id).get_num_dram_views();
    for (uint32_t dram_channel = 0; dram_channel < num_dram_channels; dram_channel++) {
        for (uint32_t noc = 0; noc < hal_->get_num_nocs(); noc++) {
            auto worker_dram_ep = soc_d.get_preferred_worker_core_for_dram_view(dram_channel, noc);
            auto eth_dram_ep = soc_d.get_preferred_eth_core_for_dram_view(dram_channel, noc);
            auto physical_worker_dram_ep =
                soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            auto physical_eth_dram_ep =
                soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            dram_cores.insert(physical_worker_dram_ep);
            dram_cores.insert(physical_eth_dram_ep);
        }
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores =
        soc_d.get_cores(CoreType::ETH, CoordSystem::NOC0);  // make these translated and then convert to physical

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= core_info.non_worker_cores().size(),
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= core_info.virtual_non_worker_cores().size(),
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    auto set_addressable_core =
        [](dev_msgs::addressable_core_t::View core, const CoreCoord& core_coord, dev_msgs::AddressableCoreType type) {
            core.x() = core_coord.x;
            core.y() = core_coord.y;
            core.type() = type;
        };
    for (auto non_worker_core : core_info.non_worker_cores()) {
        set_addressable_core(
            non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    for (auto virtual_non_worker_core : core_info.virtual_non_worker_cores()) {
        set_addressable_core(
            virtual_non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    // On Blackhole, virtualized Tensix coordinates overlap with NoC1 physical DRAM and PCIe coordinates beause
    // virtualized Tensix coordinates == NoC0 Tensix physical coordinates. This causes false negative Watcher
    // sanitization errors because it appears as a mixed use of physical and virtual To workaround this, skip over
    // populating `non_worker_cores` for BH DRAM when virtualization is enabled
    int non_worker_cores_idx = 0;
    bool skip_physical = cluster_->arch() == ARCH::BLACKHOLE and hal_->is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::PCIE);
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::DRAM);
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::ETH);
        }
    }

    if (hal_->is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            set_addressable_core(
                core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                virtual_core,
                dev_msgs::AddressableCoreType::ETH);
        }

        if (cluster_->arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::PCIE);
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::DRAM);
            }
        }
    }

    // Determine which noc-coords are harvested
    std::vector<uint32_t> harvested_axis_coord;
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_coords = umd::CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        cluster_->get_soc_desc(device_id).arch, cluster_->get_harvesting_mask(device_id));
    uint32_t max_along_axis =
        hal_->get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW ? soc_d.grid_size.y : soc_d.grid_size.x;
    for (uint32_t idx = 0; idx < max_along_axis; idx++) {
        bool harvested_axis = (harvested_noc_coords >> idx) & 0x1;
        if (harvested_axis) {
            harvested_axis_coord.push_back(idx);
        }
    }
    TT_ASSERT(
        harvested_axis_coord.size() <= core_info.harvested_coords().size(),
        "Detected more harvested rows than fit in mailbox.");
    for (size_t idx = 0; idx < core_info.harvested_coords().size(); idx++) {
        core_info.harvested_coords()[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : dev_msgs::CORE_COORD_INVALID;
        // Populate harvested rows/cols in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows/cols in the virtual space are placed at the end of the worker grid,
        if (hal_->is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            // On BH virtual coordinates are not contiguous
            uint32_t end_virtual_grid;
            if (hal_->get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW) {
                end_virtual_grid = hal_->get_virtual_worker_start_y() + logical_grid_size.y;
            } else if (cluster_->arch() == ARCH::BLACKHOLE) {
                end_virtual_grid = max_along_axis - 1;
            } else {
                end_virtual_grid = hal_->get_virtual_worker_start_x() + logical_grid_size.x;
            }

            // BH translated tensix cores are same as noc0 physical
            core_info.virtual_harvested_coords()[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info.virtual_harvested_coords()[idx] = dev_msgs::CORE_COORD_INVALID;
        }
    }

    core_info.noc_size_x() = soc_d.grid_size.x;
    core_info.noc_size_y() = soc_d.grid_size.y;
    core_info.worker_grid_size_x() = logical_grid_size.x;  // Grid size as virtual coords see it (workers only)
    core_info.worker_grid_size_y() = logical_grid_size.y;

    return buffer;
}

void MetalContext::initialize_and_launch_firmware(ChipId device_id) {
    ZoneScoped;

    // Download to worker cores
    log_debug(tt::LogMetal, "Initializing worker cores");
    std::unordered_set<CoreCoord> not_done_cores;
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);

    auto dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::TENSIX);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            // Setup the absolute logical coordinates of this worker which are relative to true origin. not the sub
            // device. When running the user kernel, which potentially is on a sub device, send that info using the
            // launch message using dispatch.
            core_info.view().absolute_logical_x() = logical_core.x;
            core_info.view().absolute_logical_y() = logical_core.y;
            // Must write to core before starting it
            cluster_->write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), worker_core},
                hal_->get_dev_addr(llrt::get_core_type(device_id, worker_core), HalL1MemAddrType::CORE_INFO));
            not_done_cores.insert(worker_core);
        }
    }
    CoreCoord start_core =
        cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord(0, 0), CoreType::WORKER);
    CoreCoord end_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
        device_id, CoreCoord(logical_grid_size.x - 1, logical_grid_size.y - 1), CoreType::WORKER);
    initialize_firmware(
        device_id, HalProgrammableCoreType::TENSIX, start_core, launch_msg.view(), go_msg.view(), end_core);

    // Clear erisc sync info
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal_->get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

        cluster_->write_core_immediate(
            device_id,
            virtual_core,
            zero_vec_erisc_init,
            hal_->get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
    log_debug(tt::LogMetal, "Initializing active ethernet cores");
    dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::ACTIVE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::ACTIVE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    std::unordered_set<CoreCoord> multi_risc_active_eth_cores;
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_->write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_->get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        if (!hal_->get_eth_fw_is_cooperative()) {
            multi_risc_active_eth_cores.insert(virtual_core);
            not_done_cores.insert(virtual_core);
        }
    }

    log_debug(tt::LogMetal, "Initializing idle ethernet cores");
    dev_msgs_factory = hal_->get_dev_msgs_factory(HalProgrammableCoreType::IDLE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::IDLE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;
    for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_->write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_->get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        not_done_cores.insert(virtual_core);
    }

    // Barrier between L1 writes above and deassert below
    cluster_->l1_barrier(device_id);

    // Deassert worker cores
    for (const auto& worker_core : not_done_cores) {
        if (multi_risc_active_eth_cores.contains(worker_core) && rtoptions_.get_enable_2_erisc_mode()) {
            // Not needed for 2 erisc mode. primary erisc handles deasserting subordinate
            continue;
        }

        tt::umd::RiscType reset_val;
        if (cluster_->arch() == ARCH::QUASAR) {
            reset_val = tt::umd::RiscType::ALL_NEO_DMS;
        } else {
            reset_val = tt::umd::RiscType::BRISC;
            if (multi_risc_active_eth_cores.contains(worker_core)) {
                // bit 12 needs to be deasserted to run second erisc on BH
                reset_val |= tt::umd::RiscType::ERISC1;
            }
        }
        cluster_->deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug(LogDevice, "Waiting for firmware init complete");
    const int timeout_ms = 10000;  // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error& e) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", device_id);
    }
    log_debug(LogDevice, "Firmware init complete");
}

// Command queue id stack for thread
thread_local MetalContext::CommandQueueIdStack MetalContext::command_queue_id_stack_for_thread_;

MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() {
    return MetalContext::command_queue_id_stack_for_thread_;
}
const MetalContext::CommandQueueIdStack& MetalContext::get_command_queue_id_stack_for_thread() const {
    return MetalContext::command_queue_id_stack_for_thread_;
}

uint32_t MetalContext::get_active_erisc_launch_flag_addr() {
    auto core_type_idx = hal_->get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    std::uint32_t launch_erisc_addr = hal_->get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
    return launch_erisc_addr;
};

bool MetalContext::erisc_app_still_running(ChipId device_id, CoreCoord virtual_core) {
    // Check if the kernel/erisc_app is still running on a ethernet core with context switching enabled
    // The LAUNCH_ERISC_APP_FLAG is reset to 0 after reset/reboot, and set to 1 when Metal runtime launches erisc
    // app FW Only applicable to WORMHOLE ethernet cores today, but could in theory extend to other cores, remove
    // assert if so
    if (cluster_->arch() != ARCH::WORMHOLE_B0) {
        return false;
    }
    TT_ASSERT(
        cluster_->is_ethernet_core(virtual_core, device_id),
        "Invalid core {} for context switch check",
        virtual_core.str());
    std::uint32_t launch_erisc_addr = get_active_erisc_launch_flag_addr();
    auto data = cluster_->read_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
    return (data[0] != 0);
};

// Send exit_erisc_kernel to the launch message
void MetalContext::erisc_send_exit_signal(ChipId device_id, CoreCoord virtual_core, bool is_idle_eth) {
    HalProgrammableCoreType programmable_core_type =
        is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    auto dev_msgs_factory = hal_->get_dev_msgs_factory(programmable_core_type);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    DeviceAddr launch_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);

    cluster_->read_core(
        launch_msg.data(), launch_msg.size(), {static_cast<size_t>(device_id), virtual_core}, launch_addr);

    launch_msg.view().kernel_config().exit_erisc_kernel() = 1;
    llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg.view(), go_msg.view(), false);

    if (!is_idle_eth) {
        // Active
        std::vector<uint32_t> clear_flag_data = {0};
        cluster_->write_core_immediate(device_id, virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
    }
};

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

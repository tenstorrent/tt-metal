// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_context.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/inspector.hpp"
#include "tt_metal/impl/debug/inspector_impl.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/debug/debug_helpers.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <filesystem>
#include <tt-metalium/device_pool.hpp>

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

// Check for environment variable override for custom mesh graph descriptor path
const char* get_custom_mesh_graph_desc_path() {
    const char* custom_mesh_graph_desc_path = std::getenv("TT_MESH_GRAPH_DESC_PATH");
    return custom_mesh_graph_desc_path;
}

}  // namespace

void MetalContext::reinitialize() {
    force_reinit_ = true;
    initialize(dispatch_core_config_, num_hw_cqs_, l1_bank_remap_, worker_l1_size_, false);
}

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config,
    uint8_t num_hw_cqs,
    const BankMapping& l1_bank_remap,
    size_t worker_l1_size,
    bool minimal) {
    // Workaround for galaxy and BH, need to always re-init
    if (rtoptions_.get_force_context_reinit() or cluster_->is_galaxy_cluster() or cluster_->arch() == ARCH::BLACKHOLE) {
        force_reinit_ = true;
    }
    // Settings that affect FW build can also trigger a re-initialization
    auto fw_compile_hash = std::hash<std::string>{}(rtoptions_.get_compile_hash_string());
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
                log_warning(
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

    // Initialize dispatch state
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    // Need DispatchMemMap for both dispatch core types
    tt_metal::DispatchSettings::initialize(*cluster_);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs);
    // Initialize debug servers. Attaching individual devices done below
    if (rtoptions_.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        TT_FATAL(!rtoptions_.get_profiler_enabled(), "Both DPRINT and Profiler cannot be enabled at the same time.");
        rtoptions_.set_disable_dma_ops(true);  // DMA is not thread-safe
        dprint_server_ = std::make_unique<DPrintServer>(rtoptions_);
    }
    watcher_server_ =
        std::make_unique<WatcherServer>();  // Watcher server always created, since we use it to register kernels

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // Clear state, build FW
    auto all_devices = cluster_->all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        // Clear L1/DRAM if requested
        if (rtoptions_.get_clear_l1()) {
            clear_l1_state(device_id);
        }
        if (rtoptions_.get_clear_dram()) {
            clear_dram_state(device_id);
        }
        int ai_clk = cluster_->get_device_aiclk(device_id);
        log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
        generate_device_bank_to_noc_tables(device_id);

        // Create build env for this device, and build FW if it's not built already
        BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
        uint32_t fw_build_key = BuildEnvManager::get_instance().get_device_build_env(device_id).build_key;
        if (!firmware_built_keys_.contains(fw_build_key)) {
            BuildEnvManager::get_instance().build_firmware(device_id);
            firmware_built_keys_.insert(fw_build_key);
        }

        // Clear the entire launch message ring buffer on ethernet cores before application firmware is activated.
        // This is required since ethernet cores context switch between application and routing firmware.
        // If ERISC application firmware is activated before the launch messages are cleared, it can enter an undefined
        // state by reading a corrupted launch message. Routing firmware will never run in this case, causing UMD issued
        // transactions to hang.
        clear_launch_messages_on_eth_cores(device_id);
    }

    // Populate FD topology across all devices
    if (rtoptions_.get_fast_dispatch()) {
        std::set<chip_id_t> all_devices_set(all_devices.begin(), all_devices.end());
        // TODO: enable this when dispatch init/teardown moves to MetalContext
        // populate_fd_kernels(all_devices_set, num_hw_cqs);
    }

    // Set internal routing for active ethernet cores, this is required for our FW to run
    cluster_->set_internal_routing_info_for_ethernet_cores(true);

    // Initialize debug tools, reset cores, init FW
    if (dprint_server_) {
        dprint_server_->attach_devices();
    }
    watcher_server_->init_devices();
    for (chip_id_t device_id : all_devices) {
        ClearNocData(device_id);

        // TODO: as optimization, investigate removing all this call for already initialized devivces
        if (!rtoptions_.get_skip_reset_cores_on_init()) {
            reset_cores(device_id);
        }

        initialize_and_launch_firmware(device_id);
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

void MetalContext::teardown() {
    if (!initialized_) {
        return;
    }
    initialized_ = false;

    // Set internal routing to false to exit active ethernet FW & go back to base FW
    cluster_->set_internal_routing_info_for_ethernet_cores(false);

    if (dprint_server_) {
        dprint_server_->detach_devices();
        dprint_server_.reset();
        rtoptions_.set_disable_dma_ops(false);
    }

    auto all_devices = cluster_->all_chip_ids();
    watcher_server_->detach_devices();
    watcher_server_.reset();
    for (chip_id_t device_id : all_devices) {
        assert_cores(device_id);

        cluster_->l1_barrier(device_id);
    }

    for (auto& mem_map : dispatch_mem_map_) {
        if (mem_map) {
            mem_map.reset();
        }
    }
    dispatch_query_manager_.reset();
    dispatch_core_manager_.reset();
    tt::tt_metal::reset_topology_state();
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

MetalContext::MetalContext() {
    bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    hal_ = std::make_unique<Hal>(get_platform_architecture(rtoptions_), is_base_routing_fw_enabled);
    cluster_ = std::make_unique<Cluster>(rtoptions_, *hal_);
    distributed_context_ = distributed::multihost::DistributedContext::get_current_world();
}

distributed::multihost::DistributedContext& MetalContext::get_distributed_context() {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return *distributed_context_;
}

MetalContext::~MetalContext() {
    cluster_.reset();
    hal_.reset();
}

llrt::RunTimeOptions& MetalContext::rtoptions() { return rtoptions_; }

Cluster& MetalContext::get_cluster() {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const { return rtoptions_; }

const Cluster& MetalContext::get_cluster() const {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const Hal& MetalContext::hal() const {
    TT_FATAL(hal_, "Trying to get hal before intializing it.");
    return *hal_;
}

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before intializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before intializing it.");
    return *dispatch_query_manager_;
}

const DispatchMemMap& MetalContext::dispatch_mem_map() const {
    return dispatch_mem_map(dispatch_core_config_.get_core_type());
}

const DispatchMemMap& MetalContext::dispatch_mem_map(const CoreType& core_type) const {
    auto& mem_map = dispatch_mem_map_[magic_enum::enum_integer(core_type)];
    TT_FATAL(mem_map, "Tried to get dispatch_mem_map for {} before intializing it.", core_type);
    return *mem_map;
}

void MetalContext::clear_l1_state(chip_id_t device_id) {
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
            tt::llrt::write_hex_vec_to_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    // Clear erisc unreserved L1
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        static uint32_t zero_vec_size = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        llrt::write_hex_vec_to_core(device_id, virtual_core, zero_vec, zero_vec_addr);
    }
    // TODO: clear idle eriscs as well
    cluster_->l1_barrier(device_id);
}

void MetalContext::clear_dram_state(chip_id_t device_id) {
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

void MetalContext::clear_launch_messages_on_eth_cores(chip_id_t device_id) {
    launch_msg_t launch_msg;
    go_msg_t go_msg;
    go_msg.signal = RUN_MSG_INIT;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, launch_msg);

    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core) {
        CoreCoord virtual_eth_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        cluster_->write_core(
            init_launch_msg_data.data(),
            launch_msg_buffer_num_entries * sizeof(launch_msg_t),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_->get_dev_addr(get_programmable_core_type(virtual_eth_core, device_id), HalL1MemAddrType::LAUNCH));
        uint32_t go_addr =
            hal_->get_dev_addr(get_programmable_core_type(virtual_eth_core, device_id), HalL1MemAddrType::GO_MSG);
        cluster_->write_core(&go_msg, sizeof(go_msg_t), tt_cxy_pair(device_id, virtual_eth_core), go_addr);
    };

    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core);
    }
    for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core);
    }
}

tt::tt_fabric::ControlPlane& MetalContext::get_control_plane() {
    if (!global_control_plane_) {
        this->initialize_control_plane();
    }
    return global_control_plane_->get_local_node_control_plane();
}

void MetalContext::set_custom_control_plane_mesh_graph(
    const std::string& mesh_graph_desc_file,
    const std::map<tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    TT_FATAL(
        !DevicePool::is_initialized() || DevicePool::instance().get_all_active_devices().size() == 0,
        "Modifying control plane requires no devices to be active");

    global_control_plane_ = std::make_unique<tt::tt_fabric::GlobalControlPlane>(
        mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalContext::set_default_control_plane_mesh_graph() {
    TT_FATAL(
        !DevicePool::is_initialized() || DevicePool::instance().get_all_active_devices().size() == 0,
        "Modifying control plane requires no devices to be active");
    global_control_plane_.reset();
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
    std::optional<uint8_t> num_routing_planes) {
    // Changes to fabric force a re-init. TODO: We should supply the fabric config in the same way as the dispatch config, not through this function exposed in the detail API.
    force_reinit_ = true;
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED || fabric_config == tt_fabric::FabricConfig::DISABLED) {
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
}

void MetalContext::initialize_fabric_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->cluster_->configure_ethernet_cores_for_fabric_routers(
        this->fabric_config_, this->num_fabric_active_routing_planes_);
    auto& control_plane = this->get_control_plane();
    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        control_plane.initialize_fabric_context(this->fabric_config_);
    }
    control_plane.configure_routing_tables_for_fabric_ethernet_channels(
        this->fabric_config_, this->fabric_reliability_mode_);
}

tt_fabric::FabricConfig MetalContext::get_fabric_config() const {
    return fabric_config_;
}

void MetalContext::initialize_control_plane() {
    if (auto* custom_mesh_graph_desc_path = get_custom_mesh_graph_desc_path(); custom_mesh_graph_desc_path != nullptr) {
        std::filesystem::path mesh_graph_desc_path = std::filesystem::path(custom_mesh_graph_desc_path);
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Custom mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());

        log_info(tt::LogDistributed, "Using custom mesh graph descriptor: {}", mesh_graph_desc_path.string());
        global_control_plane_ = std::make_unique<tt::tt_fabric::GlobalControlPlane>(
            mesh_graph_desc_path.string());
        return;
    }

    // Default mode, auto select mesh graph descriptor. In future, we can add a way for user to specify custom
    // descriptors
    std::string mesh_graph_descriptor;
    auto cluster_type = cluster_->get_cluster_type();
    switch (cluster_type) {
        case tt::ClusterType::N150: mesh_graph_descriptor = "n150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::N300: mesh_graph_descriptor = "n300_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::T3K: mesh_graph_descriptor = "t3k_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::GALAXY:
            if (tt::tt_fabric::get_fabric_type(this->fabric_config_, cluster_type) ==
                tt::tt_fabric::FabricType::TORUS_XY) {
                mesh_graph_descriptor = "single_galaxy_torus_xy_graph_descriptor.yaml";
            } else {
                mesh_graph_descriptor = "single_galaxy_mesh_graph_descriptor.yaml";
            }
            break;
        case tt::ClusterType::TG: mesh_graph_descriptor = "tg_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P100: mesh_graph_descriptor = "p100_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150: mesh_graph_descriptor = "p150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150_X2: mesh_graph_descriptor = "p150_x2_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::P150_X4: mesh_graph_descriptor = "p150_x4_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::SIMULATOR_WORMHOLE_B0: mesh_graph_descriptor = "n150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::SIMULATOR_BLACKHOLE: mesh_graph_descriptor = "p150_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::N300_2x2: mesh_graph_descriptor = "n300_2x2_mesh_graph_descriptor.yaml"; break;
        case tt::ClusterType::INVALID: TT_THROW("Unknown cluster type");
    }
    const std::filesystem::path mesh_graph_desc_path = std::filesystem::path(rtoptions_.get_root_dir()) /
                                                       "tt_metal/fabric/mesh_graph_descriptors" / mesh_graph_descriptor;

    global_control_plane_ = std::make_unique<tt::tt_fabric::GlobalControlPlane>(
        mesh_graph_desc_path.string());
}

void MetalContext::reset_cores(chip_id_t device_id) {
    ZoneScoped;

    auto get_active_erisc_launch_flag_addr = [&]() {
        auto core_type_idx = hal_->get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
        std::uint32_t launch_erisc_addr = hal_->get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
        return launch_erisc_addr;
    };

    auto erisc_app_still_running = [&](CoreCoord virtual_core) {
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
        auto data = tt::llrt::read_hex_vec_from_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
        return (data[0] != 0);
    };

    // Send exit_erisc_kernel to the launch message
    auto erisc_send_exit_signal = [&](CoreCoord virtual_core, bool is_idle_eth) {
        go_msg_t go_msg;
        std::memset(&go_msg, 0, sizeof(go_msg_t));
        log_info(
            tt::LogMetal,
            "While initializing device {}, {} ethernet dispatch core {} detected as still "
            "running, issuing exit signal.",
            device_id,
            is_idle_eth ? "idle" : "active",
            virtual_core.str());

        DeviceAddr launch_addr = hal_->get_dev_addr(
            is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH,
            HalL1MemAddrType::LAUNCH);

        std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
        data = tt::llrt::read_hex_vec_from_core(device_id, virtual_core, launch_addr, sizeof(launch_msg_t));

        launch_msg_t* launch_msg = (launch_msg_t*)(&data[0]);
        launch_msg->kernel_config.exit_erisc_kernel = 1;
        llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg, &go_msg, launch_addr, false);

        if (!is_idle_eth) {
            // Active
            std::vector<uint32_t> clear_flag_data = {0};
            tt::llrt::write_hex_vec_to_core(
                device_id, virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
        }
    };

    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    // Active ethernet
    if (hal_->get_eth_fw_is_cooperative()) {
        for (const auto& logical_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
            if (erisc_app_still_running(virtual_core)) {
                erisc_send_exit_signal(virtual_core, false /* is_idle_eth */);
                device_to_early_exit_cores[device_id].insert(virtual_core);
            }
        }
    }

    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto& id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000;  // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(id_and_cores.first, RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error& e) {
                log_warning(
                    tt::LogAlways,
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW intialization...");
            }
        }
    }

    // Reset Tensix cores, ignore storage only cores
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    const auto& storage_only_cores = tt::get_logical_storage_cores(device_id, num_hw_cqs_, dispatch_core_config_);
    auto storage_only_cores_set = std::unordered_set<CoreCoord>(storage_only_cores.begin(), storage_only_cores.end());
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            if (!storage_only_cores_set.contains(logical_core)) {
                cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core));
            }
        }
    }

    // Reset idle ethernet cores
    // TODO: reset BH eth cores as well
    for (const auto& logical_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core));
    }
}

void MetalContext::assert_cores(chip_id_t device_id) {
    auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(device_id);
    auto routing_cores = tt::tt_metal::get_virtual_dispatch_routing_cores(device_id);

    // Assert riscs on Tensix, minus storage cores
    CoreCoord grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    const auto& storage_only_cores = tt::get_logical_storage_cores(device_id, num_hw_cqs_, dispatch_core_config_);
    auto storage_only_cores_set = std::unordered_set<CoreCoord>(storage_only_cores.begin(), storage_only_cores.end());
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);

            if (!dispatch_cores.contains(worker_core) && !routing_cores.contains(worker_core)) {
                if (!storage_only_cores_set.contains(logical_core)) {
                    cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), device_id);
            }
        }
    }

    if (!hal_->get_eth_fw_is_cooperative()) {
        // Assert riscs on active eth
        for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_eth_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
            TensixSoftResetOptions reset_val =
                TENSIX_ASSERT_SOFT_RESET &
                static_cast<TensixSoftResetOptions>(
                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            cluster_->assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_eth_core), reset_val);
        }
    }
}

CoreCoord MetalContext::virtual_noc0_coordinate(chip_id_t device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = cluster_->get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y || cluster_->arch() == ARCH::BLACKHOLE) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, coord);
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            hal_->noc_coordinate(noc_index, grid_size.x, coord.x),
            hal_->noc_coordinate(noc_index, grid_size.y, coord.y)};
        return virtual_coord;
    }
}

void MetalContext::generate_device_bank_to_noc_tables(chip_id_t device_id) {
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
    bool dram_is_virtualized =
        hal_->get_virtualized_core_types().find(AddressableCoreType::DRAM) != hal_->get_virtualized_core_types().end();
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
        for (unsigned int bank_id = 0; bank_id < l1_noc_coord_per_bank.size(); bank_id++) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, l1_noc_coord_per_bank[bank_id]);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal_->get_noc_addr_node_id_bits()) | noc_x) << hal_->get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }
}

void MetalContext::initialize_device_bank_to_noc_tables(
    chip_id_t device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core) {
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

    cluster_->write_core(
        &dram_bank_to_noc_xy_[device_id][0],
        dram_to_noc_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        mem_bank_to_noc_addr);
    uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
    cluster_->write_core(
        &l1_bank_to_noc_xy_[device_id][0], l1_to_noc_sz_in_bytes, tt_cxy_pair(device_id, virtual_core), l1_noc_addr);

    uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
    cluster_->write_core(
        &dram_bank_offset_map_[device_id][0],
        dram_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        dram_offset_addr);
    uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
    cluster_->write_core(
        &l1_bank_offset_map_[device_id][0],
        l1_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        l1_offset_addr);
}

void MetalContext::initialize_firmware(
    chip_id_t device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    launch_msg_t* launch_msg,
    go_msg_t* go_msg) {
    ZoneScoped;

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core);
    uint32_t core_type_idx = hal_->get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal_->get_processor_classes_count(core_type);
    auto jit_build_config =
        hal_->get_jit_build_config(core_type_idx, 0, 0);  // Only the first risc needs to be programmed

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [build_idx, num_build_states] =
                    BuildEnvManager::get_instance().get_build_index_and_state_count(core_type_idx, processor_class);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance()
                                       .get_firmware_build_state(device_id, core_type_idx, processor_class, riscv_id)
                                       .get_target_out_path("");
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    if (riscv_id + build_idx == 1) {  // TODO: clean up how brisc/ncrisc are handled
                        // In this context, ncrisc_kernel_size16 is the size of the fw
                        launch_msg->kernel_config.ncrisc_kernel_size16 = (fw_size + 15) >> 4;
                    }
                    log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, fw_size);

                    if (not rtoptions_.get_skip_loading_fw()) {
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, riscv_id);
                    }
                }
            }

            if (!rtoptions_.get_fast_dispatch()) {
                // Host always writes launch messages
                launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
            } else {
                std::unordered_set<CoreCoord> virtual_dispatch_cores;
                if (dispatch_core_manager_->get_dispatch_core_type() == CoreType::WORKER) {
                    for (const auto& logical_core : dispatch_core_manager_->get_all_logical_dispatch_cores(device_id)) {
                        virtual_dispatch_cores.insert(cluster_->get_virtual_coordinate_from_logical_coordinates(
                            device_id, logical_core, CoreType::WORKER));
                    }
                }
                if (virtual_dispatch_cores.contains(virtual_core)) {
                    // Dispatch cores - Host writes launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
                } else {
                    // Worker cores - Dispatcher will write launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_DEV;
                }
            }

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
            if (not is_idle_eth) {
                reset_val =
                    reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
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
                        uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(LogDevice, "ERISC fw binary size: {} in bytes", fw_size);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg->kernel_config.mode =
                (!rtoptions_.get_fast_dispatch() or is_idle_eth) ? DISPATCH_MODE_HOST : DISPATCH_MODE_DEV;
            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", magic_enum::enum_name(core_type));
    }

    cluster_->write_core(
        &jit_build_config.fw_launch_addr_value,
        sizeof(uint32_t),
        tt_cxy_pair(device_id, virtual_core),
        jit_build_config.fw_launch_addr);

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
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, *launch_msg);
    auto programmable_core_type = get_programmable_core_type(virtual_core, device_id);
    cluster_->write_core(
        init_launch_msg_data.data(),
        launch_msg_buffer_num_entries * sizeof(launch_msg_t),
        tt_cxy_pair(device_id, virtual_core),
        hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
    uint32_t go_addr = hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
    cluster_->write_core(go_msg, sizeof(go_msg_t), tt_cxy_pair(device_id, virtual_core), go_addr);
    uint64_t launch_msg_buffer_read_ptr_addr =
        hal_->get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    uint32_t zero = 0;
    cluster_->write_core(
        &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
}

void MetalContext::initialize_and_launch_firmware(chip_id_t device_id) {
    ZoneScoped;

    launch_msg_t launch_msg;
    go_msg_t go_msg;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    go_msg.signal = RUN_MSG_INIT;

    // Populate core info, which will be written to device
    std::vector<uint32_t> core_info_vec(sizeof(core_info_msg_t) / sizeof(uint32_t));
    core_info_msg_t* core_info = (core_info_msg_t*)core_info_vec.data();

    const metal_SocDescriptor& soc_d = cluster_->get_soc_desc(device_id);
    uint64_t pcie_chan_base_addr = cluster_->get_pcie_base_addr_from_device(device_id);
    uint32_t num_host_channels = cluster_->get_num_host_channels(device_id);
    uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
    for (int pcie_chan = 0; pcie_chan < num_host_channels; pcie_chan++) {
        pcie_chan_end_addr += cluster_->get_host_channel_size(device_id, pcie_chan);
    }
    core_info->noc_pcie_addr_base = pcie_chan_base_addr;
    core_info->noc_pcie_addr_end = pcie_chan_end_addr;
    core_info->noc_dram_addr_base = 0;
    core_info->noc_dram_addr_end = soc_d.dram_core_size;
    core_info->l1_unreserved_start = align(worker_l1_unreserved_start_, hal_->get_alignment(HalMemType::DRAM));

    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, soc_d.get_umd_coord_system());
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
                soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
            auto physical_eth_dram_ep =
                soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
            dram_cores.insert(physical_worker_dram_ep);
            dram_cores.insert(physical_eth_dram_ep);
        }
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores =
        soc_d.get_cores(CoreType::ETH, CoordSystem::PHYSICAL);  // make these translated and then convert to physical

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= MAX_PHYSICAL_NON_WORKER_CORES,
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= MAX_VIRTUAL_NON_WORKER_CORES,
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    for (int idx = 0; idx < MAX_PHYSICAL_NON_WORKER_CORES; idx++) {
        core_info->non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }
    for (int idx = 0; idx < MAX_VIRTUAL_NON_WORKER_CORES; idx++) {
        core_info->virtual_non_worker_cores[idx] = {
            CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }

    // On Blackhole, virtualized Tensix coordinates overlap with NoC1 physical DRAM and PCIe coordinates beause
    // virtualized Tensix coordinates == NoC0 Tensix physical coordinates. This causes false negative Watcher
    // sanitization errors because it appears as a mixed use of physical and virtual To workaround this, skip over
    // populating `non_worker_cores` for BH DRAM when virtualization is enabled
    int non_worker_cores_idx = 0;
    bool skip_physical = cluster_->arch() == ARCH::BLACKHOLE and hal_->is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::PCIE};
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::DRAM};
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::ETH};
        }
    }

    if (hal_->is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                virtual_core.x, virtual_core.y, AddressableCoreType::ETH};
        }

        if (cluster_->arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                    virtual_core.x, virtual_core.y, AddressableCoreType::PCIE};
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core =
                    cluster_->get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                    virtual_core.x, virtual_core.y, AddressableCoreType::DRAM};
            }
        }
    }

    // Determine which noc-coords are harvested
    std::vector<uint32_t> harvested_axis_coord;
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_coords = CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
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
        harvested_axis_coord.size() <= MAX_HARVESTED_ON_AXIS, "Detected more harvested rows than fit in mailbox.");
    for (int idx = 0; idx < MAX_HARVESTED_ON_AXIS; idx++) {
        core_info->harvested_coords[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : CORE_COORD_INVALID;
        // Populate harvested rows/cols in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows/cols in the virtual space are placed at the end of the worker grid,
        if (hal_->is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            // On BH virtual coordinates are not contiguous
            uint32_t end_virtual_grid = hal_->get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW
                                            ? hal_->get_virtual_worker_start_y() + logical_grid_size.y
                                        : (cluster_->arch() == ARCH::BLACKHOLE)
                                            ? max_along_axis - 1
                                            : hal_->get_virtual_worker_start_x() + logical_grid_size.x;

            // BH translated tensix cores are same as noc0 physical
            core_info->virtual_harvested_coords[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info->virtual_harvested_coords[idx] = CORE_COORD_INVALID;
        }
    }

    core_info->noc_size_x = soc_d.grid_size.x;
    core_info->noc_size_y = soc_d.grid_size.y;
    core_info->worker_grid_size_x = logical_grid_size.x;  // Grid size as virtual coords see it (workers only)
    core_info->worker_grid_size_y = logical_grid_size.y;

    // Download to worker cores
    log_debug(LogDevice, "Initializing firmware");
    std::unordered_set<CoreCoord> not_done_cores;

    const auto& storage_only_cores = tt::get_logical_storage_cores(device_id, num_hw_cqs_, dispatch_core_config_);
    auto storage_only_cores_set = std::unordered_set<CoreCoord>(storage_only_cores.begin(), storage_only_cores.end());
    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!storage_only_cores_set.count(logical_core)) {
                CoreCoord worker_core = cluster_->get_virtual_coordinate_from_logical_coordinates(
                    device_id, logical_core, CoreType::WORKER);
                // Setup the absolute logical coordinates of this worker which are relative to true origin. not the sub
                // device. When running the user kernel, which potentially is on a sub device, send that info using the
                // launch message using dispatch.
                core_info->absolute_logical_x = logical_core.x;
                core_info->absolute_logical_y = logical_core.y;
                // Must write to core before starting it
                tt::llrt::write_hex_vec_to_core(
                    device_id,
                    worker_core,
                    core_info_vec,
                    hal_->get_dev_addr(
                        get_programmable_core_type(worker_core, device_id), HalL1MemAddrType::CORE_INFO));
                initialize_firmware(device_id, HalProgrammableCoreType::TENSIX, worker_core, &launch_msg, &go_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Clear erisc sync info
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal_->get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

        llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            zero_vec_erisc_init,
            hal_->get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
    std::unordered_set<CoreCoord> active_eth_cores;
    for (const auto& eth_core : this->get_control_plane().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            core_info_vec,
            hal_->get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, &launch_msg, &go_msg);
        if (!hal_->get_eth_fw_is_cooperative()) {
            active_eth_cores.insert(virtual_core);
            not_done_cores.insert(virtual_core);
        }
    }

    for (const auto& eth_core : this->get_control_plane().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            core_info_vec,
            hal_->get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, &launch_msg, &go_msg);
        not_done_cores.insert(virtual_core);
    }

    // Barrier between L1 writes above and deassert below
    cluster_->l1_barrier(device_id);

    // Deassert worker cores
    TensixSoftResetOptions reset_val;
    for (const auto& worker_core : not_done_cores) {
        if (active_eth_cores.find(worker_core) != active_eth_cores.end()) {
            // bit 12 needs to be deasserted to run second erisc on BH
            reset_val = TENSIX_DEASSERT_SOFT_RESET &
                        static_cast<TensixSoftResetOptions>(
                            ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        } else {
            reset_val = TENSIX_DEASSERT_SOFT_RESET;
        }
        cluster_->deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug(LogDevice, "Waiting for firmware init complete");
    const int timeout_ms = 10000;  // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error& e) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", device_id);
    }
    log_debug(LogDevice, "Firmware init complete");
}

}  // namespace tt::tt_metal

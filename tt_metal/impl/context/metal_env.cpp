
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>
#include <algorithm>
#include <filesystem>
#include <enchantum/enchantum.hpp>
#include <tt_stl/fmt.hpp>
#include <limits>
#include <unordered_set>
#include "metal_env_impl.hpp"
#include "firmware_capability.hpp"
#include "get_platform_architecture.hpp"
#include "profiler_state_manager.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <utility>

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/mesh_graph.hpp>
#include <distributed_context.hpp>
#include <system_mesh.hpp>
#include "fabric/fabric_host_utils.hpp"
#include "fabric/channel_trimming_export.hpp"

namespace tt::tt_metal {

// ─── MetalEnvDescriptor ───────────────────────────────────────────────────────

std::mutex MetalEnvImpl::s_registry_mutex_;
std::set<MetalEnvImpl*> MetalEnvImpl::s_registry_;
std::once_flag MetalEnvImpl::s_atfork_registered_;

MetalEnvDescriptor::MetalEnvDescriptor(const std::string& mock_cluster_desc_path) :
    mock_cluster_desc_path_(
        mock_cluster_desc_path.empty() ? std::nullopt : std::optional<std::string>(mock_cluster_desc_path)) {}
MetalEnvDescriptor::MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path) :
    mock_cluster_desc_path_(std::move(mock_cluster_desc_path)) {}
MetalEnvDescriptor::MetalEnvDescriptor(
    std::optional<std::string> mock_cluster_desc_path, FabricConfigDescriptor fabric_config_desc) :
    mock_cluster_desc_path_(std::move(mock_cluster_desc_path)), fabric_config_desc_(fabric_config_desc) {}

// ─── MetalEnvImpl core ───────────────────────────────────────────────────────

void MetalEnvImpl::prefork_check_all() {
    std::lock_guard<std::mutex> lock(s_registry_mutex_);
    for (auto* impl : s_registry_) {
        impl->check_use_count_zero();
    }
}

MetalEnvImpl::MetalEnvImpl(MetalEnvDescriptor descriptor) : descriptor_(std::move(descriptor)) {
    initialize_base_objects();
    verify_fw_capabilities();

    // Apply fabric config from descriptor
    const auto& fc = descriptor_.fabric_config_descriptor();
    fabric_config_ = fc.fabric_config;
    fabric_reliability_mode_ = fc.reliability_mode;
    fabric_tensix_config_ = fc.fabric_tensix_config;
    fabric_udm_mode_ = fc.fabric_udm_mode;
    fabric_manager_ = fc.fabric_manager;
    fabric_router_config_ = fc.router_config;
    if (fc.num_routing_planes.has_value()) {
        num_fabric_active_routing_planes_ = fc.num_routing_planes.value();
    }

    // Pick up any custom mesh graph descriptor from env/rtoptions
    if (rtoptions_->is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_->get_custom_fabric_mesh_graph_desc_path();
    }

    // Initialize distributed context
    distributed_context_ = distributed::multihost::DistributedContext::get_current_world();

    std::call_once(s_atfork_registered_, []() { pthread_atfork(prefork_check_all, nullptr, nullptr); });

    std::lock_guard<std::mutex> lock(s_registry_mutex_);
    s_registry_.insert(this);
}

MetalEnvImpl::~MetalEnvImpl() {
    {
        std::lock_guard<std::mutex> lock(s_registry_mutex_);
        s_registry_.erase(this);
    }
    check_use_count_zero();
    teardown_fabric_objects();
    cluster_.reset();
    hal_.reset();
    rtoptions_.reset();
}

void MetalEnvImpl::acquire() { use_count_.fetch_add(1, std::memory_order_acq_rel); }
void MetalEnvImpl::release() { use_count_.fetch_sub(1, std::memory_order_acq_rel); }

bool MetalEnvImpl::check_use_count_zero() const {
    const int use_count = use_count_.load(std::memory_order_acquire);
    if (use_count > 0) {
        log_error(
            tt::LogMetal,
            "MetalEnv has {} outstanding reference(s) at teardown or fork. All objects using the MetalEnv (e.g. open "
            "devices) must stop using it before the MetalEnv is destroyed or the process forks.",
            use_count);
        return false;
    }
    return true;
}

llrt::RunTimeOptions& MetalEnvImpl::get_rtoptions() { return *rtoptions_; }
const Hal& MetalEnvImpl::get_hal() { return *hal_; }
Cluster& MetalEnvImpl::get_cluster() { return *cluster_; }
const MetalEnvDescriptor& MetalEnvImpl::get_descriptor() const { return descriptor_; }

void MetalEnvImpl::initialize_base_objects() {
    this->rtoptions_ = std::make_unique<llrt::RunTimeOptions>();

    if (descriptor_.is_mock_device()) {
        this->rtoptions_->set_mock_cluster_desc(std::string(descriptor_.mock_cluster_desc_path()));
    }

    const bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(*this->rtoptions_));
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);

    cluster_ = std::make_unique<Cluster>(*this->rtoptions_);
    this->verify_fw_capabilities();
    this->hal_ = std::make_unique<Hal>(
        platform_arch,
        is_base_routing_fw_enabled,
        this->rtoptions_->get_enable_2_erisc_mode(),
        get_profiler_dram_bank_size_for_hal_allocation(*this->rtoptions_));

    this->rtoptions_->ParseAllFeatureEnv(*hal_);
    this->cluster_->set_hal(hal_.get());
}

void MetalEnvImpl::verify_fw_capabilities() {
    FirmwareCapabilityRequest req;
    req.enable_2_erisc_mode = this->rtoptions_->get_enable_2_erisc_mode();

    FirmwareCapabilityResult res;
    const auto platform_arch = get_platform_architecture(*this->rtoptions_);
    if (!check_firmware_capabilities(platform_arch, {.eth_fw = cluster_->get_ethernet_firmware_version()}, req, res)) {
        this->rtoptions_->set_enable_2_erisc_mode(res.enable_2_erisc_mode);
    }
}

// ─── Fabric config ────────────────────────────────────────────────────────────

tt_fabric::FabricConfig MetalEnvImpl::get_fabric_config() const { return fabric_config_; }

tt_fabric::FabricReliabilityMode MetalEnvImpl::get_fabric_reliability_mode() const { return fabric_reliability_mode_; }

const tt_fabric::FabricRouterConfig& MetalEnvImpl::get_fabric_router_config() const { return fabric_router_config_; }

tt_fabric::FabricTensixConfig MetalEnvImpl::get_fabric_tensix_config() const { return fabric_tensix_config_; }

tt_fabric::FabricUDMMode MetalEnvImpl::get_fabric_udm_mode() const { return fabric_udm_mode_; }

tt_fabric::FabricManagerMode MetalEnvImpl::get_fabric_manager() const { return fabric_manager_; }

uint8_t MetalEnvImpl::get_num_fabric_active_routing_planes() const { return num_fabric_active_routing_planes_; }

void MetalEnvImpl::set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config) {
    fabric_tensix_config_ = fabric_tensix_config;
}

// The fabric config is normally set once, from the FabricConfigDescriptor supplied at MetalEnv construction time.
// However, for the legacy backward-compatibility path, the DeviceManager may call set_fabric_config a second time
// to enable minimal fabric (FABRIC_1D) for dispatch when the user has not explicitly configured fabric.
// See DeviceManager::initialize for that path.
bool MetalEnvImpl::set_fabric_config(
    tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    tt_fabric::FabricTensixConfig fabric_tensix_config,
    tt_fabric::FabricUDMMode fabric_udm_mode,
    tt_fabric::FabricManagerMode fabric_manager,
    tt_fabric::FabricRouterConfig router_config) {
    force_reinit_ = true;

    // Export channel trimming capture data before fabric config changes.
    // Must happen while fabric_config_ is still active and fabric context is alive.
    bool is_tearing_down_fabric =
        fabric_config == tt_fabric::FabricConfig::DISABLED && this->fabric_config_ != tt_fabric::FabricConfig::DISABLED;
    if (is_tearing_down_fabric) {
        tt::tt_fabric::export_channel_trimming_capture(*this);
    }

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED ||
        fabric_config == tt_fabric::FabricConfig::DISABLED) {
        this->fabric_config_ = fabric_config;
        this->fabric_reliability_mode_ = reliability_mode;
    } else {
        TT_FATAL(
            this->fabric_config_ == fabric_config,
            "Tried to override previous value of fabric config: {}, with: {}",
            enchantum::to_string(this->fabric_config_),
            enchantum::to_string(fabric_config));
    }

    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        if (num_routing_planes.has_value()) {
            log_warning(
                tt::LogMetal,
                "Got num_routing_planes while disabling fabric, ignoring it and disabling all active routing planes");
        }

        this->teardown_fabric_config();
        return false;
    }

    if (num_routing_planes.has_value() && num_routing_planes.value() < this->num_fabric_active_routing_planes_) {
        log_warning(
            tt::LogMetal,
            "Got num_routing_planes: {}, which is less than current value: {}, ignoring the override",
            num_routing_planes.value(),
            this->num_fabric_active_routing_planes_);
        return false;
    }

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

    this->set_fabric_tensix_config(fabric_tensix_config);
    this->fabric_udm_mode_ = fabric_udm_mode;
    this->fabric_manager_ = fabric_manager;
    this->fabric_router_config_ = router_config;

    if (control_plane_ != nullptr) {
        log_info(
            tt::LogMetal,
            "Fabric config changed from {} to {}, reinitializing control plane",
            this->get_control_plane().get_fabric_config(),
            this->fabric_config_);
        system_mesh_.reset();
        this->initialize_control_plane_impl();
    }

    return true;
}

void MetalEnvImpl::teardown_fabric_config() {
    this->fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    this->get_cluster().configure_ethernet_cores_for_fabric_routers(this->fabric_config_);
    this->num_fabric_active_routing_planes_ = 0;
    this->get_control_plane().clear_fabric_context();
}

void MetalEnvImpl::initialize_fabric_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->get_cluster().configure_ethernet_cores_for_fabric_routers(
        this->fabric_config_, this->num_fabric_active_routing_planes_);
    auto& cp = this->get_control_plane();
    cp.configure_routing_tables_for_fabric_ethernet_channels();
}

void MetalEnvImpl::initialize_fabric_tensix_datamover_config() {
    if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    if (get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }

    if (tt::tt_fabric::is_tt_fabric_config(this->fabric_config_)) {
        auto& cp = this->get_control_plane();
        cp.initialize_fabric_tensix_datamover_config();
    }
}

bool MetalEnvImpl::consume_force_reinit() {
    bool val = force_reinit_;
    force_reinit_ = false;
    return val;
}

// ─── Control plane ────────────────────────────────────────────────────────────

tt::tt_fabric::ControlPlane& MetalEnvImpl::get_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!control_plane_) {
        this->initialize_control_plane_impl();
    }
    return *control_plane_;
}

void MetalEnvImpl::initialize_control_plane() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    initialize_control_plane_impl();
}

void MetalEnvImpl::initialize_control_plane_impl() {
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
    log_info(tt::LogDistributed, "Using auto discovery to generate mesh graph.");

    if (*distributed_context_->size() == 1) {
        this->construct_control_plane();
    } else {
        const auto cluster_type = get_cluster().get_cluster_type();
        auto fabric_type = tt::tt_fabric::get_fabric_type(this->fabric_config_, get_cluster().is_ubb_galaxy());
        std::filesystem::path mesh_graph_desc_path =
            tt::tt_fabric::MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
                cluster_type, rtoptions_->get_root_dir(), fabric_type);

        log_debug(tt::LogMetal, "Using mesh graph descriptor: {}", mesh_graph_desc_path.string());

        TT_FATAL(!mesh_graph_desc_path.empty(), "No mesh graph descriptor found for cluster type");
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Mesh graph descriptor file not found: {}",
            mesh_graph_desc_path.string());
        this->construct_control_plane(mesh_graph_desc_path);
    }
}

void MetalEnvImpl::construct_control_plane(const std::filesystem::path& mesh_graph_desc_path) {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_info(tt::LogDistributed, "Using custom Fabric Node Id to physical chip mapping.");
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
            get_cluster(),
            *rtoptions_,
            get_hal(),
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
            get_cluster(),
            *rtoptions_,
            get_hal(),
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

void MetalEnvImpl::construct_control_plane() {
    if (!logical_mesh_chip_id_to_physical_chip_id_mapping_.empty()) {
        log_warning(
            tt::LogDistributed,
            "Custom Fabric Node Id to physical chip mapping provided but no mesh graph descriptor path. "
            "Mapping will be ignored. Please provide a custom mesh graph descriptor path for custom logical to "
            "physical mapping.");
    }
    log_info(tt::LogDistributed, "Constructing control plane using auto-discovery (no mesh graph descriptor).");
    control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(
        get_cluster(),
        *rtoptions_,
        get_hal(),
        *distributed_context_,
        this->fabric_config_,
        this->fabric_reliability_mode_,
        this->fabric_tensix_config_,
        this->fabric_udm_mode_,
        this->fabric_router_config_,
        this->fabric_manager_);
}

// ─── System mesh ──────────────────────────────────────────────────────────────

distributed::SystemMesh& MetalEnvImpl::get_system_mesh() {
    std::lock_guard<std::mutex> lock(control_plane_mutex_);
    if (!system_mesh_) {
        if (!control_plane_) {
            this->initialize_control_plane_impl();
        }
        system_mesh_ = std::unique_ptr<distributed::SystemMesh>(new distributed::SystemMesh(*control_plane_));
    }
    return *system_mesh_;
}

// ─── Custom topology ──────────────────────────────────────────────────────────

void MetalEnvImpl::set_custom_fabric_topology(
    const std::string& mesh_graph_desc_file,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    custom_mesh_graph_desc_path_ = mesh_graph_desc_file;
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

void MetalEnvImpl::set_default_fabric_topology() {
    system_mesh_.reset();
    control_plane_.reset();
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.clear();

    if (rtoptions_->is_custom_fabric_mesh_graph_desc_path_specified()) {
        custom_mesh_graph_desc_path_ = rtoptions_->get_custom_fabric_mesh_graph_desc_path();
    } else {
        custom_mesh_graph_desc_path_ = std::nullopt;
    }
    this->set_fabric_config(fabric_config_, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

// ─── Distributed context ──────────────────────────────────────────────────────

namespace {
std::shared_ptr<distributed::multihost::DistributedContext> construct_compute_only_distributed_context(
    tt::tt_fabric::ControlPlane& control_plane) {
    const auto& global_context = distributed::multihost::DistributedContext::get_current_world();
    if (*global_context->size() == 1) {
        return global_context;
    }

    const auto& mesh_graph = control_plane.get_mesh_graph();

    if (mesh_graph.get_switch_ids().empty()) {
        return global_context;
    }

    const auto& compute_mesh_ids = mesh_graph.get_mesh_ids();
    const auto& global_logical_bindings = control_plane.get_global_logical_bindings();

    std::unordered_set<int> compute_mpi_ranks;
    for (const auto& [rank, mesh_binding] : global_logical_bindings) {
        const auto& [mesh_id, _] = mesh_binding;
        if (std::find(compute_mesh_ids.begin(), compute_mesh_ids.end(), mesh_id) != compute_mesh_ids.end()) {
            compute_mpi_ranks.insert(rank.get());
        }
    }

    if (compute_mpi_ranks.empty()) {
        TT_THROW("No compute meshes found in mesh graph.");
    }

    std::vector<int> compute_ranks_vec(compute_mpi_ranks.begin(), compute_mpi_ranks.end());
    std::sort(compute_ranks_vec.begin(), compute_ranks_vec.end());

    int current_rank = *global_context->rank();
    bool is_current_rank_in_compute =
        std::find(compute_ranks_vec.begin(), compute_ranks_vec.end(), current_rank) != compute_ranks_vec.end();

    if (!is_current_rank_in_compute) {
        return control_plane.get_host_local_context();
    }

    return global_context->create_sub_context(compute_ranks_vec);
}
}  // namespace

const distributed::multihost::DistributedContext& MetalEnvImpl::full_world_distributed_context() const {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return *distributed_context_;
}

const distributed::multihost::DistributedContext& MetalEnvImpl::global_distributed_context() {
    if (!control_plane_) {
        return *distributed_context_;
    }
    if (!compute_only_distributed_context_) {
        compute_only_distributed_context_ = construct_compute_only_distributed_context(get_control_plane());
    }
    return *compute_only_distributed_context_;
}

std::shared_ptr<distributed::multihost::DistributedContext> MetalEnvImpl::get_distributed_context_ptr() {
    TT_FATAL(distributed_context_, "Distributed context not initialized.");
    return distributed_context_;
}

void MetalEnvImpl::teardown_fabric_objects() {
    system_mesh_.reset();
    control_plane_.reset();
    distributed_context_.reset();
    compute_only_distributed_context_.reset();
}

// ─── MetalEnv public forwarding ───────────────────────────────────────────────

MetalEnv::MetalEnv(MetalEnvDescriptor descriptor) : impl_(std::make_unique<MetalEnvImpl>(std::move(descriptor))) {}

MetalEnv::~MetalEnv() { this->impl_.reset(); }

const MetalEnvDescriptor& MetalEnv::get_descriptor() const { return impl_->get_descriptor(); }

tt::ARCH MetalEnv::get_arch() const { return impl_->get_cluster().arch(); }
std::string MetalEnv::get_arch_name() const { return tt::get_string_lowercase(get_arch()); }
uint32_t MetalEnv::get_num_pcie_devices() const { return impl_->get_cluster().number_of_pci_devices(); }
uint32_t MetalEnv::get_l1_size() const {
    return impl_->get_hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
}
uint32_t MetalEnv::get_dram_alignment() const { return impl_->get_hal().get_alignment(HalMemType::DRAM); }
uint32_t MetalEnv::get_l1_alignment() const { return impl_->get_hal().get_alignment(HalMemType::L1); }
uint32_t MetalEnv::get_arch_num_circular_buffers() const { return impl_->get_hal().get_arch_num_circular_buffers(); }
uint32_t MetalEnv::get_max_worker_l1_unreserved_size() const {
    size_t l1_end = impl_->get_hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                    impl_->get_hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return l1_end - impl_->get_hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
}
float MetalEnv::get_eps() const { return impl_->get_hal().get_eps(); }
float MetalEnv::get_nan() const { return impl_->get_hal().get_nan(); }
float MetalEnv::get_inf() const { return impl_->get_hal().get_inf(); }

tt::tt_fabric::ControlPlane& MetalEnv::get_control_plane() { return impl_->get_control_plane(); }
distributed::SystemMesh& MetalEnv::get_system_mesh() { return impl_->get_system_mesh(); }

}  // namespace tt::tt_metal

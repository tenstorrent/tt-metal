// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <mutex>
#include <set>
#include <atomic>
#include <filesystem>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal::distributed {
class SystemMesh;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt::tt_metal {

class MetalEnvImpl {
public:
    explicit MetalEnvImpl(MetalEnvDescriptor descriptor);
    ~MetalEnvImpl();

    llrt::RunTimeOptions& get_rtoptions();
    const Hal& get_hal();
    Cluster& get_cluster();
    distributed::SystemMesh& get_system_mesh();
    const MetalEnvDescriptor& get_descriptor() const;

    bool check_use_count_zero() const;

    void acquire();
    void release();

    // --- Fabric config ---
    tt_fabric::FabricConfig get_fabric_config() const;
    tt_fabric::FabricReliabilityMode get_fabric_reliability_mode() const;
    const tt_fabric::FabricRouterConfig& get_fabric_router_config() const;
    tt_fabric::FabricTensixConfig get_fabric_tensix_config() const;
    tt_fabric::FabricUDMMode get_fabric_udm_mode() const;
    tt_fabric::FabricManagerMode get_fabric_manager() const;
    uint8_t get_num_fabric_active_routing_planes() const;

    // Returns true if updated
    bool set_fabric_config(
        tt_fabric::FabricConfig fabric_config,
        tt_fabric::FabricReliabilityMode reliability_mode =
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::optional<uint8_t> num_routing_planes = std::nullopt,
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED,
        tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED,
        tt_fabric::FabricManagerMode fabric_manager = tt_fabric::FabricManagerMode::DEFAULT,
        tt_fabric::FabricRouterConfig router_config = tt_fabric::FabricRouterConfig{});
    void set_fabric_tensix_config(tt_fabric::FabricTensixConfig fabric_tensix_config);
    void initialize_fabric_config();
    void initialize_fabric_tensix_datamover_config();
    void teardown_fabric_config();

    // --- Control plane ---
    tt::tt_fabric::ControlPlane& get_control_plane();
    void initialize_control_plane();

    // --- Custom topology ---
    // Need to call set_fabric_config to reinit the control plane after calling this
    // TODO: Remove these two functions in favor of a unified set fabric config where you can
    // pass in a custom topology.
    void set_custom_fabric_topology(
        const std::string& mesh_graph_desc_file,
        const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    void set_default_fabric_topology();

    // --- Distributed context ---
    const distributed::multihost::DistributedContext& full_world_distributed_context() const;
    const distributed::multihost::DistributedContext& global_distributed_context();
    std::shared_ptr<distributed::multihost::DistributedContext> get_distributed_context_ptr();

    // Teardown fabric-layer objects (control plane, system mesh, distributed context).
    void teardown_fabric_objects();

    // Register this env as a MetalContext (silicon: default slot; mock: non-default slot) so that legacy
    // fabric code reached during control-plane / system-mesh / mesh-device construction (which still calls
    // the bare MetalContext::instance()) resolves to this env instead of implicitly creating a default
    // silicon context that opens a second real PCIe device and self-deadlocks on its CHIP_IN_USE lock
    // (GitHub #50041, #50043). Idempotent; returns the env-owned context id (created on first call). The
    // context lifetime is tied to the env (uplift of mesh-device-level cleanup, GitHub #21500).
    int ensure_context_registered(MetalEnv& env);

    // True once ensure_context_registered has created the env-owned context.
    bool has_registered_context() const { return registered_context_id_.has_value(); }

    // Destroy the env-owned context created by ensure_context_registered, if any. Must be called from
    // MetalEnv::~MetalEnv() while MetalEnv::impl_ is still valid (MetalContext::teardown() reaches back
    // through *env_ -> impl_), not from ~MetalEnvImpl where impl_ has already been nulled by reset().
    void teardown_registered_context();

    // Returns true if set_fabric_config changed state requiring a reinit.
    bool consume_force_reinit();

private:
    // During init we need to bypass set_fabric_config to avoid reinitialization of the control plane
    // by directly setting the fabric config state.
    friend class DeviceManager;

    MetalEnvDescriptor descriptor_;

    std::unique_ptr<llrt::RunTimeOptions> rtoptions_;
    std::unique_ptr<Cluster> cluster_;
    std::unique_ptr<Hal> hal_;

    std::atomic<int> use_count_{0};

    // MetalContext id registered lazily for this env (see ensure_context_registered). Stored as a raw int
    // to avoid pulling metal_context.hpp into this header; wrapped back into a ContextId at the
    // metal_context.cpp boundary. Torn down by teardown_registered_context() from MetalEnv::~MetalEnv().
    std::optional<int> registered_context_id_ = std::nullopt;

    // --- Fabric config state ---
    tt_fabric::FabricConfig fabric_config_ = tt_fabric::FabricConfig::DISABLED;
    tt_fabric::FabricReliabilityMode fabric_reliability_mode_ =
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    tt_fabric::FabricTensixConfig fabric_tensix_config_ = tt_fabric::FabricTensixConfig::DISABLED;
    tt_fabric::FabricUDMMode fabric_udm_mode_ = tt_fabric::FabricUDMMode::DISABLED;
    tt_fabric::FabricManagerMode fabric_manager_ = tt_fabric::FabricManagerMode::DEFAULT;
    tt_fabric::FabricRouterConfig fabric_router_config_ = tt_fabric::FabricRouterConfig{};
    uint8_t num_fabric_active_routing_planes_ = 0;
    std::map<tt_fabric::FabricNodeId, ChipId> logical_mesh_chip_id_to_physical_chip_id_mapping_;
    std::optional<std::string> custom_mesh_graph_desc_path_ = std::nullopt;

    bool force_reinit_ = false;

    // --- Control plane / system mesh ---
    std::mutex control_plane_mutex_;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;
    std::unique_ptr<distributed::SystemMesh> system_mesh_;

    // --- Distributed context ---
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context_;
    std::shared_ptr<distributed::multihost::DistributedContext> compute_only_distributed_context_;

    void initialize_base_objects();
    void verify_fw_capabilities();

    void construct_control_plane(const std::filesystem::path& mesh_graph_desc_path);
    void construct_control_plane();
    void initialize_control_plane_impl();

    static std::mutex s_registry_mutex_;
    static std::set<MetalEnvImpl*> s_registry_;
    static std::once_flag s_atfork_registered_;
    static void prefork_check_all();
};

}  // namespace tt::tt_metal

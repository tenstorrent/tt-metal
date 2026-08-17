// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/channel_trimming_report.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

namespace {

// Does any intermesh edge in this fabric sit on a Z direction?
//
// Only such an edge produces the Z-facing intermesh boundary shape, whose channel counts the
// fabric-wide maximum below has to cover. Deliberately local to that maximum: no router's shape is
// derived from it -- each is derived from its own facing and edge capability -- so it is a property
// of the fabric being sized, not a configuration fact worth carrying in IntermeshVCConfig.
bool fabric_has_intermesh_z_edge(const MeshGraph& mesh_graph) {
    for (const auto& mesh_connections : mesh_graph.get_inter_mesh_connectivity()) {
        for (const auto& chip_connections : mesh_connections) {
            for (const auto& [dst_mesh_id, router_edge] : chip_connections) {
                if (router_edge.port_direction == RoutingDirection::Z) {
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace

StreamAssignment FabricBuilderContext::compute_stream_assignment(MeshId mesh_id) const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const bool express_enabled = control_plane.express_routing_enabled(mesh_id);
    // The credit plan follows express enablement (per mesh) and multi-TXQ (device-wide, from the
    // shared router config) -- the same facts the per-router derivation used, lifted to the scope
    // they actually vary at.
    const auto& base_config = get_fabric_router_config();
    const bool multi_txq_enabled = base_config.sender_txq_id != base_config.receiver_txq_id;
    const CreditTransportPlan plan{
        .vc0_uses_counters = multi_txq_enabled, .vc1_uses_counters = multi_txq_enabled || express_enabled};

    std::array<uint32_t, builder_config::MAX_NUM_VCS> max_senders{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> max_receivers{};
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        max_senders[vc] = static_cast<uint32_t>(max_sender_channels_per_vc_[vc]);
        max_receivers[vc] = static_cast<uint32_t>(max_receiver_channels_per_vc_[vc]);
    }
    const StreamPlacementInputs placement{
        .max_sender_counts = max_senders,
        .max_receiver_counts = max_receivers,
        .vc2_present = intermesh_vc_config_.requires_vc2,
        .tensix_relay_present = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() ==
                                tt::tt_fabric::FabricTensixConfig::UDM};
    return make_stream_assignment(stream_requirements(placement, plan));
}

const StreamAssignment& FabricBuilderContext::get_stream_assignment(MeshId mesh_id) const {
    const auto it = stream_assignments_.find(mesh_id);
    TT_FATAL(
        it != stream_assignments_.end(),
        "No stream assignment for mesh M{}: the map is filled at construction for the meshes bound to this host, "
        "and every caller asks about its own router's mesh",
        *mesh_id);
    return it->second;
}

void FabricBuilderContext::compute_max_channel_counts() {
    // Derive the shape of every router family present in this fabric. These are archetype
    // queries -- "what shape would a router with these facts have" -- so no router or layout
    // object is constructed; the derivation is a function call per family.
    const auto topology = fabric_context_.get_fabric_topology();

    std::vector<RouterVcShape> possible_shapes;

    // An express mesh router's VC0 is five wide, so the fabric-wide maximum has to account for it or
    // the shared router config would report fewer sender channels than a router actually maps -- the
    // variant-to-router channel lookup would then index past its end. Asked across every local mesh
    // rather than per node, since this maximum is fabric-wide.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    bool any_mesh_uses_express = false;
    for (const auto mesh_id : control_plane.get_local_mesh_id_bindings()) {
        any_mesh_uses_express = any_mesh_uses_express || control_plane.express_routing_enabled(mesh_id);
    }

    // The families are named by the one chip fact that distinguishes them: what the Z port is for.
    // Every cardinal is an ordinary same-mesh edge in all of them, and the shape reads only this
    // router's own capability and the Z role off the set.
    const auto chip_with_z = [](std::optional<EdgeCapability> z_capability) {
        PerDirectionCapabilities caps;
        for (const auto direction :
             {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
            caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
        }
        caps.at(RoutingDirection::Z) = z_capability;
        return caps;
    };

    // Always have MESH routers
    bool intermesh_vc_config_active = intermesh_vc_config_.requires_vc1 || intermesh_vc_config_.requires_vc2;
    possible_shapes.push_back(router_vc_shape(
        topology,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        any_mesh_uses_express,
        intermesh_vc_config_active ? &intermesh_vc_config_ : nullptr));

    // If Z-facing intermesh boundary routers exist in this fabric, enumerate both families they
    // introduce: the boundary itself (5 VC0 / 4 VC1 by its wiring rules), and the mesh routers on
    // the same chips, whose VC1 gains the from-Z slot (4 VC0 / 4 VC1 in the non-express case).
    // The boundary family's 9 dominates the maximum today, but the enumeration is the contract:
    // every family present in the fabric is represented here.
    //
    // Gated on VC1 as well as on the edge: the boundary family's entire shape is its from-boundary
    // VC1 fanout, so without VC1 it cannot be constructed at all (router_vc_shape rejects it). That
    // also reproduces the condition under which the intermesh config was enabled in the first place.
    const bool has_intermesh_z_boundary =
        intermesh_vc_config_.requires_vc1 && fabric_has_intermesh_z_edge(control_plane.get_mesh_graph());
    if (has_intermesh_z_boundary) {
        const auto boundary_chip = chip_with_z(EdgeCapability::INTERMESH);
        possible_shapes.push_back(router_vc_shape(
            topology,
            RoutingDirection::Z,
            boundary_chip,
            any_mesh_uses_express,  // inert for the boundary family, but this is the fabric's state
            &intermesh_vc_config_));
        // Mesh routers on boundary chips carry the from-Z slot.
        possible_shapes.push_back(router_vc_shape(
            topology,
            RoutingDirection::N,
            boundary_chip,
            any_mesh_uses_express,
            intermesh_vc_config_active ? &intermesh_vc_config_ : nullptr));
    }

    // Compute max channel counts across all router families in this fabric
    max_sender_channels_per_vc_.fill(0);
    max_receiver_channels_per_vc_.fill(0);

    for (const auto& shape : possible_shapes) {
        for (uint32_t vc = 0; vc < shape.num_vcs; ++vc) {
            max_sender_channels_per_vc_[vc] =
                std::max(max_sender_channels_per_vc_[vc], static_cast<std::size_t>(shape.sender_counts[vc]));
            max_receiver_channels_per_vc_[vc] =
                std::max(max_receiver_channels_per_vc_[vc], static_cast<std::size_t>(shape.receiver_counts[vc]));
        }
    }
}

FabricBuilderContext::FabricBuilderContext(const FabricContext& fabric_context) : fabric_context_(fabric_context) {
    // Load channel trimming overrides from profile if specified
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    TT_FATAL(
        !(rtoptions.has_fabric_trimming_profile() && rtoptions.get_enable_channel_trimming_capture()),
        "TT_METAL_FABRIC_TRIMMING_PROFILE and TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE are mutually exclusive. "
        "Capture mode instruments routers to record usage; import mode applies a previously captured profile to "
        "optimize router construction. Enable only one at a time.");
    TT_FATAL(
        !(rtoptions.has_fabric_trimming_override() && rtoptions.get_enable_channel_trimming_capture()),
        "TT_METAL_FABRIC_TRIMMING_OVERRIDE and TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE are mutually exclusive. "
        "Capture mode instruments routers to record usage; override mode applies forced channel settings. "
        "Enable only one at a time.");
    if (rtoptions.has_fabric_trimming_profile()) {
        const auto& path = rtoptions.get_fabric_trimming_profile_path();
        log_info(tt::LogFabric, "Loading channel trimming profile: {}", path);
        channel_trimming_overrides_ = load_channel_trimming_overrides(path);
    }

    // Load global overrides from override file if specified
    if (rtoptions.has_fabric_trimming_override()) {
        const auto& override_path = rtoptions.get_fabric_trimming_override_path();
        log_info(tt::LogFabric, "Loading channel trimming global overrides: {}", override_path);
        channel_trimming_global_overrides_ = load_channel_trimming_global_overrides(override_path);
    }

    this->intermesh_vc_config_ = this->compute_intermesh_vc_config();

    // Log trimming report after intermesh config is known (VC1 affects expected channel counts)
    if (rtoptions.has_fabric_trimming_profile()) {
        const auto& path = rtoptions.get_fabric_trimming_profile_path();
        generate_and_log_channel_trimming_report(path, fabric_context.get_fabric_topology(), intermesh_vc_config_.requires_vc1);
    }

    // Compute max channel counts for this fabric instance
    compute_max_channel_counts();

    // Create configs using computed max
    router_config_ = create_edm_config();
    for (size_t direction = 0; direction < eth_chan_directions::COUNT; direction++) {
        router_with_mux_config_[direction] =
            create_edm_config(FabricTensixConfig::MUX, static_cast<eth_chan_directions>(direction));
    }

    tensix_config_ = nullptr;

    // Derive each local mesh's stream assignment now that its inputs exist (the family maxima above
    // and the router config just built). It is read during kernel creation, which runs one thread
    // per device against this shared context, so it has to be in place before that starts.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    for (const auto mesh_id : control_plane.get_local_mesh_id_bindings()) {
        stream_assignments_.emplace(mesh_id, compute_stream_assignment(mesh_id));
    }

    // Initialize per-device build state
    num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    auto num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
    if (num_devices_ != 4 && num_pcie_devices == 4) {
        num_devices_ += num_pcie_devices;
    }
    master_router_chans_.resize(num_devices_, UNINITIALIZED_MASTER_ROUTER_CHAN);
    num_initialized_routers_.resize(num_devices_, UNINITIALIZED_ROUTERS);
}

std::unique_ptr<FabricEriscDatamoverConfig> FabricBuilderContext::create_edm_config(
    FabricTensixConfig fabric_tensix_config, eth_chan_directions direction) const {
    auto edm_options = FabricEriscDatamoverOptions{
        .fabric_tensix_config = fabric_tensix_config,
        .direction = direction,
    };

    // MUX/UDM modes are mutually exclusive with VC2 — zero out VC2 channels
    auto sender_channels = max_sender_channels_per_vc_;
    auto receiver_channels = max_receiver_channels_per_vc_;
    if (fabric_tensix_config == FabricTensixConfig::MUX || fabric_tensix_config == FabricTensixConfig::UDM) {
        sender_channels[2] = 0;
        receiver_channels[2] = 0;
    }

    return std::make_unique<FabricEriscDatamoverConfig>(
        fabric_context_.get_fabric_channel_buffer_size_bytes(),
        fabric_context_.get_fabric_topology(),
        edm_options,
        sender_channels,
        receiver_channels);
}

FabricEriscDatamoverConfig& FabricBuilderContext::get_fabric_router_config(
    FabricTensixConfig fabric_tensix_config, eth_chan_directions direction) const {
    switch (fabric_tensix_config) {
        case FabricTensixConfig::DISABLED:
        case FabricTensixConfig::UDM:
            TT_FATAL(router_config_ != nullptr, "Error, fabric router config is uninitialized");
            return *router_config_;
        case FabricTensixConfig::MUX:
            TT_FATAL(
                router_with_mux_config_[direction] != nullptr,
                "Error, fabric router config with mux extension is uninitialized for direction {}",
                direction);
            return *router_with_mux_config_[direction].get();
        default: TT_FATAL(false, "Error, invalid fabric_tensix_config: {}", fabric_tensix_config);
    }
}

void FabricBuilderContext::set_num_fabric_initialized_routers(ChipId chip_id, size_t num_routers) {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        num_initialized_routers_[chip_id] == UNINITIALIZED_ROUTERS,
        "Error, tried to set num initialized routers again for device {}",
        chip_id);
    num_initialized_routers_[chip_id] = num_routers;
}

uint32_t FabricBuilderContext::get_num_fabric_initialized_routers(ChipId chip_id) const {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        num_initialized_routers_[chip_id] != UNINITIALIZED_ROUTERS,
        "Error, querying num initialized routers for an unknown device {}",
        chip_id);
    return num_initialized_routers_[chip_id];
}

void FabricBuilderContext::set_fabric_master_router_chan(ChipId chip_id, chan_id_t chan_id) {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        master_router_chans_[chip_id] == UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, tried to set master router channel again for the same device {}",
        chip_id);
    master_router_chans_[chip_id] = chan_id;
}

chan_id_t FabricBuilderContext::get_fabric_master_router_chan(ChipId chip_id) const {
    TT_FATAL(chip_id < num_devices_, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices_);
    TT_FATAL(
        master_router_chans_[chip_id] != UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, querying master router channel for an unknown device {}",
        chip_id);
    return master_router_chans_[chip_id];
}

std::vector<size_t> FabricBuilderContext::get_fabric_router_addresses_to_clear() const {
    std::vector<size_t> addresses_to_clear = {
        router_config_->edm_local_sync_address,
        router_config_->edm_local_tensix_sync_address,
        router_config_->termination_signal_address};

    if (router_config_->sender_txq_id != router_config_->receiver_txq_id) {
        addresses_to_clear.push_back(router_config_->to_sender_channel_remote_ack_counters_base_addr);
        addresses_to_clear.push_back(router_config_->to_sender_channel_remote_completion_counters_base_addr);
        addresses_to_clear.push_back(router_config_->receiver_channel_remote_ack_counters_base_addr);
        addresses_to_clear.push_back(router_config_->receiver_channel_remote_completion_counters_base_addr);
    }

    return addresses_to_clear;
}

std::pair<uint32_t, uint32_t> FabricBuilderContext::get_fabric_router_sync_address_and_status() const {
    return std::make_pair(router_config_->edm_status_address, EDMStatus::LOCAL_HANDSHAKE_COMPLETE);
}

std::optional<std::pair<uint32_t, EDMStatus>> FabricBuilderContext::get_fabric_router_ready_address_and_signal() const {
    return std::make_pair(router_config_->edm_status_address, EDMStatus::READY_FOR_TRAFFIC);
}

std::pair<uint32_t, uint32_t> FabricBuilderContext::get_fabric_router_termination_address_and_signal() const {
    return std::make_pair(router_config_->termination_signal_address, TerminationSignal::IMMEDIATELY_TERMINATE);
}

FabricTensixDatamoverConfig& FabricBuilderContext::get_tensix_config() const {
    TT_FATAL(tensix_config_ != nullptr, "Error, fabric tensix config is uninitialized");
    return *tensix_config_;
}

void FabricBuilderContext::initialize_tensix_config() {
    TT_FATAL(tensix_config_ == nullptr, "Trying to re-initialize fabric tensix config");

    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config != FabricTensixConfig::DISABLED) {
        // Now it's safe to call get_active_fabric_eth_channels() because
        // configure_routing_tables_for_fabric_ethernet_channels() has already run
        tensix_config_ = std::make_unique<FabricTensixDatamoverConfig>();
    }
}

IntermeshVCConfig FabricBuilderContext::compute_intermesh_vc_config() const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    auto config = IntermeshVCConfig::disabled();

    // Check if multiple meshes exist — needed for VC1 (intermesh traffic)
    const auto& mesh_ids = mesh_graph.get_mesh_ids();
    constexpr size_t single_mesh_count = 1;
    bool is_multi_mesh = mesh_ids.size() > single_mesh_count;

    if (is_multi_mesh) {
        // Check if intermesh connections exist (use inter_mesh_connectivity which has actual parsed connections)
        const auto& inter_mesh_connectivity = mesh_graph.get_inter_mesh_connectivity();

        // Count total intermesh connections across all meshes
        size_t total_intermesh_connections = 0;
        for (const auto& mesh_connections : inter_mesh_connectivity) {
            for (const auto& chip_connections : mesh_connections) {
                total_intermesh_connections += chip_connections.size();
            }
        }

        if (total_intermesh_connections > 0) {
            // Default to FULL_MESH when intermesh exists
            // TODO: Implement detection logic for:
            //   - EDGE_ONLY: Check if workload only needs edge nodes (optimization)
            //   - FULL_MESH_WITH_PASS_THROUGH: Auto-detect when a mesh forwards traffic between other meshes
            //
            // EXPERIMENTAL: pass-through (A->B->C inter-mesh routing) is currently opt-in via env var.
            // It reuses VC1 for both in-mesh delivery and cross-mesh pass-through and is NOT guaranteed
            // deadlock-free (a fully deadlock-free implementation requires a dedicated pass-through VC).
            const bool needs_mesh_pass_through =
                tt::tt_metal::MetalContext::instance().rtoptions().get_enable_fabric_mesh_pass_through();

            config = needs_mesh_pass_through ? IntermeshVCConfig::full_mesh_with_pass_through()
                                             : IntermeshVCConfig::full_mesh();
        }
    }

    // VC2 is independent of VC1 — only requires: RT option + Blackhole + no UDM/mux + 2D topology
    // (2D topology check happens in initialize_vc2_mappings, not here)
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_vc2()) {
        auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
        auto tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
        bool is_blackhole = (arch == tt::ARCH::BLACKHOLE);
        bool is_udm_mode = (tensix_config == FabricTensixConfig::UDM);
        bool is_mux_extension = (tensix_config == FabricTensixConfig::MUX);
        config.requires_vc2 = is_blackhole && !is_udm_mode && !is_mux_extension;
    }

    return config;
}


}  // namespace tt::tt_fabric

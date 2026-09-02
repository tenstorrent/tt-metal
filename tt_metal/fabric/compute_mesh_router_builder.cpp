// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_mesh_router_builder.hpp"
#include <enchantum/enchantum.hpp>
#include <cstdlib>
#include <limits>
#include <string>
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/builder/injection_policy.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/third_party/umd/device/api/umd/device/types/core_coordinates.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "tt_metal.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

namespace tt::tt_fabric {

namespace {

// Set bits [0, channels_per_vc[0]) | [offset1, offset1+channels_per_vc[1]) | ... in a bitfield.
void set_all_channel_bits(
    uint16_t& bitfield, const std::array<std::size_t, builder_config::MAX_NUM_VCS>& channels_per_vc) {
    size_t offset = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        bitfield |= static_cast<uint16_t>(((1u << channels_per_vc[vc]) - 1) << offset);
        offset += channels_per_vc[vc];
    }
}

// Replace a VC's channel bits in a bitfield using override settings.
// Clears all bits in [offset, offset+count), then sets only override-specified bits.
void apply_vc_override_to_bitfield(
    uint16_t& bitfield,
    size_t offset,
    size_t count,
    size_t vc,
    const std::optional<bool>& force_enable_all,
    const std::optional<std::vector<size_t>>& force_enable_indices,
    const char* direction_name) {
    uint16_t vc_mask = static_cast<uint16_t>(((1u << count) - 1) << offset);

    // CLEAR all bits for this VC range
    bitfield &= ~vc_mask;

    // SET only what the override says
    if (force_enable_all.value_or(false)) {
        bitfield |= vc_mask;
    } else if (force_enable_indices.has_value()) {
        for (size_t idx : *force_enable_indices) {
            TT_FATAL(
                idx < count,
                "Override {} channel index {} exceeds VC{} {} count {}",
                direction_name,
                idx,
                vc,
                direction_name,
                count);
            bitfield |= (1u << (offset + idx));
        }
    }
}

}  // namespace

void apply_global_overrides_to_entry(
    ChannelTrimmingOverrides& entry,
    const ChannelTrimmingGlobalOverrides& global_overrides,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc) {
    size_t sender_offset = 0;
    size_t receiver_offset = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        const auto& vc_override = global_overrides.per_vc[vc];

        if (vc_override.has_sender_override()) {
            apply_vc_override_to_bitfield(
                entry.sender_channel_used_bitfield_by_vc,
                sender_offset,
                sender_channels_per_vc[vc],
                vc,
                vc_override.force_enable_all_sender_channels,
                vc_override.force_enable_sender_channels,
                "sender");
        }
        sender_offset += sender_channels_per_vc[vc];

        if (vc_override.has_receiver_override()) {
            apply_vc_override_to_bitfield(
                entry.receiver_channel_data_forwarded_bitfield_by_vc,
                receiver_offset,
                receiver_channels_per_vc[vc],
                vc,
                vc_override.force_enable_all_receiver_channels,
                vc_override.force_enable_receiver_channels,
                "receiver");
        }
        receiver_offset += receiver_channels_per_vc[vc];
    }
}

namespace {

// Resolve the effective channel trimming overrides for a single router.
//
// Looks up the router in the capture map (if present), then conditionally applies
// global overrides. Returns nullopt if neither capture nor global override applies
// (meaning the builder should use its default: all channels enabled).
//
// Priority:
//   - No capture, no global override → nullopt (builder default)
//   - Capture only → capture entry (existing behavior)
//   - Global override only → fully-enabled baseline with override applied
//   - Capture + global override → capture entry with override applied
std::optional<ChannelTrimmingOverrides> resolve_channel_trimming_for_router(
    const std::optional<ChannelTrimmingOverrideMap>& capture_overrides,
    const ChannelTrimmingGlobalOverrides& global_overrides,
    ChipId chip_id,
    chan_id_t eth_chan,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc) {
    // Look up capture entry for this router
    std::optional<ChannelTrimmingOverrides> result;
    if (capture_overrides.has_value()) {
        auto key = make_override_key(chip_id, eth_chan);
        auto it = capture_overrides->find(key);
        if (it != capture_overrides->end()) {
            result = it->second;
        }
    }

    bool has_global_override = global_overrides.has_any_override();
    if (!result.has_value() && !has_global_override) {
        return std::nullopt;  // No trimming — builder default
    }

    if (!result.has_value()) {
        // No capture entry — construct a fully-enabled baseline for the override to modify
        ChannelTrimmingOverrides baseline;
        baseline.sender_channel_min_packet_size_seen_bytes_by_vc.fill(std::numeric_limits<uint16_t>::max());
        set_all_channel_bits(baseline.sender_channel_used_bitfield_by_vc, sender_channels_per_vc);
        set_all_channel_bits(baseline.receiver_channel_data_forwarded_bitfield_by_vc, receiver_channels_per_vc);
        result = baseline;
    }

    if (has_global_override) {
        apply_global_overrides_to_entry(*result, global_overrides, sender_channels_per_vc, receiver_channels_per_vc);
    }

    return result;
}

struct RouterChannelCounts {
    std::array<std::size_t, builder_config::MAX_NUM_VCS> sender = {};
    std::array<std::size_t, builder_config::MAX_NUM_VCS> receiver = {};
};

RouterChannelCounts compute_router_channel_counts(
    const FabricContext& fabric_context,
    const ControlPlane& control_plane,
    const FabricNodeId& fabric_node_id,
    RoutingDirection direction) {
    const auto topology = fabric_context.get_fabric_topology();
    // The channel shape follows the facing direction and the edge's capability: only a Z-facing
    // router whose edge crosses a mesh boundary gets the intermesh boundary shape. A same-mesh Z is
    // an express chord, which is an ordinary mesh-like forwarding direction, not a shape family.
    // discover_channels() already rejects more than one neighbor mesh per direction, so the
    // direction has exactly one neighbor to classify. This is the archetype query: what shape
    // WOULD a router with these facts have -- no router is constructed to find out.
    auto chip_capabilities = chip_capabilities_of(control_plane, fabric_node_id);
    // A direction with no discovered neighbor still has a shape question to answer here, so the
    // query reads it as an ordinary same-mesh edge rather than refusing.
    if (!chip_capabilities.at(direction).has_value()) {
        chip_capabilities.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    const auto& intermesh_config = fabric_context.get_builder_context().get_intermesh_vc_config();
    const auto vc_shape = router_vc_shape(
        topology,
        direction,
        chip_capabilities,
        control_plane.express_routing_enabled(fabric_node_id.mesh_id),
        &intermesh_config);

    RouterChannelCounts counts;
    for (uint32_t vc = 0; vc < vc_shape.num_vcs; ++vc) {
        counts.sender[vc] = vc_shape.sender_counts[vc];
        counts.receiver[vc] = vc_shape.receiver_counts[vc];
    }
    return counts;
}

std::optional<Vc0TrimFastPathInfo> resolve_vc0_trim_fast_path_info(
    const FabricBuilderContext& builder_context,
    ChipId chip_id,
    chan_id_t eth_chan,
    const RouterChannelCounts& channel_counts) {
    const auto& capture_overrides = builder_context.get_channel_trimming_overrides();
    if (!has_real_channel_trimming_capture_entry(capture_overrides, chip_id, eth_chan)) {
        return std::nullopt;
    }

    auto resolved_overrides = resolve_channel_trimming_for_router(
        capture_overrides,
        builder_context.get_channel_trimming_global_overrides(),
        chip_id,
        eth_chan,
        channel_counts.sender,
        channel_counts.receiver);
    if (!resolved_overrides.has_value()) {
        return std::nullopt;
    }

    return try_derive_vc0_trim_fast_path_info(
        *resolved_overrides, channel_counts.sender[0], builder_context.get_channel_trimming_global_overrides());
}

}  // namespace

ComputeMeshRouterBuilder::ComputeMeshRouterBuilder(
    FabricNodeId local_node,
    const RouterLocation& location,
    std::unique_ptr<FabricEriscDatamoverBuilder> erisc_builder,
    std::optional<FabricTensixDatamoverBuilder> tensix_builder,
    RouterVcShape vc_shape,
    RouterTurnSet turns_by_vc,
    bool downstream_is_tensix_builder,
    std::shared_ptr<ConnectionRegistry> connection_registry) :
    FabricRouterBuilder(local_node, location),
    erisc_builder_(std::move(erisc_builder)),
    tensix_builder_(std::move(tensix_builder)),
    vc_shape_(std::move(vc_shape)),
    turns_by_vc_(std::move(turns_by_vc)),
    downstream_is_tensix_builder_(downstream_is_tensix_builder),
    connection_registry_(std::move(connection_registry)),
    is_inter_mesh_(local_node.mesh_id != location.remote_node.mesh_id) {
    TT_FATAL(erisc_builder_ != nullptr, "Erisc builder cannot be null");
}

std::unique_ptr<ComputeMeshRouterBuilder> ComputeMeshRouterBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricNodeId local_node,
    const RouterLocation& location,
    const ChipRoutingFacts& chip_facts,
    std::shared_ptr<ConnectionRegistry> connection_registry) {
    // Get fabric context and config
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto topology = fabric_context.get_fabric_topology();

    // Convert RoutingDirection to eth_chan_directions
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto eth_direction = control_plane.routing_direction_to_eth_direction(location.direction);

    // Express enablement is resolved once per router and reused below (the MUX guard, the
    // archetype, the injection-flag derivation); the query itself is a cached lazy derivation.
    const bool express_enabled = control_plane.express_routing_enabled(local_node.mesh_id);

    // Get SOC descriptor for eth core lookup
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    auto eth_logical_core = soc_desc.get_eth_core_for_channel(location.eth_chan, CoordSystem::LOGICAL);

    // Determine tensix config
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    bool fabric_tensix_extension_enabled = fabric_tensix_config != FabricTensixConfig::DISABLED;
    bool fabric_tensix_extension_mux_mode = fabric_tensix_config == FabricTensixConfig::MUX;
    bool fabric_tensix_extension_udm_mode = fabric_tensix_config == FabricTensixConfig::UDM;

    // Determine if tensix builder will be created (reusable condition). This is the one link-scope
    // input to the channel layout: a dispatch link never gets a tensix builder downstream, so in
    // MUX mode two same-facing routers can differ here even though every other fact is shared.
    bool will_create_tensix_builder = fabric_tensix_extension_enabled && !location.is_dispatch_link;
    bool downstream_is_tensix_builder = will_create_tensix_builder && fabric_tensix_extension_mux_mode;

    // {express, MUX} is an unsupported combination: the tensix path was never widened for the
    // five-wide express VC0 (its sender count is still the 2D constant, and the tensix builder's
    // downstream EDM count is taken without the express flag), so the variant channel map would
    // index past its end. Fail the configuration with a stated message instead.
    TT_FATAL(
        !(downstream_is_tensix_builder && express_enabled),
        "Express routing with the tensix MUX extension is not supported: the tensix path is not "
        "widened for the express VC0");

    // Get the appropriate EDM config from builder context
    auto tensix_config_for_lookup = will_create_tensix_builder ? fabric_tensix_config : FabricTensixConfig::DISABLED;
    const auto& edm_config = builder_context.get_fabric_router_config(tensix_config_for_lookup, eth_direction);

    // The facts behind this router's mappings are resolved once, each at its own scope: topology
    // and the intermesh VC config (fabric), express_routing_enabled (mesh), the chip's edge
    // capabilities and Z port role (chip, classified once at discovery and threaded in), facing
    // (router); the eth channel enters only at establishment. Both mappings are pure functions of
    // those facts, so routers with an identical fact tuple are byte-identical archetypes -- and
    // constructing one to ask "what would a router like this look like" (the fabric-wide max pass,
    // the peer fast-path query below, which asks about ANOTHER chip and so stays a live query) is
    // intended use.
    //
    // The router exists because discovery found and classified a neighbor in this direction, so
    // the array entry must be present.
    const auto& facing_capability = chip_facts.per_direction_capabilities.at(location.direction);
    TT_FATAL(
        facing_capability.has_value(),
        "Router facing {} has no classified edge from discovery",
        enchantum::to_string(location.direction));
    const auto edge_capability = *facing_capability;
    // What this chip's Z port is for: an intermesh boundary, an express chord, or nothing.
    const auto chip_z_role = z_role_of(chip_facts.per_direction_capabilities);

    // Create the archetype EARLY (the shape is needed for computing injection flags). The
    // Z-related channel shapes exist to reach an intermesh Z router, so they are gated on that
    // rather than on the presence of any Z port. The edge's capability selects the template --
    // an intermesh Z edge gets the from-boundary fanout, an express chord is wired as ordinary
    // same-VC cardinal/Z transitions, and everything else gets the non-express set. The direction
    // selects only the slot arithmetic. The boundary target exists to reach an intermesh Z
    // router, so it follows the intermesh Z edge rather than the presence of any Z port: on a
    // chip whose only Z edge crosses a mesh boundary, an express-style Z target would resolve to
    // the intermesh Z router and leak same-mesh traffic onto the boundary link.
    const auto& intermesh_config = fabric_context.get_builder_context().get_intermesh_vc_config();
    auto archetype = router_archetype(
        topology, location.direction, chip_facts.per_direction_capabilities, express_enabled, &intermesh_config);
    const auto& vc_shape = archetype.shape;

    // Compute injection channel flags at router level BEFORE creating builders
    // Injection semantics are per-VC, so compute for each VC and flatten into router-level vector
    // Injection channel status flags are used by sender channels to understand if that channel must
    // implement bubble flow-control behaviour.

    // First, compute the total number of channels across all VCs
    size_t total_router_channels = 0;
    uint32_t num_vcs = vc_shape.num_vcs;
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        total_router_channels += vc_shape.sender_counts[vc];
    }

    std::vector<bool> router_injection_flags;
    router_injection_flags.reserve(total_router_channels);

    // Express routing derives each producer's effect from the protected-ring facts. The cardinal
    // axis-turn heuristic cannot express it: at an express node the same Z output is transit when fed
    // by the ring and an acquisition when fed by a leaf attachment, and both share one axis pair.
    // Non-express meshes keep the heuristic untouched.
    // This router's producer-slot mapping, shared by every VC's derivation.
    const builder::RouterProducerSlots producer_slots(
        builder::routing_direction_to_eth_direction(location.direction), vc_shape.sender_counts);
    // The policy is a per-router fact: express enablement is mesh-wide, so it is selected once,
    // not per VC. Its facts arrive bound from the chip scope (the queries were bound in the
    // FabricBuilder constructor; the capabilities were classified at discovery).
    const NonExpressInjectionPolicy non_express_policy(topology, eth_direction);
    const ExpressInjectionPolicy express_policy(
        chip_facts.protected_ring_queries,
        chip_facts.per_direction_capabilities,
        chip_z_role,
        location.direction,
        edge_capability);
    const InjectionPolicy& injection_policy = express_enabled ? static_cast<const InjectionPolicy&>(express_policy)
                                                              : static_cast<const InjectionPolicy&>(non_express_policy);

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        uint32_t num_channels_in_vc = vc_shape.sender_counts[vc];
        // A policy answers whether a slot acquires a protected ring; whether that acquisition is
        // guarded is the separate question of which VCs realize bubble flow control. The express
        // derivation classifies a VC1 intermesh landing's first protected egress as an acquisition,
        // which is the correct classification, but VC1 runs unguarded today -- so emitting the flag
        // there would demand a bubble the rest of the stack does not provide.
        auto vc_injection_flags = builder_config::bubble_flow_control_enabled_on_vc(vc)
                                      ? compute_sender_channel_injection_flags(producer_slots, vc, injection_policy)
                                      : std::vector<bool>(num_channels_in_vc, false);
        // Flatten into router-level vector
        for (uint32_t ch_idx = 0; ch_idx < num_channels_in_vc; ++ch_idx) {
            router_injection_flags.push_back(vc_injection_flags.at(ch_idx));
        }
    }

    // Build reverse channel maps and compute injection flags for each builder variant
    // Get ERISC's channel count from config
    size_t erisc_num_channels = edm_config.num_used_sender_channels;
    auto erisc_to_router_channel_map = get_variant_to_router_channel_map(
        vc_shape, downstream_is_tensix_builder, BuilderType::ERISC, erisc_num_channels);
    auto erisc_injection_flags =
        get_child_builder_variant_sender_channel_injection_flags(router_injection_flags, erisc_to_router_channel_map);

    std::vector<bool> tensix_injection_flags;
    if (downstream_is_tensix_builder) {
        size_t tensix_num_channels = builder_config::get_num_tensix_sender_channels(topology, fabric_tensix_config);
        auto tensix_to_router_channel_map = get_variant_to_router_channel_map(
            vc_shape, downstream_is_tensix_builder, BuilderType::TENSIX, tensix_num_channels);
        tensix_injection_flags = get_child_builder_variant_sender_channel_injection_flags(
            router_injection_flags, tensix_to_router_channel_map);
    }

    // Compute actual per-VC channel counts for this router
    std::array<std::size_t, builder_config::MAX_NUM_VCS> actual_sender_channels_per_vc{};
    std::array<std::size_t, builder_config::MAX_NUM_VCS> actual_receiver_channels_per_vc{};
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        actual_sender_channels_per_vc[vc] = vc_shape.sender_counts[vc];
        actual_receiver_channels_per_vc[vc] = vc_shape.receiver_counts[vc];
    }

    // Resolve channel trimming for this router: capture lookup + global override application
    const auto& capture_overrides = builder_context.get_channel_trimming_overrides();
    const bool local_router_has_real_capture_entry =
        has_real_channel_trimming_capture_entry(capture_overrides, device->id(), location.eth_chan);
    auto channel_trimming_overrides_for_router = resolve_channel_trimming_for_router(
        capture_overrides,
        builder_context.get_channel_trimming_global_overrides(),
        device->id(),
        location.eth_chan,
        actual_sender_channels_per_vc,
        actual_receiver_channels_per_vc);

    auto local_vc0_fast_path_info =
        local_router_has_real_capture_entry && channel_trimming_overrides_for_router.has_value()
            ? try_derive_vc0_trim_fast_path_info(
                  *channel_trimming_overrides_for_router,
                  actual_sender_channels_per_vc[0],
                  builder_context.get_channel_trimming_global_overrides())
            : std::nullopt;

    // Inspect the exact peer router on this physical link whenever this local
    // router can use a speedy VC0 path. Its captured sender packet size controls
    // this router's receiver-credit cadence. The terminal speedy-RX decision is
    // separately constrained to the existing non-UDM Fabric2D configuration.
    auto maybe_finalize_vc0_fast_path_pair = [&]() {
        if (!local_vc0_fast_path_info.has_value()) {
            return;
        }

        // A terminal may become speedy only after its exact peer is resolved;
        // all other conditions use the same predicate as the ERISC builder.
        const bool local_can_use_speedy_vc0 =
            vc0_speedy_path_enabled(
                actual_sender_channels_per_vc[0],
                fabric_context.need_deadlock_avoidance_support(control_plane, local_node, eth_direction),
                *local_vc0_fast_path_info) ||
            local_vc0_fast_path_info->terminal_only_nonforwarding;
        if (!local_can_use_speedy_vc0) {
            return;
        }

        const auto connected_peer = control_plane.try_get_connected_mesh_chip_chan_ids(local_node, location.eth_chan);
        if (!connected_peer.has_value()) {
            log_debug(
                tt::LogFabric,
                "Channel trimming: unable to resolve peer for chip {} channel {}; using conservative receiver credit "
                "amortization",
                device->id(),
                location.eth_chan);
            return;
        }
        const auto [connected_peer_node, connected_peer_chan] = *connected_peer;
        if (connected_peer_node != location.remote_node) {
            log_debug(
                tt::LogFabric,
                "Channel trimming: resolved peer {} for chip {} channel {} does not match router peer {}; using "
                "conservative receiver credit amortization",
                connected_peer_node,
                device->id(),
                location.eth_chan,
                location.remote_node);
            return;
        }

        auto peer_direction = control_plane.get_forwarding_direction(connected_peer_node, local_node);
        if (!peer_direction.has_value()) {
            log_debug(
                tt::LogFabric,
                "Channel trimming: unable to determine the peer forwarding direction from {} to {}; using conservative "
                "receiver credit amortization",
                connected_peer_node,
                local_node);
            return;
        }

        // Logical-to-physical mapping membership guarantees that routing-channel metadata was seeded for this node.
        const auto peer_physical_chip_id =
            control_plane.try_get_physical_chip_id_from_fabric_node_id(connected_peer_node);
        if (!peer_physical_chip_id.has_value()) {
            log_debug(
                tt::LogFabric,
                "Channel trimming: peer {} is not mapped to a local physical chip; using conservative receiver "
                "credit amortization",
                connected_peer_node);
            return;
        }
        const auto peer_channel_counts =
            compute_router_channel_counts(fabric_context, control_plane, connected_peer_node, *peer_direction);
        auto peer_vc0_fast_path_info = resolve_vc0_trim_fast_path_info(
            builder_context, *peer_physical_chip_id, connected_peer_chan, peer_channel_counts);
        if (!peer_vc0_fast_path_info.has_value()) {
            log_debug(
                tt::LogFabric,
                "Channel trimming: no peer capture entry for chip {} channel {}; using conservative receiver credit "
                "amortization",
                *peer_physical_chip_id,
                connected_peer_chan);
            return;
        }

        apply_vc0_trim_fast_path_peer_info(
            *local_vc0_fast_path_info,
            *peer_vc0_fast_path_info,
            fabric_context.is_2D_routing_enabled() && local_node.mesh_id == location.remote_node.mesh_id &&
                !fabric_tensix_extension_udm_mode && !location.is_dispatch_link);
    };
    maybe_finalize_vc0_fast_path_pair();

    // The stream-register assignment is fabric-scoped (shared per mesh) and lives in
    // FabricBuilderContext; the erisc and tensix builders read it from there, so every router in
    // the mesh agrees on the flat-channel -> register-id map by construction.
    auto edm_builder = std::make_unique<FabricEriscDatamoverBuilder>(FabricEriscDatamoverBuilder::build(
        device,
        program,
        eth_logical_core,
        local_node,
        location.remote_node,
        edm_config,
        std::move(erisc_injection_flags),
        false, /* build_in_worker_connection_mode */
        eth_direction,
        downstream_is_tensix_builder,
        actual_sender_channels_per_vc,
        actual_receiver_channels_per_vc,
        channel_trimming_overrides_for_router,
        local_vc0_fast_path_info));

    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode()) {
        // Enable updates at a fixed interval for link stability and link status updates
        constexpr uint32_t k_BlackholeFabricRouterContextSwitchInterval = 32;
        edm_builder->set_firmware_context_switch_interval(k_BlackholeFabricRouterContextSwitchInterval);
        edm_builder->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE);
    }

    // Create tensix builder if needed
    std::optional<FabricTensixDatamoverBuilder> tensix_builder_opt;
    if (will_create_tensix_builder) {
        tensix_builder_opt = FabricTensixDatamoverBuilder::build(
            device,
            program,
            local_node,
            location.remote_node,
            location.eth_chan,
            eth_direction,
            std::move(tensix_injection_flags));
    }

    // Use unique_ptr constructor directly since ComputeMeshRouterBuilder constructor is private
    auto router_builder = std::unique_ptr<ComputeMeshRouterBuilder>(new ComputeMeshRouterBuilder(
        local_node,
        location,
        std::move(edm_builder),
        std::move(tensix_builder_opt),
        std::move(archetype.shape),
        std::move(archetype.turns),
        downstream_is_tensix_builder,
        std::move(connection_registry)));

    // Setup the local relay kernel connection if in UDM mode
    if (fabric_tensix_extension_udm_mode && router_builder->has_tensix_builder()) {
        router_builder->connect_to_local_tensix_builder(router_builder->get_tensix_builder());
    }

    return router_builder;
}

uint32_t ComputeMeshRouterBuilder::get_downstream_sender_channel(
    const bool is_2D_routing, const eth_chan_directions downstream_direction, uint32_t vc) const {
    // Which slot on the downstream router this router (as producer) feeds. The 2D establishment
    // path reads it off the downstream router's own RouterProducerSlots at the call site; this
    // delegates to the free helper, whose 1D branch (a single forwarding channel, no compact
    // ranking) is the one in use.
    return builder::get_downstream_sender_channel_for_vc(
        is_2D_routing, vc, this->get_eth_direction(), downstream_direction);
}

eth_chan_directions ComputeMeshRouterBuilder::get_eth_direction() const { return erisc_builder_->get_direction(); }

size_t ComputeMeshRouterBuilder::get_noc_x() const { return erisc_builder_->get_noc_x(); }

size_t ComputeMeshRouterBuilder::get_noc_y() const { return erisc_builder_->get_noc_y(); }

size_t ComputeMeshRouterBuilder::get_configured_risc_count() const {
    return erisc_builder_->get_configured_risc_count();
}

std::vector<bool> ComputeMeshRouterBuilder::get_child_builder_variant_sender_channel_injection_flags(
    const std::vector<bool>& router_injection_flags,
    const std::vector<std::optional<size_t>>& variant_to_router_channel_map) {
    std::vector<bool> variant_injection_flags;
    variant_injection_flags.reserve(variant_to_router_channel_map.size());

    // Iterate through variant's internal channels in order (0, 1, 2, ...)
    // For each variant channel, look up its corresponding router channel and get the injection flag
    for (auto router_channel_opt : variant_to_router_channel_map) {
        if (router_channel_opt.has_value()) {
            // Channel is externally-facing, get injection status from router
            size_t router_channel_id = *router_channel_opt;
            TT_FATAL(
                router_channel_id < router_injection_flags.size(),
                "Internal error: Router channel ID {} out of bounds (max {})",
                router_channel_id,
                router_injection_flags.size());
            variant_injection_flags.push_back(router_injection_flags.at(router_channel_id));
        } else {
            // Channel is internal-only (e.g., ERISC in MUX mode fed by TENSIX)
            // Internal channels are never injection channels
            variant_injection_flags.push_back(false);
        }
    }

    return variant_injection_flags;
}

std::vector<std::optional<size_t>> ComputeMeshRouterBuilder::get_variant_to_router_channel_map(
    const RouterVcShape& vc_shape,
    bool downstream_is_tensix_builder,
    BuilderType builder_type,
    size_t variant_num_sender_channels) {
    // Walk the shape's (vc, channel) pairs and keep the ones this variant owns -- ownership is
    // decided per VC, so the check hoists out of the channel loop. A variant's internal channel
    // ID and the router's flat channel index are the same number today, so the map is the
    // identity over the owned channels; the shape carries the flat bases that make it so.
    std::vector<std::optional<size_t>> variant_to_router_channel_map(variant_num_sender_channels);

    for (uint32_t vc = 0; vc < vc_shape.num_vcs; ++vc) {
        if (builder_type_for_vc(vc, downstream_is_tensix_builder) != builder_type) {
            continue;
        }
        for (uint32_t ch = 0; ch < vc_shape.sender_counts[vc]; ++ch) {
            const size_t router_flat_id = vc_shape.flat_sender_id(vc, ch);
            TT_FATAL(
                router_flat_id < variant_num_sender_channels,
                "Builder variant {} owns flat channel {} on VC{} but has only {} sender channels: the "
                "variant's channel space was not widened for this router's shape",
                enchantum::to_string(builder_type),
                router_flat_id,
                vc,
                variant_num_sender_channels);
            variant_to_router_channel_map[router_flat_id] = router_flat_id;
        }
    }

    return variant_to_router_channel_map;
}

void ComputeMeshRouterBuilder::connect_to_local_tensix_builder(FabricTensixDatamoverBuilder& tensix_builder) {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    TT_FATAL(is_2D_routing, "connect_to_local_tensix_builder requires 2D routing");

    // In UDM mode, router receiver connects to local relay on the tensix core
    // Get connection specs from relay builder to set up receiver-to-relay connection
    // Relay only has one channel (ROUTER_CHANNEL = 0) for upstream fabric router traffic
    eth_chan_directions local_tensix_dir = tensix_builder.get_direction();
    auto adapter_spec = tensix_builder.build_connection_to_relay_channel();

    // Enable UDM mode and store relay buffer count
    erisc_builder_->udm_mode = true;
    erisc_builder_->local_tensix_relay_num_buffers = adapter_spec.num_buffers_per_channel;

    // Only one local relay connection (we can consider the local relay connection as one ds tensix connection)
    erisc_builder_->num_downstream_tensix_connections = 1;

    auto* adapter_ptr = erisc_builder_->receiver_channel_to_downstream_adapter.get();
    const auto tensix_noc_x = tensix_builder.get_noc_x();
    const auto tensix_noc_y = tensix_builder.get_noc_y();
    adapter_ptr->add_local_tensix_connection(adapter_spec, local_tensix_dir, tt::tt_metal::CoreCoord(tensix_noc_x, tensix_noc_y));

    // Provide router NOC coordinates to relay kernel for sending packets back to router
    tensix_builder.append_relay_router_noc_xy(erisc_builder_->get_noc_x(), erisc_builder_->get_noc_y());
}

void ComputeMeshRouterBuilder::establish_connections_to_router(ComputeMeshRouterBuilder& downstream_router) {
    // Establish VC connections between this router and the specified downstream router
    // This function does NOT iterate through targets - it connects to the single downstream_router passed in.
    // Every direction-matching target is established: there is no longer a type-based filter --
    // the boundary turns (to and from the intermesh Z router) are wired through the same path as
    // every other local turn, which is what made the second establishment pass redundant.
    uint32_t num_vcs = vc_shape_.num_vcs;

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();

    // The downstream router's own producer-slot mapping, with its actual channel counts: the slot
    // this router (as producer) feeds is read off the same mapping the injection-flag derivation
    // names producers from, so placement and naming cannot drift apart.
    const builder::RouterProducerSlots downstream_slots(
        downstream_router.get_eth_direction(), downstream_router.vc_shape_.sender_counts);

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        const auto& targets = turns_by_vc_[vc];
        log_debug(
            LogMetal,
            "Router at x={}, y={}, Channel={}, Direction={}, FabricNodeId={} :: VC{} has {} targets",
            get_noc_x(),
            get_noc_y(),
            location_.eth_chan,
            get_eth_direction(),
            local_node_,
            vc,
            targets.size());
        for (const auto& target : targets) {
            // Only apply direction filtering for 2D routing or when there are multiple targets
            // For 1D routing with a single target, skip direction check to avoid false negatives
            bool should_check_direction = is_2D_routing || targets.size() > 1;
            if (should_check_direction && target.target_direction.has_value() &&
                target.target_direction.value() != downstream_router.get_location().direction) {
                // connection mapping contains connection info to all downstream directions.
                // Proceed with connection setup only for the current downstream direction.
                log_debug(
                    LogMetal,
                    "Skipping VC{} connection to direction({}), Downstream router is at direction({})",
                    vc,
                    target.target_direction.value(),
                    downstream_router.get_location().direction);
                continue;
            }

            // Compute the sender channel on the downstream router this router (as producer) feeds.
            // 2D reads it off the downstream's own slot mapping (constructed above); 1D keeps its
            // own layout -- a single forwarding channel, no compact ranking.
            uint32_t downstream_sender_channel;
            if (is_2D_routing) {
                const auto slot = downstream_slots.channel_for(vc, get_eth_direction());
                TT_FATAL(
                    slot.has_value(),
                    "The {}-facing producer has no slot on the downstream router's VC{}: the turn is "
                    "wired but that router does not have the slot",
                    enchantum::to_string(get_eth_direction()),
                    vc);
                downstream_sender_channel = *slot;
            } else {
                downstream_sender_channel =
                    get_downstream_sender_channel(is_2D_routing, downstream_router.get_eth_direction(), vc);

                // 1D computes without the downstream's counts, so validate against them before
                // taking the flat channel ID from its prefix sums (bounds-checked there as well).
                TT_FATAL(
                    downstream_sender_channel < downstream_router.vc_shape_.sender_counts[vc],
                    "Computed downstream sender channel {} exceeds available channels ({}) for VC{} on downstream "
                    "router",
                    downstream_sender_channel,
                    downstream_router.vc_shape_.sender_counts[vc],
                    vc);
            }

            // Get downstream builder and the flat channel ID
            auto* downstream_builder = downstream_router.get_builder_for_vc_channel(vc, downstream_sender_channel);

            // Get both absolute and VC-relative channel IDs
            // - absolute_channel_id: flattened across all VCs (used for flat arrays)
            // - vc_relative_channel_id: 0-based within the VC (used for allocator calls)
            uint32_t absolute_channel_id = downstream_router.vc_shape_.flat_sender_id(vc, downstream_sender_channel);
            uint32_t vc_relative_channel_id = downstream_sender_channel;  // This is already VC-relative

            // Setup producer → consumer connection
            // Pass both indices - build_connection_to_fabric_channel needs both
            erisc_builder_->setup_downstream_vc_connection(
                downstream_builder, vc, vc, absolute_channel_id, vc_relative_channel_id);
            // Record connection in registry if present
            if (connection_registry_) {
                RouterConnectionRecord record{
                    .source_node = local_node_,
                    .source_direction = location_.direction,
                    .source_eth_chan = location_.eth_chan,
                    .source_vc = vc,
                    .source_receiver_channel = 0,
                    .dest_node = downstream_router.local_node_,
                    .dest_direction = downstream_router.location_.direction,
                    .dest_eth_chan = downstream_router.location_.eth_chan,
                    .dest_vc = vc,
                    .dest_sender_channel = absolute_channel_id};
                connection_registry_->record_connection(record);
            }

            log_debug(
                tt::LogTest,
                "M{}-D{}: Router at x={}, y={}, UpstreamEthernetChannel={}, Direction={} :: Connecting VC{} "
                "receiver_ch={} to "
                "downstream "
                "router at x={}, y={}, DownstreamEthernetChannel={}, Direction={}, VC{}, RelativeChannel={}, "
                "AbsoluteChannel={}",
                local_node_.mesh_id.get(),
                local_node_.chip_id,
                get_noc_x(),
                get_noc_y(),
                location_.eth_chan,
                get_eth_direction(),
                vc,
                0,
                downstream_builder->get_noc_x(),
                downstream_builder->get_noc_y(),
                downstream_router.location_.eth_chan,
                downstream_builder->get_direction(),
                vc,
                vc_relative_channel_id,
                absolute_channel_id);
        }
    }
}

void ComputeMeshRouterBuilder::configure_connection(
    FabricRouterBuilder& peer, uint32_t link_idx, uint32_t num_links, Topology topology, bool is_galaxy) {
    // Validate invariant: FabricBuilder guarantees all routers on a device are the same concrete type
    auto* peer_compute_ptr = dynamic_cast<ComputeMeshRouterBuilder*>(&peer);
    TT_FATAL(
        peer_compute_ptr != nullptr,
        "Router type mismatch: expected ComputeMeshRouterBuilder but got different type. "
        "This indicates a bug in FabricBuilder::create_routers()");
    auto& peer_compute = *peer_compute_ptr;

    TT_FATAL(
        !erisc_builder_->build_in_worker_connection_mode,
        "Tried to connect router to downstream in worker connection mode");

    // Establish every direction-matching connection between the two routers (bidirectional).
    // There is no type filter anymore: the boundary turns (to and from the intermesh Z router) are
    // established here like every other local turn, so this is the single establishment pass for
    // all of them.
    establish_connections_to_router(peer_compute);
    peer_compute.establish_connections_to_router(*this);

    // Configure NOC VC based on link index (must be same for both routers)
    auto edm_noc_vc = tt::tt_fabric::FabricEriscDatamoverConfig::DEFAULT_NOC_VC +
                      (link_idx % tt::tt_fabric::FabricEriscDatamoverConfig::NUM_EDM_NOC_VCS);
    erisc_builder_->config.edm_noc_vc = edm_noc_vc;
    peer_compute.erisc_builder_->config.edm_noc_vc = edm_noc_vc;

    // Apply core placement optimizations
    core_placement::CorePlacementContext cctx{
        .topology = topology,
        .is_galaxy = is_galaxy,
        .num_links = num_links,
    };
    core_placement::apply_core_placement_optimizations(cctx, *erisc_builder_, *peer_compute.erisc_builder_, link_idx);
}

void ComputeMeshRouterBuilder::configure_for_dispatch() {
    // Dispatch requires higher context switching frequency to service slow dispatch / UMD / debug tools
    constexpr uint32_t k_DispatchFabricRouterContextSwitchInterval = 16;
    erisc_builder_->set_firmware_context_switch_interval(k_DispatchFabricRouterContextSwitchInterval);
    erisc_builder_->set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType::INTERVAL);
}

void ComputeMeshRouterBuilder::compile_ancillary_kernels(tt::tt_metal::Program& program) {
    // Compile tensix builder if present
    if (tensix_builder_.has_value()) {
        tensix_builder_->create_and_compile(program);
    }
}

void ComputeMeshRouterBuilder::create_kernel(tt::tt_metal::Program& program, const KernelCreationContext& ctx) {
    // Build defines
    std::map<std::string, std::string> defines = {};
    if (ctx.is_2D_routing) {
        defines["FABRIC_2D"] = "";

        // FABRIC_2D_VC1_ACTIVE: Set when router actually has VC1 channels
        bool vc1_active = vc_shape_.sender_counts[1] > 0;
        if (vc1_active) {
            defines["FABRIC_2D_VC1_ACTIVE"] = "";
        }

        // FABRIC_2D_VC1_SERVICED: Set when router actively services VC1 traffic
        // - Intra-mesh routers service VC1 when full_mesh mode is enabled
        // - Inter-mesh routers service VC1 when pass_through mode is enabled
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const auto& intermesh_config = fabric_context.get_builder_context().get_intermesh_vc_config();

        // VC1 is serviced when:
        // - Intra-mesh router with full mesh VC1, or
        // - Inter-mesh router with pass-through VC1
        bool vc1_serviced = (!is_inter_mesh_ && intermesh_config.requires_vc1_full_mesh) ||
                            (is_inter_mesh_ && intermesh_config.requires_vc1_mesh_pass_through);

        if (vc1_serviced) {
            defines["FABRIC_2D_VC1_SERVICED"] = "";
        }

        // FABRIC_2D_VC2_SERVICED: Set when router has VC2 sender channels
        bool vc2_active = vc_shape_.sender_counts[2] > 0;
        if (vc2_active) {
            defines["FABRIC_2D_VC2_SERVICED"] = "";
        }

        // FABRIC_2D_VC0_CROSSOVER_TO_VC1: Set for inter-mesh routers that perform VC0→VC1 crossover
        // Inter-mesh routers crossover incoming VC0 traffic to downstream intra-mesh VC1
        bool vc0_crossover_to_vc1 = is_inter_mesh_ && intermesh_config.requires_vc1_full_mesh;
        if (vc0_crossover_to_vc1) {
            defines["FABRIC_2D_VC0_CROSSOVER_TO_VC1"] = "";
        }

        // FABRIC_2D selects action-map decode for every 2D router. FABRIC_EXPRESS_ENABLED is omitted;
        // its remaining device use is worker-side Z-port capacity.
    }

    // Get SOC descriptor for eth core lookup
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto device_id = control_plane.get_physical_chip_id_from_fabric_node_id(local_node_);
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    const auto eth_chan = location_.eth_chan;
    auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);

    // Configure for host signal wait
    erisc_builder_->set_wait_for_host_signal(true);

    // Get runtime args (same for all RISC cores)
    const std::vector<uint32_t> rt_args = erisc_builder_->get_runtime_args();

    const auto num_enabled_risc_cores = get_configured_risc_count();

    for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
        // Get compile-time args (positional + named) and append cluster-wide coordination info
        auto [ct_args, named_ct_args] = erisc_builder_->get_compile_time_args(risc_id);

        const auto is_master_risc_core = (eth_chan == ctx.master_router_chan) && (risc_id == 0);
        named_ct_args["IS_LOCAL_HANDSHAKE_MASTER"] = is_master_risc_core;
        named_ct_args["LOCAL_HANDSHAKE_MASTER_ETH_CHAN"] = ctx.master_router_chan;
        named_ct_args["NUM_LOCAL_EDMS"] = ctx.num_local_fabric_routers;
        named_ct_args["EDM_CHANNELS_MASK"] = ctx.router_channels_mask;

        // Determine processor
        auto proc = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id);
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
            tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode() &&
            num_enabled_risc_cores == 1) {
            // Force fabric to run on erisc1 due to stack usage exceeded with MUX on erisc0
            proc = tt::tt_metal::DataMovementProcessor::RISCV_1;
        }

        auto opt_level = erisc_builder_->get_kernel_opt_level();

        // Create the kernel
        auto kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
            eth_logical_core,
            tt::tt_metal::EthernetConfig{
                .noc = erisc_builder_->config.risc_configs[risc_id].get_configured_noc(),
                .processor = proc,
                .compile_args = ct_args,
                .defines = defines,
                .named_compile_args = named_ct_args,
                .opt_level = opt_level});

        tt::tt_metal::SetRuntimeArgs(program, kernel, eth_logical_core, rt_args);
    }

    log_debug(
        tt::LogMetal,
        "Fabric router kernel created: eth_chan={}, direction={}, is_master={}",
        eth_chan,
        get_eth_direction(),
        eth_chan == ctx.master_router_chan);
}

FabricDatamoverBuilderBase* ComputeMeshRouterBuilder::get_builder_for_vc_channel(
    uint32_t vc, uint32_t /*channel*/) const {
    // Ownership is decided per VC: the tensix extension takes all of VC0, so the channel index
    // does not factor in.
    if (builder_type_for_vc(vc, downstream_is_tensix_builder_) == BuilderType::TENSIX) {
        TT_FATAL(tensix_builder_.has_value(), "Tensix builder required but not present");
        return const_cast<FabricTensixDatamoverBuilder*>(&tensix_builder_.value());
    }
    return erisc_builder_.get();
}

}  // namespace tt::tt_fabric

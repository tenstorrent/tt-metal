// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_env_impl.hpp"

namespace tt::tt_metal {

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::RoutingDirection;

// The worker semaphores of one connection: teardown and buffer index.
constexpr size_t kSemaphoresPerConnection = 2;

// Where one target of a spec goes on a device: the direction of the first hop, and the fabric node of that hop.
struct ResolvedFabricConnectionTarget {
    RoutingDirection direction;
    FabricNodeId next_hop;
};

// The neighbour of `src` on its own mesh in `direction`, if it has one. A 1D fabric has one mesh, so a 1D target
// never connects to another mesh.
std::optional<FabricNodeId> NeighbourInDirection(
    const tt::tt_fabric::ControlPlane& control_plane, const FabricNodeId& src, RoutingDirection direction) {
    const auto chips = control_plane.get_intra_chip_neighbors(src, direction);
    if (chips.empty()) {
        return std::nullopt;
    }
    return FabricNodeId(src.mesh_id, chips.front());
}

// Each target of `connection` resolved on the device of `src`, in target order; nullopt if not connected there.
std::vector<std::optional<ResolvedFabricConnectionTarget>> ResolveTargets(
    const tt::tt_fabric::ControlPlane& control_plane,
    bool is_2d_fabric,
    const FabricNodeId& src,
    const FabricConnectionSpec& connection) {
    std::vector<std::optional<ResolvedFabricConnectionTarget>> resolved;
    if (const auto* directions = std::get_if<Fabric1DTargets>(&connection.targets)) {
        TT_FATAL(
            !is_2d_fabric,
            "FabricConnectionSpec '{}' has 1D targets, but the fabric is not in a 1D mode",
            connection.unique_id);
        std::ranges::transform(*directions, std::back_inserter(resolved), [&](RoutingDirection direction) {
            const auto neighbour = NeighbourInDirection(control_plane, src, direction);
            return neighbour.has_value() ? std::optional(ResolvedFabricConnectionTarget{direction, *neighbour})
                                         : std::nullopt;
        });
    } else {
        TT_FATAL(
            is_2d_fabric,
            "FabricConnectionSpec '{}' has 2D targets, but the fabric is not in a 2D mode",
            connection.unique_id);
        TT_THROW("FabricConnectionSpec '{}': 2D targets are not supported yet", connection.unique_id);
    }

    std::set<RoutingDirection> directions;
    for (const auto& target : resolved) {
        if (target.has_value()) {
            TT_FATAL(
                directions.insert(target->direction).second,
                "FabricConnectionSpec '{}': two targets resolve to direction {} on fabric node {}",
                connection.unique_id,
                target->direction,
                src);
        }
    }
    return resolved;
}

}  // namespace

FabricConnectionsBySpec RegisterFabricConnections(
    distributed::MeshDevice& mesh_device,
    const std::optional<distributed::MeshCoordinate>& device,
    const ProgramSpec& spec,
    detail::ProgramImpl& program_impl) {
    FabricConnectionsBySpec connections;
    if (spec.fabric_connections.empty()) {
        return connections;
    }
    TT_FATAL(
        device.has_value(),
        "A Program with fabric connections must run on exactly one device: the connection args differ per device");
    TT_FATAL(
        tt::tt_fabric::GetFabricConfig(mesh_device) != tt::tt_fabric::FabricConfig::DISABLED,
        "A Program with fabric connections needs the fabric to be enabled");

    const auto& control_plane = mesh_device.impl().metal_env().get_control_plane();
    TT_FATAL(
        control_plane.get_fabric_tensix_config() != tt::tt_fabric::FabricTensixConfig::UDM,
        "A Program with fabric connections does not support the UDM fabric tensix config yet");
    const bool is_2d_fabric = control_plane.get_fabric_context().is_2D_routing_enabled();
    const FabricNodeId src = mesh_device.get_fabric_node_id(*device);
    const bool exclusive_links = !spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check;
    std::set<std::pair<RoutingDirection, uint32_t>> used_links;

    for (const auto& connection : spec.fabric_connections) {
        const auto targets = ResolveTargets(control_plane, is_2d_fabric, src, connection);
        ResolvedFabricConnection& resolved = connections[connection.unique_id];
        resolved.num_targets = static_cast<uint32_t>(targets.size());
        for (const auto& [node, link] : connection.links) {
            std::vector<uint32_t>& words = resolved.words_by_node[node];
            for (const auto& target : targets) {
                if (!target.has_value()) {
                    words.insert(words.end(), FabricConnectionBindingHandle::kWordsPerTarget, 0u);
                    continue;
                }
                const size_t num_links =
                    control_plane.get_active_fabric_eth_channels_in_direction(src, target->direction).size();
                TT_FATAL(
                    link < num_links,
                    "FabricConnectionSpec '{}': node {} uses link {}, but direction {} of fabric node {} has {} links",
                    connection.unique_id,
                    node.str(),
                    link,
                    target->direction,
                    src,
                    num_links);
                TT_FATAL(
                    used_links.emplace(target->direction, link).second || !exclusive_links,
                    "FabricConnectionSpec '{}': (direction {}, link {}) of fabric node {} is used by more than one "
                    "fabric connection of the Program",
                    connection.unique_id,
                    target->direction,
                    link,
                    src);
                const auto eth_channel = tt::tt_fabric::get_worker_fabric_router_channel(
                    control_plane, src, target->next_hop, target->direction, link);
                tt::tt_fabric::record_worker_fabric_connection_for_emule(src, target->next_hop, node, target->direction);
                const size_t num_node_semaphores = program_impl.semaphores_on_core(node, CoreType::WORKER).size();
                TT_FATAL(
                    num_node_semaphores + kSemaphoresPerConnection <= NUM_SEMAPHORES,
                    "FabricConnectionSpec '{}': node {} has no semaphores left for the connection to target {}. Each "
                    "connected target of a node uses {} of the {} semaphores of the node, and the Program already "
                    "uses {} on the node. Hint: use fewer SemaphoreSpecs or fewer connected fabric connection targets "
                    "on the node",
                    connection.unique_id,
                    node.str(),
                    target->next_hop,
                    kSemaphoresPerConnection,
                    NUM_SEMAPHORES,
                    num_node_semaphores);
                const CoreRangeSet node_set(CoreRange(node, node));
                const uint32_t teardown_semaphore_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER);
                const uint32_t buffer_index_semaphore_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER);
                words.insert(
                    words.end(),
                    {1u, static_cast<uint32_t>(eth_channel), teardown_semaphore_id, buffer_index_semaphore_id});
            }
        }
    }
    return connections;
}

FabricConnectionBindingsForKernel ResolveFabricConnectionBindingsForKernel(
    const KernelSpec& kernel, const FabricConnectionsBySpec& connections, uint32_t fabric_connection_base_rta_word) {
    FabricConnectionBindingsForKernel out;
    uint32_t rta_word = fabric_connection_base_rta_word;
    for (const auto& binding : kernel.fabric_connection_bindings) {
        const ResolvedFabricConnection& resolved = connections.at(binding.fabric_connection_spec_name);
        out.handles.push_back(FabricConnectionBindingHandle{
            .accessor_name = binding.accessor_name,
            .rta_word = rta_word,
            .num_targets = resolved.num_targets,
            .words_by_node = resolved.words_by_node,
        });
        rta_word += resolved.num_targets * FabricConnectionBindingHandle::kWordsPerTarget;
    }
    out.section_words = rta_word - fabric_connection_base_rta_word;
    return out;
}

}  // namespace tt::tt_metal

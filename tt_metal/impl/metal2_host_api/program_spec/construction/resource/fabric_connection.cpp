// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <set>
#include <unordered_map>
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

// The neighbour of `src` on its own mesh in direction target `direction` of `connection`. A 1D fabric has one mesh, so
// a 1D target never connects to another mesh.
FabricNodeId NeighbourInDirection(
    const tt::tt_fabric::ControlPlane& control_plane,
    const experimental::FabricConnectionsSpec& connection,
    const FabricNodeId& src,
    RoutingDirection direction) {
    const auto chips = control_plane.get_intra_chip_neighbors(src, direction);
    TT_FATAL(
        !chips.empty(),
        "FabricConnectionsSpec '{}': fabric node {} has no fabric neighbour in direction {}. Hint: the op handles the "
        "edges of the fabric: give the devices at an edge their own ProgramSpec that lists only the directions they "
        "have (one ProgramSpec per device range of a MeshWorkload), or that does not bind the spec",
        connection.unique_id,
        src,
        direction);
    return FabricNodeId(src.mesh_id, chips.front());
}

// Each target of `connection` resolved on the device of `src`, in target order.
std::vector<ResolvedFabricConnectionTarget> ResolveTargets(
    const tt::tt_fabric::ControlPlane& control_plane,
    bool is_2d_fabric,
    const FabricNodeId& src,
    const experimental::FabricConnectionsSpec& connection) {
    std::vector<ResolvedFabricConnectionTarget> resolved;
    if (const auto* directions = std::get_if<experimental::Fabric1DTargets>(&connection.targets)) {
        TT_FATAL(
            !is_2d_fabric,
            "FabricConnectionsSpec '{}' has 1D targets, but the fabric is not in a 1D mode",
            connection.unique_id);
        std::ranges::transform(*directions, std::back_inserter(resolved), [&](RoutingDirection direction) {
            return ResolvedFabricConnectionTarget{
                direction, NeighbourInDirection(control_plane, connection, src, direction)};
        });
    } else {
        TT_FATAL(
            is_2d_fabric,
            "FabricConnectionsSpec '{}' has 2D targets, but the fabric is not in a 2D mode",
            connection.unique_id);
        TT_THROW("FabricConnectionsSpec '{}': 2D targets are not supported yet", connection.unique_id);
    }

    std::set<RoutingDirection> directions;
    for (const auto& target : resolved) {
        TT_FATAL(
            directions.insert(target.direction).second,
            "FabricConnectionsSpec '{}': two targets resolve to direction {} on fabric node {}",
            connection.unique_id,
            target.direction,
            src);
    }
    return resolved;
}

// The semaphore ids of the connections of one (node, target).
struct ConnectionSemaphores {
    uint32_t teardown_id;
    uint32_t buffer_index_id;
};

// Allocates the two worker semaphores of the connections of target `target_index` of `connection` on `node`.
// side-effect: adds the semaphores to `program_impl`.
ConnectionSemaphores CreateConnectionSemaphores(
    detail::ProgramImpl& program_impl,
    const experimental::FabricConnectionsSpec& connection,
    const NodeCoord& node,
    uint32_t target_index) {
    const size_t num_node_semaphores = program_impl.semaphores_on_core(node, CoreType::WORKER).size();
    TT_FATAL(
        num_node_semaphores + kSemaphoresPerConnection <= NUM_SEMAPHORES,
        "FabricConnectionsSpec '{}': node {} has no semaphores left for the connection of target {}. Each "
        "target of a node uses {} of the {} semaphores of the node, and the Program already uses {} on the node. "
        "Hint: use fewer SemaphoreSpecs or fewer fabric connection targets on the node",
        connection.unique_id,
        node.str(),
        target_index,
        kSemaphoresPerConnection,
        NUM_SEMAPHORES,
        num_node_semaphores);
    const CoreRangeSet node_set(CoreRange(node, node));
    return ConnectionSemaphores{
        .teardown_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER),
        .buffer_index_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER)};
}

}  // namespace

FabricConnectionsBySpec RegisterFabricConnections(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinateRange& devices,
    const ProgramSpec& spec,
    detail::ProgramImpl& program_impl) {
    FabricConnectionsBySpec connections;
    if (spec.advanced_options.fabric_connections.empty()) {
        return connections;
    }
    TT_FATAL(
        tt::tt_fabric::GetFabricConfig(mesh_device) != tt::tt_fabric::FabricConfig::DISABLED,
        "A Program with fabric connections needs the fabric to be enabled");

    const auto& control_plane = mesh_device.impl().metal_env().get_control_plane();
    TT_FATAL(
        control_plane.get_fabric_tensix_config() != tt::tt_fabric::FabricTensixConfig::UDM,
        "A Program with fabric connections does not support the UDM fabric tensix config yet");
    const bool is_2d_fabric = control_plane.get_fabric_context().is_2D_routing_enabled();
    const std::vector<distributed::MeshCoordinate> coords(devices.begin(), devices.end());
    std::vector<FabricNodeId> srcs;
    std::ranges::transform(coords, std::back_inserter(srcs), [&](const distributed::MeshCoordinate& device) {
        return mesh_device.get_fabric_node_id(device);
    });
    const bool exclusive_links = !spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check;
    std::vector<std::set<std::pair<RoutingDirection, uint32_t>>> used_links_by_device(srcs.size());

    for (const auto& connection : spec.advanced_options.fabric_connections) {
        std::vector<std::vector<ResolvedFabricConnectionTarget>> targets_by_device;
        std::ranges::transform(srcs, std::back_inserter(targets_by_device), [&](const FabricNodeId& src) {
            return ResolveTargets(control_plane, is_2d_fabric, src, connection);
        });
        ResolvedFabricConnection& resolved = connections[connection.unique_id];
        resolved.num_targets = static_cast<uint32_t>(targets_by_device.front().size());
        resolved.words_per_target = FabricConnectionBindingHandle::kWordsPerTarget1D;
        std::ranges::transform(coords, std::back_inserter(resolved.words_by_device), [](const auto& coord) {
            return std::make_pair(coord, std::unordered_map<NodeCoord, std::vector<uint32_t>>{});
        });
        for (const auto& [node, link] : connection.links) {
            for (uint32_t target_index = 0; target_index < resolved.num_targets; ++target_index) {
                const ConnectionSemaphores semaphores =
                    CreateConnectionSemaphores(program_impl, connection, node, target_index);
                for (size_t device_index = 0; device_index < srcs.size(); ++device_index) {
                    const FabricNodeId& src = srcs[device_index];
                    const ResolvedFabricConnectionTarget& target = targets_by_device[device_index][target_index];
                    std::vector<uint32_t>& words = resolved.words_by_device[device_index].second[node];
                    const size_t num_links =
                        control_plane.get_active_fabric_eth_channels_in_direction(src, target.direction).size();
                    TT_FATAL(
                        link < num_links,
                        "FabricConnectionsSpec '{}': node {} uses link {}, but direction {} of fabric node {} has {} "
                        "links",
                        connection.unique_id,
                        node.str(),
                        link,
                        target.direction,
                        src,
                        num_links);
                    TT_FATAL(
                        used_links_by_device[device_index].emplace(target.direction, link).second || !exclusive_links,
                        "FabricConnectionsSpec '{}': (direction {}, link {}) of fabric node {} is used by more than "
                        "one "
                        "fabric connection of the Program",
                        connection.unique_id,
                        target.direction,
                        link,
                        src);
                    const auto eth_channel = tt::tt_fabric::get_worker_fabric_router_channel(
                        control_plane, src, target.next_hop, target.direction, link);
                    tt::tt_fabric::record_worker_fabric_connection_for_emule(
                        src, target.next_hop, node, target.direction);
                    words.insert(
                        words.end(),
                        {static_cast<uint32_t>(eth_channel), semaphores.teardown_id, semaphores.buffer_index_id});
                }
            }
        }
    }
    return connections;
}

FabricConnectionBindingsForKernel ResolveFabricConnectionBindingsForKernel(
    const KernelSpec& kernel, const FabricConnectionsBySpec& connections, uint32_t fabric_connection_base_rta_word) {
    FabricConnectionBindingsForKernel out;
    uint32_t rta_word = fabric_connection_base_rta_word;
    for (const auto& binding : kernel.advanced_options.fabric_connection_bindings) {
        const ResolvedFabricConnection& resolved = connections.at(binding.fabric_connections_spec_name);
        out.handles.push_back(FabricConnectionBindingHandle{
            .accessor_name = binding.accessor_name,
            .rta_word = rta_word,
            .num_targets = resolved.num_targets,
            .words_per_target = resolved.words_per_target,
            .words_by_device = resolved.words_by_device,
        });
        rta_word += resolved.num_targets * resolved.words_per_target;
    }
    out.section_words = rta_word - fabric_connection_base_rta_word;
    return out;
}

}  // namespace tt::tt_metal

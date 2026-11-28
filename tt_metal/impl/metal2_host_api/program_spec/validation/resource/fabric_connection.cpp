// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal {

namespace {

bool IsAllowedDirectionTarget(tt::tt_fabric::RoutingDirection direction) {
    using tt::tt_fabric::RoutingDirection;
    return direction == RoutingDirection::N || direction == RoutingDirection::E || direction == RoutingDirection::S ||
           direction == RoutingDirection::W || direction == RoutingDirection::Z;
}

template <typename T>
bool HasRepeatedTarget(const std::vector<T>& targets) {
    return std::ranges::any_of(targets, [&](const T& target) { return std::ranges::count(targets, target) > 1; });
}

}  // namespace

void ValidateFabricConnectionsSpec(const experimental::FabricConnectionsSpec& connection) {
    const bool is_empty = std::visit([](const auto& targets) { return targets.empty(); }, connection.targets);
    TT_FATAL(!is_empty, "FabricConnectionsSpec '{}' has no targets", connection.unique_id);
    TT_FATAL(!connection.links.empty(), "FabricConnectionsSpec '{}' has no links", connection.unique_id);
    const bool has_repeated_target =
        std::visit([](const auto& targets) { return HasRepeatedTarget(targets); }, connection.targets);
    TT_FATAL(!has_repeated_target, "FabricConnectionsSpec '{}' has a target more than once", connection.unique_id);
    if (const auto* directions = std::get_if<experimental::Fabric1DTargets>(&connection.targets)) {
        for (const auto direction : *directions) {
            TT_FATAL(
                IsAllowedDirectionTarget(direction),
                "FabricConnectionsSpec '{}' has direction target {}; a direction target is N, E, S, W or Z",
                connection.unique_id,
                direction);
        }
    }
}

void ValidateFabricConnectionBindings(const KernelSpec& kernel, tt::ARCH arch) {
    if (kernel.advanced_options.fabric_connection_bindings.empty()) {
        return;
    }
    TT_FATAL(
        !is_gen2_arch(arch),
        "Kernel '{}' binds a FabricConnectionsSpec, but fabric connection bindings are not supported on 2nd-gen "
        "architectures yet",
        kernel.unique_id);
    TT_FATAL(
        kernel.is_data_movement_kernel(),
        "Kernel '{}' binds a FabricConnectionsSpec, but only data-movement kernels can bind a FabricConnectionsSpec",
        kernel.unique_id);

    std::unordered_set<std::string> accessor_names;
    std::unordered_set<experimental::FabricConnectionsSpecName> spec_names;
    for (const auto& binding : kernel.advanced_options.fabric_connection_bindings) {
        TT_FATAL(
            spec_names.insert(binding.fabric_connections_spec_name).second,
            "Kernel '{}' binds FabricConnectionsSpec '{}' more than once",
            kernel.unique_id,
            binding.fabric_connections_spec_name);
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate fabric connection accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' fabric connection accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "fabric connection", binding.accessor_name);
    }
}

void ValidateFabricConnectionsUsed(const ValidationContext& ctx) {
    for (const auto& connection : ctx.spec.advanced_options.fabric_connections) {
        const auto binders = ctx.collected.fabric_connection_binders.find(connection.unique_id);
        TT_FATAL(
            binders != ctx.collected.fabric_connection_binders.end(),
            "FabricConnectionsSpec '{}' is defined but not bound by any kernel",
            connection.unique_id);

        // On each node, at most one kernel instance binds the spec, and each node of a binding kernel is in links.
        NodeRangeSet binder_nodes;
        for (const KernelSpec* kernel : binders->second) {
            const NodeRangeSet& kernel_nodes = ctx.collected.kernel_node_set.at(kernel->unique_id);
            TT_FATAL(
                !binder_nodes.intersects(kernel_nodes),
                "FabricConnectionsSpec '{}' is bound by more than one kernel on a node (kernel '{}' is one of them)",
                connection.unique_id,
                kernel->unique_id);
            binder_nodes = binder_nodes.merge(kernel_nodes);
            const std::vector<NodeCoord> nodes = corerange_to_cores(kernel_nodes);
            const auto node_outside_links =
                std::ranges::find_if(nodes, [&](const NodeCoord& node) { return !connection.links.contains(node); });
            TT_FATAL(
                node_outside_links == nodes.end(),
                "Kernel '{}' binds FabricConnectionsSpec '{}', but node {} of the kernel is not in the spec's links. "
                "Hint: "
                "split the KernelSpec so that only nodes in links bind the spec, or do not bind the spec",
                kernel->unique_id,
                connection.unique_id,
                node_outside_links->str());
        }

        for (const auto& [node, link] : connection.links) {
            TT_FATAL(
                binder_nodes.contains(node),
                "FabricConnectionsSpec '{}': node {} is in links, but no kernel that binds the spec runs there",
                connection.unique_id,
                node.str());
        }
    }
}

}  // namespace tt::tt_metal

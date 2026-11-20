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

void ValidateFabricConnectionSpec(const FabricConnectionSpec& connection) {
    const bool is_empty = std::visit([](const auto& targets) { return targets.empty(); }, connection.targets);
    TT_FATAL(!is_empty, "FabricConnectionSpec '{}' has no targets", connection.unique_id);
    TT_FATAL(!connection.links.empty(), "FabricConnectionSpec '{}' has no links", connection.unique_id);
    const bool has_repeated_target =
        std::visit([](const auto& targets) { return HasRepeatedTarget(targets); }, connection.targets);
    TT_FATAL(!has_repeated_target, "FabricConnectionSpec '{}' has a target more than once", connection.unique_id);
    if (const auto* directions = std::get_if<Fabric1DTargets>(&connection.targets)) {
        for (const auto direction : *directions) {
            TT_FATAL(
                IsAllowedDirectionTarget(direction),
                "FabricConnectionSpec '{}' has direction target {}; a direction target is N, E, S, W or Z",
                connection.unique_id,
                direction);
        }
    }
}

void ValidateFabricConnectionBindings(const KernelSpec& kernel, tt::ARCH arch) {
    if (kernel.fabric_connection_bindings.empty()) {
        return;
    }
    TT_FATAL(
        !is_gen2_arch(arch),
        "Kernel '{}' binds a FabricConnectionSpec, but fabric connection bindings are not supported on 2nd-gen "
        "architectures yet",
        kernel.unique_id);
    TT_FATAL(
        kernel.is_data_movement_kernel(),
        "Kernel '{}' binds a FabricConnectionSpec, but only data-movement kernels can bind a FabricConnectionSpec",
        kernel.unique_id);

    std::unordered_set<std::string> accessor_names;
    std::unordered_set<FabricConnectionSpecName> spec_names;
    for (const auto& binding : kernel.fabric_connection_bindings) {
        TT_FATAL(
            spec_names.insert(binding.fabric_connection_spec_name).second,
            "Kernel '{}' binds FabricConnectionSpec '{}' more than once",
            kernel.unique_id,
            binding.fabric_connection_spec_name);
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
    for (const auto& connection : ctx.spec.fabric_connections) {
        const auto binders = ctx.collected.fabric_connection_binders.find(connection.unique_id);
        TT_FATAL(
            binders != ctx.collected.fabric_connection_binders.end(),
            "FabricConnectionSpec '{}' is defined but not bound by any kernel",
            connection.unique_id);

        // On each node, at most one kernel instance binds the spec.
        NodeRangeSet binder_nodes;
        for (const KernelSpec* kernel : binders->second) {
            const NodeRangeSet& kernel_nodes = ctx.collected.kernel_node_set.at(kernel->unique_id);
            TT_FATAL(
                !binder_nodes.intersects(kernel_nodes),
                "FabricConnectionSpec '{}' is bound by more than one kernel on a node (kernel '{}' is one of them)",
                connection.unique_id,
                kernel->unique_id);
            binder_nodes = binder_nodes.merge(kernel_nodes);
        }

        for (const auto& [node, link] : connection.links) {
            TT_FATAL(
                binder_nodes.contains(node),
                "FabricConnectionSpec '{}': node {} is in links, but no kernel that binds the spec runs there",
                connection.unique_id,
                node.str());
        }
    }
}

}  // namespace tt::tt_metal

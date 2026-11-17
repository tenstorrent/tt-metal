// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal {

void ValidateFabricConnectionSpec(const experimental::FabricConnectionSpec& connection) {
    for (const auto& [node, targets] : connection.targets) {
        TT_FATAL(
            targets.size() <= 1,
            "FabricConnectionSpec '{}' has {} targets on node {}; a node has at most one target",
            connection.unique_id,
            targets.size(),
            node.str());
    }
}

void ValidateFabricConnectionBindings(const KernelSpec& kernel) {
    const auto& bindings = kernel.advanced_options.fabric_connection_bindings;
    if (bindings.empty()) {
        return;
    }
    TT_FATAL(
        kernel.is_data_movement_kernel(),
        "Kernel '{}' is a compute kernel; a compute kernel cannot bind a fabric connection",
        kernel.unique_id);
    TT_FATAL(
        kernel.num_threads == 1,
        "Kernel '{}' binds a fabric connection, so it must have num_threads == 1 (it has {})",
        kernel.unique_id,
        kernel.num_threads);
    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : bindings) {
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

void ValidateFabricConnectionUsage(const ValidationContext& ctx) {
    const CollectedSpecData& collected = ctx.collected;
    using Link = std::tuple<tt::tt_fabric::FabricNodeId, tt::tt_fabric::FabricNodeId, uint32_t>;
    std::set<Link> used_links;
    for (const auto& connection : ctx.spec.advanced_options.fabric_connections) {
        const auto binders = collected.fabric_connection_binders.find(connection.unique_id);
        TT_FATAL(
            binders != collected.fabric_connection_binders.end(),
            "FabricConnectionSpec '{}' is defined but not bound by any kernel",
            connection.unique_id);

        // On each node, at most one kernel instance binds the connection.
        NodeRangeSet binder_nodes;
        for (const KernelSpec* kernel : binders->second) {
            const NodeRangeSet& kernel_nodes = collected.kernel_node_set.at(kernel->unique_id);
            for (const auto& range : kernel_nodes.ranges()) {
                for (const auto& node : range) {
                    TT_FATAL(
                        !binder_nodes.contains(node),
                        "FabricConnectionSpec '{}' is bound by more than one kernel on node {}",
                        connection.unique_id,
                        node.str());
                }
            }
            binder_nodes = binder_nodes.merge(kernel_nodes);
        }

        for (const auto& [node, targets] : connection.targets) {
            if (targets.empty()) {
                continue;
            }
            TT_FATAL(
                binder_nodes.contains(node),
                "FabricConnectionSpec '{}': node {} has a target, but no kernel that binds it runs there",
                connection.unique_id,
                node.str());
            for (const auto& target : targets) {
                const auto [it, inserted] =
                    used_links.emplace(connection.src_fabric_node_id, target.dst_fabric_node_id, target.link_idx);
                TT_FATAL(
                    inserted,
                    "FabricConnectionSpec '{}' uses (src {}, dst {}, link {}) more than once in the program; a "
                    "fabric router accepts one connected worker at a time",
                    connection.unique_id,
                    connection.src_fabric_node_id,
                    target.dst_fabric_node_id,
                    target.link_idx);
            }
        }
    }
}

}  // namespace tt::tt_metal

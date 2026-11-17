// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "impl/context/metal_env_impl.hpp"

namespace tt::tt_metal {

FabricConnectionArgsBySpec RegisterFabricConnections(
    distributed::MeshDevice& mesh_device, const ProgramSpec& spec, detail::ProgramImpl& program_impl) {
    FabricConnectionArgsBySpec args_by_spec;
    if (spec.advanced_options.fabric_connections.empty()) {
        return args_by_spec;
    }
    const auto& fabric_context = mesh_device.impl().metal_env().get_control_plane().get_fabric_context();
    for (const auto& connection : spec.advanced_options.fabric_connections) {
        FabricConnectionArgsByNode& args_by_node = args_by_spec[connection.unique_id];
        for (const auto& [node, targets] : connection.targets) {
            if (targets.empty()) {
                continue;
            }
            const auto& target = targets[0];
            const tt_fabric::WorkerFabricRouterChannel router = tt_fabric::get_worker_fabric_router_channel(
                fabric_context, connection.src_fabric_node_id, target.dst_fabric_node_id, target.link_idx);
            // The two L1 cells of the connection: the teardown semaphore and the router's buffer-index landing zone.
            const CoreRangeSet node_set(CoreRange(node, node));
            const uint32_t teardown_semaphore_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER);
            const uint32_t buffer_index_semaphore_id = program_impl.create_semaphore(node_set, 0, CoreType::WORKER);
            args_by_node[node] = {router.eth_channel, teardown_semaphore_id, buffer_index_semaphore_id};
        }
    }
    return args_by_spec;
}

FabricConnectionBindingsForKernel ResolveFabricConnectionBindingsForKernel(
    const KernelSpec& kernel, const FabricConnectionArgsBySpec& args_by_spec, size_t fabric_connection_base_rta_word) {
    const auto& bindings = kernel.advanced_options.fabric_connection_bindings;
    FabricConnectionBindingsForKernel out;
    out.handles.reserve(bindings.size());
    auto rta_word = static_cast<uint32_t>(fabric_connection_base_rta_word);
    for (const auto& binding : bindings) {
        out.handles.push_back(FabricConnectionBindingHandle{
            .accessor_name = binding.accessor_name,
            .rta_word = rta_word,
            .connection_args_by_node = args_by_spec.at(binding.fabric_connection_spec_name),
        });
        rta_word += FabricConnectionBindingHandle::kNumRtaWords;
    }
    out.section_words = static_cast<uint32_t>(bindings.size()) * FabricConnectionBindingHandle::kNumRtaWords;
    return out;
}

}  // namespace tt::tt_metal

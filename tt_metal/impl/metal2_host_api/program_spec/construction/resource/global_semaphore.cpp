// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <vector>

#include "impl/metal2_host_api/helpers.hpp"

namespace tt::tt_metal::experimental {

GlobalSemaphoreSlotMap ReserveGlobalSemaphoreSlots(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    detail::ProgramImpl& program_impl) {
    // One table slot per parameter, in declaration order. Every parameter is bound by some kernel
    // (ValidateGlobalSemaphoresUsed), so every slot has cores.
    GlobalSemaphoreSlotMap slots;
    for (const auto& parameter : spec.global_semaphore_parameters) {
        NodeRangeSet cores;
        for (const KernelSpec* kernel : collected.global_semaphore_binders.at(parameter.unique_id)) {
            cores = cores.merge(collected.kernel_node_set.at(kernel->unique_id));
        }
        slots[parameter.unique_id] = program_impl.register_global_semaphore_parameter(
            parameter.unique_id.get(),
            detail::ProgramImpl::GlobalSemaphoreParameterBinding{.device = &mesh_device, .cores = cores});
    }
    return slots;
}

std::vector<tt::tt_metal::GlobalSemaphoreBindingHandle> MakeGlobalSemaphoreBindingHandles(
    const KernelSpec& kernel_spec, const GlobalSemaphoreSlotMap& slots) {
    std::vector<tt::tt_metal::GlobalSemaphoreBindingHandle> handles;
    for (const auto& binding : kernel_spec.semaphore_bindings) {
        const GlobalSemaphoreParamName* name = global_semaphore_name(binding);
        if (name != nullptr) {
            handles.push_back({.accessor_name = binding.accessor_name, .slot = slots.at(*name)});
        }
    }
    return handles;
}

}  // namespace tt::tt_metal::experimental

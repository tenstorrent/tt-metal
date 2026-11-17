// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

NodeRangeSet GlobalSemaphoreBinderNodes(
    const experimental::GlobalSemaphoreParamName& parameter_name, const CollectedSpecData& collected) {
    NodeRangeSet nodes;
    const auto it = collected.global_semaphore_binders.find(parameter_name);
    if (it == collected.global_semaphore_binders.end()) {
        return nodes;
    }
    for (const KernelSpec* kernel : it->second) {
        nodes = nodes.merge(collected.kernel_node_set.at(kernel->unique_id));
    }
    return nodes;
}

GlobalSemaphoreBindingsForKernel ResolveGlobalSemaphoreBindingsForKernel(
    const KernelSpec& kernel, size_t global_semaphore_base_crta_word) {
    const auto& bindings = kernel.advanced_options.global_semaphore_bindings;
    GlobalSemaphoreBindingsForKernel out;
    out.handles.reserve(bindings.size());
    auto crta_word_index = static_cast<uint32_t>(global_semaphore_base_crta_word);
    for (const auto& binding : bindings) {
        out.handles.push_back(GlobalSemaphoreBindingHandle{
            .accessor_name = binding.accessor_name,
            .parameter_name = binding.global_semaphore_parameter_name.get(),
            .addr_crta_word = crta_word_index++,
        });
    }
    out.section_words = static_cast<uint32_t>(bindings.size());
    return out;
}

}  // namespace tt::tt_metal

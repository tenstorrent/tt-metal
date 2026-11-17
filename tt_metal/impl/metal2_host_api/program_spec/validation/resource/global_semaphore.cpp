// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal {

void ValidateGlobalSemaphoreParameter(const experimental::GlobalSemaphoreParameter& parameter) {
    TT_FATAL(
        parameter.memory_type == BufferType::L1 || parameter.memory_type == BufferType::L1_SMALL,
        "GlobalSemaphoreParameter '{}' has memory_type {}; it must be L1 or L1_SMALL",
        parameter.unique_id,
        parameter.memory_type);
}

void ValidateGlobalSemaphoreBindings(const KernelSpec& kernel) {
    const auto& bindings = kernel.advanced_options.global_semaphore_bindings;
    TT_FATAL(
        bindings.empty() || kernel.is_data_movement_kernel(),
        "Kernel '{}' is a compute kernel; a compute kernel cannot bind a GlobalSemaphore",
        kernel.unique_id);
    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate GlobalSemaphore accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' GlobalSemaphore accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "GlobalSemaphore", binding.accessor_name);
    }
}

void ValidateGlobalSemaphoreParametersUsed(const ValidationContext& ctx) {
    for (const auto& parameter : ctx.spec.advanced_options.global_semaphore_parameters) {
        TT_FATAL(
            ctx.collected.global_semaphore_binders.contains(parameter.unique_id),
            "GlobalSemaphore parameter '{}' is defined but not bound by any kernel",
            parameter.unique_id);
    }
}

}  // namespace tt::tt_metal

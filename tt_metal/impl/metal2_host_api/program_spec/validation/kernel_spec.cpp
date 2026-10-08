// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file contains validation of KernelSpec struct that's not covered witin
 * `../resource` or `../placement`.
 *
 * If you are looking for validation of bindings, etc, please check those folders first.
 */

#include <string>
#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Named RTA/CRTA schema and named CTAs
void ValidateKernelArguments(const KernelSpec& kernel) {
    // All three kinds share the args:: namespace — their names must be mutually unique.
    std::unordered_map<std::string, const char*> seen;  // name -> kind
    auto check_name = [&](const std::string& name, const char* kind) {
        TT_FATAL(
            IsValidCppIdentifier(name),
            "KernelSpec '{}' {} name '{}' is not a valid C++ identifier.",
            kernel.unique_id,
            kind,
            name);
        auto [it, inserted] = seen.try_emplace(name, kind);
        TT_FATAL(
            inserted,
            "KernelSpec '{}' has a naming collision: '{}' is declared as both a {} and a {}.",
            kernel.unique_id,
            name,
            it->second,
            kind);
    };
    for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
        check_name(name, "named RTA");
    }
    for (const auto& name : kernel.runtime_arg_schema.common_runtime_arg_names) {
        check_name(name, "named CRTA");
    }
    for (const auto& [name, value] : kernel.compile_time_args) {
        (void)value;
        check_name(name, "named CTA");
    }
}

void ValidateNumThreads(const KernelSpec& kernel, tt::ARCH arch) {
    TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
    if (kernel.is_compute_kernel()) {
        if (is_gen2_arch(arch)) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_TENSIX_ENGINES_PER_NODE,
                "KernelSpec '{}' has too many threads. The architecture supports up to {} for compute kernels.",
                kernel.unique_id,
                QUASAR_TENSIX_ENGINES_PER_NODE);
            // On Quasar, we're not allowing 3-thread compute kernels.
            TT_FATAL(
                kernel.num_threads != 3,
                "KernelSpec '{}' has 3 threads, which is not supported for compute kernels. Legal values are 1, 2, "
                "and 4.",
                kernel.unique_id);
        } else {
            TT_FATAL(
                kernel.num_threads == 1,
                "KernelSpec '{}' specifies {} compute threads, but the target architecture does not support "
                "multi-threaded kernels.",
                kernel.unique_id,
                kernel.num_threads);
        }
    }
    if (kernel.is_data_movement_kernel()) {
        if (is_gen2_arch(arch)) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_USER_DM_CORES_PER_NODE,
                "KernelSpec '{}' has too many data movement threads. The maximum is {}.",
                kernel.unique_id,
                QUASAR_USER_DM_CORES_PER_NODE);
        } else {
            TT_FATAL(
                kernel.num_threads == 1,
                "KernelSpec '{}' specifies {} DM threads, but the target architecture does not support "
                "multi-threaded kernels. "
                "num_threads must be 1.",
                kernel.unique_id,
                kernel.num_threads);
        }
    }
}

}  // namespace

void ValidateKernelSpec(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch) {
    ValidateResourceBindings(kernel, ctx, arch);
    ValidateKernelArguments(kernel);
    ValidateNumThreads(kernel, arch);
    ValidateKernelHardwareConfig(kernel, ctx, arch);
}

}  // namespace tt::tt_metal::experimental

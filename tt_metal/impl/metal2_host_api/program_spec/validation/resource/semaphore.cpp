// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <string_view>
#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateSemaphoreSpec(const SemaphoreSpec& sem, tt::ARCH arch) {
    const uint32_t init_value = sem.advanced_options.initial_value;
    if (is_gen2_arch(arch)) {
        TT_FATAL(
            init_value == 0,
            "SemaphoreSpec '{}' has initial_value={} but only zero is supported on Quasar",
            sem.unique_id,
            init_value);
    }
}

void ValidateSemaphoreBindings(const KernelSpec& kernel, tt::ARCH arch) {
    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : kernel.semaphore_bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate semaphore accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' semaphore accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "semaphore", binding.accessor_name);
    }

    // Blackhole supports local semaphore bindings on UNPACK and PACK (SemScope::COMPUTE_ATOMIC).
    // Wormhole has no compute implementation and Quasar compute remains out of scope.
    TT_FATAL(
        !kernel.is_compute_kernel() || kernel.semaphore_bindings.empty() || arch == tt::ARCH::BLACKHOLE,
        "KernelSpec '{}' has semaphore bindings. "
        "Semaphore bindings on compute kernels are supported only on Blackhole.",
        kernel.unique_id);
}

// A compute semaphore is an UNPACK <-> PACK mechanism (the Tensix hardware semaphore, driven by
// Tensix instructions a DM core cannot issue) and may not be shared with a DM kernel. Reject it
// here rather than resolve a scope that cannot serve both.
void ValidateComputeSemaphores(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;

    std::unordered_set<std::string_view> sem_has_compute;
    std::unordered_set<std::string_view> sem_has_dm;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.semaphore_bindings) {
            (kernel.is_compute_kernel() ? sem_has_compute : sem_has_dm).insert(*binding.semaphore_spec_name);
        }
    }
    for (const auto& name : sem_has_compute) {
        TT_FATAL(
            !sem_has_dm.contains(name),
            "SemaphoreSpec '{}' is bound by both a compute kernel and a data-movement kernel. "
            "Compute semaphores synchronize UNPACK and PACK with each other and cannot be shared "
            "with a DM kernel; use separate semaphores for the compute and data-movement handoffs.",
            name);
    }
    // Every compute semaphore maps onto the single free Tensix hardware semaphore (index 3), so two in
    // one program would alias the same hardware state.
    TT_FATAL(
        sem_has_compute.size() <= 1,
        "{} semaphores are bound by compute kernels; a program may bind at most one compute semaphore "
        "(Blackhole has a single free Tensix hardware semaphore).",
        sem_has_compute.size());
    // The compute semaphore lives in the Tensix Sync Unit, which the host cannot write; it is seeded
    // to 0 by compute_kernel_hw_startup() on the device, so no other initial value can be honored.
    // Its capacity (max_value) is a 4-bit hardware field and has no meaning for a DM semaphore.
    for (const auto& sem : spec.semaphores) {
        const bool compute_bound = sem_has_compute.contains(std::string_view{*sem.unique_id});
        const uint32_t init_value = sem.advanced_options.initial_value;
        const uint32_t max_value = sem.advanced_options.max_value;
        TT_FATAL(
            !compute_bound || init_value == 0,
            "SemaphoreSpec '{}' is bound by a compute kernel but has initial_value={}. Compute "
            "semaphores always start at 0 (seeded by compute_kernel_hw_startup on the device).",
            sem.unique_id,
            init_value);
        TT_FATAL(
            !compute_bound || max_value <= 15,
            "SemaphoreSpec '{}' has max_value={}; a compute semaphore's capacity is at most 15 (4-bit "
            "Tensix hardware semaphore).",
            sem.unique_id,
            max_value);
        TT_FATAL(
            compute_bound || max_value == 0,
            "SemaphoreSpec '{}' has max_value={} but is not bound by a compute kernel; max_value is the "
            "capacity of a compute semaphore and has no effect on a data-movement semaphore.",
            sem.unique_id,
            max_value);
    }
}

}  // namespace tt::tt_metal::experimental

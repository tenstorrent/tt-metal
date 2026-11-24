// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal {

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

    // Blackhole and Quasar support local semaphore bindings on UNPACK and PACK (SemScope::COMPUTE_ATOMIC).
    // Wormhole has no compute implementation.
    TT_FATAL(
        !kernel.is_compute_kernel() || kernel.semaphore_bindings.empty() || has_compute_semaphore(arch),
        "KernelSpec '{}' has semaphore bindings. "
        "Semaphore bindings on compute kernels are supported only on Blackhole and Quasar.",
        kernel.unique_id);
}

// A compute-only semaphore is an UNPACK <-> PACK mechanism (the Tensix hardware semaphore, driven by
// Tensix instructions a DM core cannot issue). A semaphore bound by both a DM and a compute kernel is
// instead its plain L1 word (SemScope::LOCAL_NONATOMIC): Quasar only, on one node, with exactly one DM
// binder instance as its only writer; compute only reads it (wait / wait_min / value).
void ValidateComputeSemaphores(const ValidationContext& ctx, tt::ARCH arch) {
    const ProgramSpec& spec = ctx.spec;

    std::unordered_set<std::string_view> sem_has_compute;
    std::unordered_set<std::string_view> sem_has_dm;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.semaphore_bindings) {
            (kernel.is_compute_kernel() ? sem_has_compute : sem_has_dm).insert(*binding.semaphore_spec_name);
        }
    }
    std::unordered_set<std::string_view> sem_compute_only;
    for (const auto& name : sem_has_compute) {
        if (!sem_has_dm.contains(name)) {
            sem_compute_only.insert(name);
            continue;
        }
        TT_FATAL(
            is_gen2_arch(arch),
            "SemaphoreSpec '{}' is bound by both a compute kernel and a data-movement kernel. "
            "Compute semaphores synchronize UNPACK and PACK with each other and cannot be shared "
            "with a DM kernel on this architecture; use separate semaphores for the compute and "
            "data-movement handoffs.",
            name);
        const auto& binders = ctx.collected.semaphore_binders.at(SemaphoreSpecName{std::string{name}});
        uint32_t dm_instances = 0;
        for (const auto& rec : binders.binders) {
            if (!rec.kernel->is_compute_kernel()) {
                dm_instances +=
                    ctx.collected.kernel_node_set.at(rec.kernel->unique_id).num_cores() * rec.kernel->num_threads;
            }
        }
        const auto sem = std::find_if(spec.semaphores.begin(), spec.semaphores.end(), [&](const SemaphoreSpec& s) {
            return std::string_view{*s.unique_id} == name;
        });
        TT_FATAL(
            sem != spec.semaphores.end() && to_node_range_set(sem->target_nodes).num_cores() == 1 && dm_instances == 1,
            "SemaphoreSpec '{}' is shared by a compute kernel and data-movement kernels with {} instance(s). "
            "A DM <-> compute semaphore lives on one node and has exactly one DM binder instance (its only "
            "writer); compute only waits on it.",
            name,
            dm_instances);
    }
    // Every compute semaphore maps onto the one Tensix hardware semaphore reserved for it, so two in
    // one program would alias the same hardware state.
    TT_FATAL(
        sem_compute_only.size() <= 1,
        "{} semaphores are bound by compute kernels only; a program may bind at most one compute semaphore "
        "(one Tensix hardware semaphore is reserved for compute).",
        sem_compute_only.size());
    // The compute semaphore lives in the Tensix Sync Unit, which the host cannot write; it is seeded
    // to 0 by compute_kernel_hw_startup() on the device, so no other initial value can be honored.
    // Its capacity (max_value) is a 4-bit hardware field and has no meaning for a DM semaphore.
    for (const auto& sem : spec.semaphores) {
        const bool compute_bound = sem_compute_only.contains(std::string_view{*sem.unique_id});
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

void ValidateGlobalSemaphoreBindings(const KernelSpec& kernel) {
    if (kernel.global_semaphore_bindings.empty()) {
        return;
    }
    TT_FATAL(
        kernel.is_data_movement_kernel(),
        "Kernel '{}' binds a GlobalSemaphore, but only data-movement kernels can bind a GlobalSemaphore",
        kernel.unique_id);

    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : kernel.global_semaphore_bindings) {
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
    const ProgramSpec& spec = ctx.spec;
    std::unordered_set<GlobalSemaphoreParamName> used;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.global_semaphore_bindings) {
            used.insert(binding.global_semaphore_parameter_name);
        }
    }
    for (const auto& parameter : spec.global_semaphore_parameters) {
        TT_FATAL(
            used.contains(parameter.unique_id),
            "GlobalSemaphoreParameter '{}' is defined but not bound by any kernel",
            parameter.unique_id);
    }
}

}  // namespace tt::tt_metal

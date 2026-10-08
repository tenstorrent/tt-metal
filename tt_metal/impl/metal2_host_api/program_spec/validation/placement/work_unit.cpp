// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string_view>

#include <tt_stl/assert.hpp>

#include "core_descriptor.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

namespace {

// ----------------------------------------------------------------------------
// ValidateNodeBounds: Node coordinate bounds checking
// ----------------------------------------------------------------------------
//
// Validates that every NodeCoord referenced by a WorkUnitSpec or SemaphoreSpec
// is within the compute worker grid on this device.
// (Kernel and DFB placement is derived from WorkUnitSpec membership, so
// bounds-checking the WorkUnitSpecs covers them too.)
//
// NOTE: We're dealing in logical coordinates. (Harvesting is handled by UMD.)
//
// ASSUMPTION: All chips in a MeshDevice are identical, so chip 0 is
// representative of every device in the mesh.

void ValidateNodeBounds(const ProgramSpec& spec, const CoreCoord& compute_grid_size) {
    auto check_target_nodes =
        [&](const Nodes& target_nodes, std::string_view entity_type, std::string_view entity_name) {
            const NodeRangeSet range_set = to_node_range_set(target_nodes);
            for (const NodeRange& range : range_set.ranges()) {
                for (const NodeCoord& node : range) {
                    TT_FATAL(
                        node.x < compute_grid_size.x && node.y < compute_grid_size.y,
                        "{} '{}' targets node ({},{}), which is out of bounds. "
                        "The compute worker grid on this device is {}x{}.",
                        entity_type,
                        entity_name,
                        node.x,
                        node.y,
                        compute_grid_size.x,
                        compute_grid_size.y);
                }
            }
        };

    for (const auto& work_unit : spec.work_units) {
        check_target_nodes(work_unit.target_nodes, "WorkUnitSpec", work_unit.name);
    }
    for (const auto& sem : spec.semaphores) {
        check_target_nodes(sem.target_nodes, "SemaphoreSpec", sem.unique_id.get());
    }
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        check_target_nodes(pipe.receivers, "PrefetcherPipeParameter", pipe.unique_id.get());
    }
}

// Does the WorkUnit have enough cores to run all of its kernels?
void ValidateWorkUnitCapacity(const WorkUnitSpec& work_unit, const CollectedSpecData& collected, tt::ARCH arch) {
    uint32_t dm_cores_needed = 0;
    uint32_t compute_engines_needed = 0;
    for (const auto& kernel_name : work_unit.kernels) {
        const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
        if (kernel_spec->is_compute_kernel()) {
            compute_engines_needed += kernel_spec->num_threads;
        }
        if (kernel_spec->is_data_movement_kernel()) {
            dm_cores_needed += kernel_spec->num_threads;
        }
    }
    if (is_gen2_arch(arch)) {
        TT_FATAL(
            compute_engines_needed <= QUASAR_TENSIX_ENGINES_PER_NODE,
            "WorkUnitSpec '{}' needs {} Tensix engines, but only {} are available",
            work_unit.name,
            compute_engines_needed,
            QUASAR_TENSIX_ENGINES_PER_NODE);
        TT_FATAL(
            dm_cores_needed <= QUASAR_USER_DM_CORES_PER_NODE,
            "WorkUnitSpec '{}' requests {} data movement cores. This exceeds the permitted maximum of {}.",
            work_unit.name,
            dm_cores_needed,
            QUASAR_USER_DM_CORES_PER_NODE);
    }
    if (is_gen1_arch(arch)) {
        TT_FATAL(
            compute_engines_needed <= 1,
            "WorkUnitSpec '{}' has {} compute kernels. The target architecture supports at most one.",
            work_unit.name,
            compute_engines_needed);
        TT_FATAL(
            dm_cores_needed <= 2,
            "WorkUnitSpec '{}' has {} data movement kernels. The target architecture supports at most two.",
            work_unit.name,
            dm_cores_needed);
    }
}

}  // namespace

void ValidateWorkUnitFields(const ValidationContext& ctx, const CoreCoord& compute_grid_size) {
    const ProgramSpec& spec = ctx.spec;

    ValidateNodeBounds(spec, compute_grid_size);

    // WorkUnitSpec is required: a valid ProgramSpec has at least one WorkUnitSpec.
    const auto& work_units = spec.work_units;
    TT_FATAL(!work_units.empty(), "At least one WorkUnitSpec is required");

    // WorkUnitSpecs may not overlap in their target nodes
    for (const auto& work_unit : work_units) {
        for (const auto& other_work_unit : work_units) {
            if (work_unit.name == other_work_unit.name) {
                continue;
            }
            if (nodes_intersect(work_unit.target_nodes, other_work_unit.target_nodes)) {
                TT_FATAL(
                    false, "WorkUnitSpecs '{}' and '{}' overlap in target nodes", work_unit.name, other_work_unit.name);
            }
        }
    }
}

void ValidateWorkUnitSpec(const WorkUnitSpec& work_unit, const ValidationContext& ctx, tt::ARCH arch) {
    const CollectedSpecData& collected = ctx.collected;

    // A WorkUnitSpec must have at least one kernel
    TT_FATAL(!work_unit.kernels.empty(), "WorkUnitSpec '{}' has no kernels", work_unit.name);

    ValidateWorkUnitCapacity(work_unit, collected, arch);

    // A work_unit can have at most one compute kernel
    uint32_t num_compute_kernels = 0;
    for (const auto& kernel_name : work_unit.kernels) {
        const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
        if (kernel_spec->is_compute_kernel()) {
            num_compute_kernels++;
        }
    }
    TT_FATAL(num_compute_kernels <= 1, "WorkUnitSpec '{}' has more than one compute kernel", work_unit.name);
}

}  // namespace tt::tt_metal::experimental

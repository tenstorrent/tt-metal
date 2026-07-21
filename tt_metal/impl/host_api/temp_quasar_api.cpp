// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_api/temp_quasar_api.hpp"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <enchantum/entries.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "impl/context/metal_context.hpp"
#include "host_api/helpers.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::quasar {

std::set<DataMovementProcessor> GetDataMovementProcessorsInUseOnKernelGroup(
    Program& program, const KernelGroup* kernel_group) {
    if (kernel_group == nullptr) {
        return {};
    }
    std::set<DataMovementProcessor> processors_in_use;
    for (const KernelHandle kernel_id : kernel_group->kernel_ids) {
        const std::shared_ptr<Kernel> kernel = program.impl().get_kernel(kernel_id);
        if (kernel->get_kernel_processor_class() == HalProcessorClassType::DM) {
            const std::shared_ptr<QuasarDataMovementKernel> dm_kernel =
                std::dynamic_pointer_cast<QuasarDataMovementKernel>(kernel);
            TT_ASSERT(dm_kernel != nullptr);
            const std::vector<DataMovementProcessor> dm_processors = dm_kernel->get_dm_processors();
            processors_in_use.insert(dm_processors.begin(), dm_processors.end());
        }
    }
    return processors_in_use;
}

bool DoesKernelGroupHaveComputeKernel(Program& program, const KernelGroup* kernel_group) {
    if (kernel_group == nullptr) {
        return false;
    }
    for (const KernelHandle kernel_id : kernel_group->kernel_ids) {
        const std::shared_ptr<Kernel> kernel = program.impl().get_kernel(kernel_id);
        if (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
            return true;
        }
    }
    return false;
}

template <typename ProcessorClassType>
std::set<ProcessorClassType> GetProcessorsPerClusterQuasar(
    Program& program, const CoreRangeSet& core_ranges, uint32_t num_processors_per_cluster) {
    std::set<ProcessorClassType> processors(
        enchantum::values<ProcessorClassType>.begin(), enchantum::values<ProcessorClassType>.end());
    std::vector<std::set<DataMovementProcessor>> dm_processors_in_use_per_kernel_group;

    std::unordered_set<const KernelGroup*> kernel_groups;
    for (const CoreRange& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup* kernel_group = program.impl().kernels_on_core(
                    CoreCoord(x, y),
                    MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX));
                kernel_groups.insert(kernel_group);
            }
        }
    }

    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (const KernelGroup* kernel_group : kernel_groups) {
        if constexpr (std::is_same_v<ProcessorClassType, DataMovementProcessor>) {
            const std::set<DataMovementProcessor> dm_processors_in_use_on_kernel_group =
                GetDataMovementProcessorsInUseOnKernelGroup(program, kernel_group);
            dm_processors_in_use_per_kernel_group.push_back(dm_processors_in_use_on_kernel_group);
            for (const DataMovementProcessor dm_processor : dm_processors_in_use_on_kernel_group) {
                processors.erase(dm_processor);
            }
        } else if constexpr (std::is_same_v<ProcessorClassType, QuasarComputeProcessor>) {
            TT_FATAL(
                !DoesKernelGroupHaveComputeKernel(program, kernel_group),
                "In Quasar, each cluster can only have a single compute kernel.");
        }
    }

    for (uint32_t i = 1; i < dm_processors_in_use_per_kernel_group.size(); i++) {
        TT_FATAL(
            dm_processors_in_use_per_kernel_group[i] == dm_processors_in_use_per_kernel_group[i - 1],
            "All clusters in {} must have the same data movement processors already in use to reserve {} new data "
            "movement processors per cluster.",
            core_ranges,
            num_processors_per_cluster);
    }

    while (processors.size() > num_processors_per_cluster) {
        processors.erase(std::prev(processors.end()));
    }

    if constexpr (std::is_same_v<ProcessorClassType, DataMovementProcessor>) {
        TT_FATAL(
            processors.size() == num_processors_per_cluster,
            "Unable to reserve {} data movement processors per cluster as only {} data movement processors per cluster "
            "are available.",
            num_processors_per_cluster,
            processors.size());
    } else if constexpr (std::is_same_v<ProcessorClassType, QuasarComputeProcessor>) {
        TT_FATAL(
            processors.size() % QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE == 0,
            "Number of compute processors reserved per cluster must be a multiple of {}.",
            QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE);
        TT_FATAL(
            processors.size() == num_processors_per_cluster,
            "Unable to reserve {} compute processors per cluster as only {} compute processors per cluster are "
            "available.",
            num_processors_per_cluster,
            processors.size());
    }

    return processors;
}

KernelHandle CreateQuasarDataMovementKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_ranges,
    const QuasarDataMovementConfig& config) {
    TT_FATAL(
        1 <= config.num_threads_per_cluster && config.num_threads_per_cluster <= QUASAR_NUM_DM_CORES_PER_CLUSTER,
        "Requested number of data movement cores per cluster must be between 1 and {} (inclusive)",
        QUASAR_NUM_DM_CORES_PER_CLUSTER);
    const std::set<DataMovementProcessor> dm_processors =
        GetProcessorsPerClusterQuasar<DataMovementProcessor>(program, core_ranges, config.num_threads_per_cluster);
    std::shared_ptr<Kernel> kernel =
        std::make_shared<QuasarDataMovementKernel>(kernel_src, core_ranges, config, dm_processors);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const QuasarDataMovementConfig& config) {
    const CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return CreateQuasarDataMovementKernel(
        program, KernelSource(file_name, KernelSource::FILE_PATH), core_ranges, config);
}

KernelHandle CreateQuasarComputeKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_ranges,
    const QuasarComputeConfig& config) {
    TT_FATAL(
        1 <= config.num_threads_per_cluster && config.num_threads_per_cluster <= QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER,
        "Requested number of Tensix engines per cluster must be between 1 and {} (inclusive)",
        QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER);
    const std::set<QuasarComputeProcessor> compute_processors = GetProcessorsPerClusterQuasar<QuasarComputeProcessor>(
        program, core_ranges, config.num_threads_per_cluster * QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE);
    std::shared_ptr<Kernel> kernel =
        std::make_shared<QuasarComputeKernel>(kernel_src, core_ranges, config, compute_processors);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const QuasarComputeConfig& config) {
    const CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return CreateQuasarComputeKernel(program, KernelSource(file_name, KernelSource::FILE_PATH), core_ranges, config);
}

}  // namespace tt::tt_metal::experimental::quasar

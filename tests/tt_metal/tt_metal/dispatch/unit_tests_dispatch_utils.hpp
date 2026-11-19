// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "llrt/hal.hpp"

namespace tt::tt_metal {

inline Program make_single_kernel_program(
    distributed::MeshDevice& mesh_device,
    const std::string& name,
    const std::string& kernel_path,
    const CoreRangeSet& core_range_set,
    HalProcessorIdentifier processor,
    KernelSpec::CompileTimeArgs compile_time_args = {}) {
    using namespace tt::tt_metal::experimental;

    std::variant<DataMovementHardwareConfig, ComputeHardwareConfig> hardware_config;
    if (processor.processor_class == HalProcessorClassType::COMPUTE) {
        hardware_config = ComputeHardwareConfig{};
    } else {
        hardware_config = DataMovementHardwareConfig{
            .config_1xx =
                DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = static_cast<DataMovementProcessor>(processor.processor_type),
                    .noc = static_cast<NOC>(processor.processor_type),
                },
        };
    }

    const KernelSpec kernel_spec{
        .unique_id = KernelSpecName{name},
        .source = kernel_path,
        .num_threads = 1,
        .compile_time_args = std::move(compile_time_args),
        .hw_config = hardware_config,
    };

    const ProgramSpec program_spec{
        .name = name,
        .kernels = {kernel_spec},
        .work_units = {WorkUnitSpec{
            .name = "work_unit",
            .kernels = {kernel_spec.unique_id},
            .target_nodes = core_range_set,
        }},
    };

    return Program(mesh_device, program_spec);
}

inline bool has_at_least_worker_cores(
    const distributed::MeshDevice& mesh_device, std::size_t required_worker_core_count) {
    const CoreCoord worker_grid_size = mesh_device.compute_with_storage_grid_size();
    return worker_grid_size.x * worker_grid_size.y >= required_worker_core_count;
}

}  // namespace tt::tt_metal

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt_stl/small_vector.hpp>

#include <umd/device/types/core_coordinates.hpp>

#include <optional>
#include <unordered_map>

namespace tt::tt_metal::experimental {

// Describes a GlobalSemaphore to be allocated at mesh level
struct GlobalSemaphoreDescriptor {
    CoreRangeSet cores;
    uint32_t initial_value = 0;
    tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::L1;
};

// Per-device runtime arg override
struct MeshRuntimeArgsDescriptor {
    // Which kernel this applies to (index into ProgramDescriptor::kernels)
    uint32_t kernel_index;

    // Per-coordinate runtime args: maps mesh coordinate → per-core runtime args
    // If a coordinate is not present, uses the base ProgramDescriptor's runtime_args
    // RuntimeArgs is a vector of pairs: (CoreCoord, CoreRuntimeArgs)
    using CoordinateRuntimeArgs = std::unordered_map<distributed::MeshCoordinate, KernelDescriptor::RuntimeArgs>;
    CoordinateRuntimeArgs coordinate_args;
};

struct MeshProgramDescriptor {
    std::unordered_map<distributed::MeshCoordinateRange, ProgramDescriptor> mesh_programs;
    std::vector<MeshRuntimeArgsDescriptor> mesh_runtime_args = {{}};

    // GlobalSemaphores to allocate at mesh level
    // Referenced via GlobalSemRef{index} in RuntimeArgsBuilder
    // ttsl::SmallVector<GlobalSemaphoreDescriptor, 3> global_semaphores;

    //------------------------------------------------------------------
    // Topology/Configuration
    //------------------------------------------------------------------
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    // Custom reflection attributes
    static constexpr auto attribute_names =
        std::forward_as_tuple("num_mesh_programs", "num_runtime_args", "sub_device_id");
    auto attribute_values() const {
        return std::forward_as_tuple(mesh_programs.size(), mesh_runtime_args.size(), sub_device_id);
    }
};

}  // namespace tt::tt_metal::experimental

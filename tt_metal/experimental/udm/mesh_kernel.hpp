// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <set>
#include "tt_metal/experimental/udm/types.hpp"
#include "tt_metal/experimental/udm/mesh_program.hpp"
#include "tt_metal/api/tt-metalium/kernel_types.hpp"

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Create a mesh kernel that operates across multiple grids
 *
 * @param builder The MeshBuilder containing mesh topology information
 * @param program The MeshProgram to add the kernel to
 * @param file_name Path to the kernel source file
 * @param gcores The global cores to place the kernel on
 * @param config Kernel configuration (DataMovementConfig, ComputeConfig, or EthernetConfig)
 * @return MeshKernelHandle Map of grid_id to kernel handle for each grid
 */
MeshKernelHandle CreateMeshKernel(
    MeshBuilder& builder,
    MeshProgram& program,
    const std::string& file_name,
    const std::vector<GlobalCore>& gcores,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig>&
        config);

/**
 * @brief Set runtime arguments for a specific gcore
 *
 * @param builder The MeshBuilder containing mesh topology information
 * @param program The MeshProgram containing the kernel
 * @param mesh_kernel_handle The mesh kernel handle (map of grid_id to kernel handle)
 * @param gcore The global core to set arguments for
 * @param runtime_args The runtime arguments
 */
void SetMeshKernelRuntimeArgs(
    MeshBuilder& builder,
    MeshProgram& program,
    const MeshKernelHandle& mesh_kernel_handle,
    const GlobalCore& gcore,
    const std::vector<uint32_t>& runtime_args);

}  // namespace tt::tt_metal::experimental::udm

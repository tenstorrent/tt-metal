// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::udm {

MeshKernelHandle CreateMeshKernel(
    MeshBuilder& builder,
    MeshProgram& program,
    const std::string& file_name,
    const std::vector<GlobalCore>& gcores,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig>&
        config) {
    // Get mesh compile-time defines and append to config
    auto mesh_defines = builder.get_compile_time_defines();

    // Create a modified config with mesh defines appended
    auto config_with_defines = std::visit(
        [&mesh_defines](auto cfg)
            -> std::
                variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig> {
                    cfg.defines.insert(mesh_defines.begin(), mesh_defines.end());
                    return cfg;
                },
        config);

    // Get grid IDs and local CoreRangeSets from gcores
    auto grid_to_cores = builder.get_grid_core_range_set_from_gcores(gcores);

    MeshKernelHandle mesh_kernel_handle;

    // Create kernel on each grid
    for (const auto& [grid_id, core_range_set] : grid_to_cores) {
        // Get the grid object to access its mesh coordinate
        const auto& all_grids = builder.get_all_grids_in_mesh();
        const Grid* grid_ptr = nullptr;
        for (const auto& g : all_grids) {
            if (g.id == grid_id) {
                grid_ptr = &g;
                break;
            }
        }
        TT_FATAL(grid_ptr != nullptr, "Grid {} not found", grid_id);

        auto& grid_program = program.program_at(grid_ptr->coord);

        auto handle = tt::tt_metal::CreateKernel(grid_program, file_name, core_range_set, config_with_defines);

        // Store the kernel handle for this grid
        mesh_kernel_handle[grid_id] = handle;

        // Register that this mesh coordinate now has a kernel
        program.register_kernel(grid_ptr->coord);
    }

    return mesh_kernel_handle;
}

void SetMeshKernelRuntimeArgs(
    MeshBuilder& builder,
    MeshProgram& program,
    const MeshKernelHandle& mesh_kernel_handle,
    const GlobalCore& gcore,
    const std::vector<uint32_t>& runtime_args) {
    // Find which grid this gcore belongs to
    const auto& all_grids = builder.get_all_grids_in_mesh();

    for (const auto& grid : all_grids) {
        const auto& grid_gcores = builder.get_all_gcores_in_grid(grid);

        // Check if this gcore is in this grid
        auto it = std::find_if(grid_gcores.begin(), grid_gcores.end(), [&gcore](const GlobalCore& gc) {
            return gc.global_id == gcore.global_id;
        });

        if (it != grid_gcores.end()) {
            // Found the grid, get the kernel handle for this grid
            auto handle_it = mesh_kernel_handle.find(grid.id);
            TT_FATAL(handle_it != mesh_kernel_handle.end(), "Kernel handle not found for grid {}", grid.id);

            auto& grid_program = program.program_at(grid.coord);

            // Convert gcore to CoreCoord using safe helper method
            tt::tt_metal::CoreCoord core_coord = gcore.to_core_coord();

            tt::tt_metal::SetRuntimeArgs(grid_program, handle_it->second, core_coord, runtime_args);
            return;
        }
    }

    TT_FATAL(false, "GlobalCore with global_id {} not found in any grid", gcore.global_id);
}

}  // namespace tt::tt_metal::experimental::udm

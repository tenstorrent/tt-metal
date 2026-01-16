// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"
#include <tt_stl/assert.hpp>
#include <umd/device/types/arch.hpp>

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

    // Check for Wormhole 2-RISC UDM restriction before creating any kernels
    auto arch = tt::tt_metal::hal::get_arch();
    bool is_wormhole = (arch == tt::ARCH::WORMHOLE_B0);
    bool is_dm_kernel = std::holds_alternative<tt::tt_metal::DataMovementConfig>(config);

    if (is_wormhole && is_dm_kernel) {
        // On Wormhole, check if any global cores already have DM kernels
        for (const auto& gcore : gcores) {
            if (program.has_dm_kernel_on_gcore(gcore.global_id)) {
                TT_THROW(
                    "2-RISC UDM mode is not supported on Wormhole architecture. "
                    "Global core {} (local coord {}) already has a data movement kernel. "
                    "Cannot add another data movement kernel to the same core on Wormhole. "
                    "Please use a single RISC (RISCV_0 or RISCV_1 only) per core for data movement kernels on "
                    "Wormhole.",
                    gcore.global_id,
                    gcore.local_coord);
            }
        }
    }

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

    // If this is a DM kernel, register all global cores that now have DM kernels
    if (is_dm_kernel) {
        for (const auto& gcore : gcores) {
            program.register_dm_kernel_on_gcore(gcore.global_id);
        }
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

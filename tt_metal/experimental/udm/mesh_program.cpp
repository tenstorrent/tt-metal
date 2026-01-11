// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_program.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::udm {

class MeshProgram::Impl {
public:
    explicit Impl(const MeshBuilder& builder) {
        // Get all grids from MeshBuilder and build mapping: mesh_coord -> program
        const auto& grids = builder.get_all_grids_in_mesh();

        // Create a program for each grid, keyed by mesh coordinate
        // Initialize all coords to has_kernel = false
        for (const auto& grid : grids) {
            programs_.emplace(grid.coord, tt::tt_metal::Program());
            has_kernel_.emplace(grid.coord, false);
        }
    }

    tt::tt_metal::Program& program_at(const tt::tt_metal::distributed::MeshCoordinate& coord) {
        auto it = programs_.find(coord);
        TT_FATAL(it != programs_.end(), "Program for mesh coordinate {} not found", coord);
        return it->second;
    }

    const tt::tt_metal::Program& program_at(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
        auto it = programs_.find(coord);
        TT_FATAL(it != programs_.end(), "Program for mesh coordinate {} not found", coord);
        return it->second;
    }

    void register_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord) {
        auto it = has_kernel_.find(coord);
        TT_FATAL(it != has_kernel_.end(), "Mesh coordinate {} not found in has_kernel tracker", coord);
        it->second = true;
    }

    bool has_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
        auto it = has_kernel_.find(coord);
        TT_FATAL(it != has_kernel_.end(), "Mesh coordinate {} not found in has_kernel tracker", coord);
        return it->second;
    }

private:
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::Program> programs_;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinate, bool> has_kernel_;
};

MeshProgram::MeshProgram(const MeshBuilder& builder) : impl_(std::make_unique<Impl>(builder)) {}

MeshProgram::~MeshProgram() = default;

MeshProgram::MeshProgram(MeshProgram&&) noexcept = default;
MeshProgram& MeshProgram::operator=(MeshProgram&&) noexcept = default;

tt::tt_metal::Program& MeshProgram::program_at(const tt::tt_metal::distributed::MeshCoordinate& coord) {
    return impl_->program_at(coord);
}

const tt::tt_metal::Program& MeshProgram::program_at(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    return impl_->program_at(coord);
}

void MeshProgram::register_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord) {
    impl_->register_kernel(coord);
}

bool MeshProgram::has_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    return impl_->has_kernel(coord);
}

MeshProgram CreateMeshProgram(const MeshBuilder& builder) { return MeshProgram(builder); }

}  // namespace tt::tt_metal::experimental::udm

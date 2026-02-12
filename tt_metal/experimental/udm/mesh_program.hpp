// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/experimental/udm/types.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Multi-device program abstraction
 *
 * MeshProgram represents a program that operates across multiple devices
 * in a virtualized Mesh configuration.
 */
class MeshProgram {
public:
    /**
     * @brief Create a MeshProgram from a MeshBuilder
     *
     * @param builder The MeshBuilder containing grid information
     */
    explicit MeshProgram(const class MeshBuilder& builder);

    ~MeshProgram();

    // Delete copy, allow move
    MeshProgram(const MeshProgram&) = delete;
    MeshProgram& operator=(const MeshProgram&) = delete;
    MeshProgram(MeshProgram&&) noexcept;
    MeshProgram& operator=(MeshProgram&&) noexcept;

    /**
     * @brief Get the program for a specific mesh coordinate
     */
    tt::tt_metal::Program& program_at(const tt::tt_metal::distributed::MeshCoordinate& coord);
    const tt::tt_metal::Program& program_at(const tt::tt_metal::distributed::MeshCoordinate& coord) const;

    /**
     * @brief Register that a kernel was created for a specific mesh coordinate
     */
    void register_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord);

    /**
     * @brief Check if a mesh coordinate has any kernels registered
     */
    bool has_kernel(const tt::tt_metal::distributed::MeshCoordinate& coord) const;

    /**
     * @brief Register a data movement kernel on a specific global core
     */
    void register_dm_kernel_on_gcore(uint32_t gcore_id);

    /**
     * @brief Check if a global core already has a data movement kernel
     */
    bool has_dm_kernel_on_gcore(uint32_t gcore_id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Create a MeshProgram from a MeshBuilder
 */
MeshProgram CreateMeshProgram(const class MeshBuilder& builder);

}  // namespace tt::tt_metal::experimental::udm

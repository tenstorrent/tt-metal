// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/api/tt-metalium/kernel_types.hpp"
#include "tt_metal/api/tt-metalium/circular_buffer_config.hpp"
#include "tt_metal/api/tt-metalium/global_semaphore.hpp"
#include <tt-metalium/shape.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::udm {

// Forward declarations
class MeshBuilder;
class MeshProgram;

// ==================================================
//                  GlobalCore
// ==================================================

/**
 * @brief Global core identifier that abstracts cores across devices
 *
 * A GlobalCore represents a compute core in the global view, abstracting away
 * the underlying device and local core coordinates.
 */
struct GlobalCore {
    uint32_t local_id;                                       // ID within the local grid
    uint32_t global_id;                                      // ID in the global GlobalCore space
    tt::tt_metal::distributed::MeshCoordinate local_coord;   // Coordinate within the grid
    tt::tt_metal::distributed::MeshCoordinate global_coord;  // Global coordinate across all devices

    GlobalCore() :
        local_id(0),
        global_id(0),
        local_coord(tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(1)),
        global_coord(tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(1)) {}

    GlobalCore(
        uint32_t local_id,
        uint32_t global_id,
        const tt::tt_metal::distributed::MeshCoordinate& local_coord,
        const tt::tt_metal::distributed::MeshCoordinate& global_coord) :
        local_id(local_id), global_id(global_id), local_coord(local_coord), global_coord(global_coord) {}

    bool operator==(const GlobalCore& other) const { return global_id == other.global_id; }
    bool operator!=(const GlobalCore& other) const { return !(*this == other); }
    bool operator<(const GlobalCore& other) const { return global_id < other.global_id; }

    /**
     * @brief Convert local_coord to CoreCoord for use with SetRuntimeArgs
     *
     * @return CoreCoord(x, y) where x=local_coord[1], y=local_coord[0]
     */
    tt::tt_metal::CoreCoord to_core_coord() const {
        TT_FATAL(
            local_coord.dims() >= 2, "GlobalCore local_coord must be at least 2D, got dims={}", local_coord.dims());
        // Note that in udm we always specify coord and dims from outer dim to inner dim, which is reverse of CoreCoord
        return tt::tt_metal::CoreCoord(local_coord[1], local_coord[0]);
    }
};

// ==================================================
//                  Grid
// ==================================================

/**
 * @brief Virtualized grid abstraction for ND device grids
 *
 * A Grid represents a single device's compute grid in ND dimensions.
 * A grid consists of multiple gcores.
 */
struct Grid {
    uint32_t id;                                      // Grid ID (assigned in row-major order)
    tt::tt_metal::Shape dims;                         // ND grid dimensions
    tt::tt_metal::distributed::MeshCoordinate coord;  // Grid's position in the mesh

    Grid() :
        id(0), dims(std::vector<uint32_t>{1}), coord(tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(1)) {}

    Grid(uint32_t id, const tt::tt_metal::Shape& dims, const tt::tt_metal::distributed::MeshCoordinate& coord) :
        id(id), dims(dims), coord(coord) {}

    bool operator==(const Grid& other) const { return id == other.id; }
    bool operator!=(const Grid& other) const { return !(*this == other); }

    uint32_t size() const {
        uint32_t total = 1;
        for (size_t i = 0; i < dims.rank(); ++i) {
            total *= dims[i];
        }
        return total;
    }

    // Getters
    uint32_t rank() const { return dims.rank(); }
    uint32_t volume() const { return dims.volume(); }
    uint32_t operator[](size_t index) const { return dims[index]; }
    const tt::tt_metal::Shape& shape() const { return dims; }
};

// ==================================================
//                  Mesh
// ==================================================

/**
 * @brief Virtualized mesh device mesh
 *
 * A Mesh represents a virtualized arrangement of mesh devices,
 * allowing users to reshape physical mesh configurations.
 * A mesh consists of multiple grids (one per device).
 */
struct Mesh {
    tt::tt_metal::Shape dims;  // ND mesh dimensions (e.g., [2, 2, 2] or [1, 8])

    Mesh() = default;

    explicit Mesh(const tt::tt_metal::Shape& dims) : dims(dims) {}

    // Getters
    uint32_t rank() const { return dims.rank(); }
    uint64_t volume() const { return dims.volume(); }
    uint32_t operator[](size_t index) const { return dims[index]; }
    const tt::tt_metal::Shape& shape() const { return dims; }
};

// ==================================================
//                  Kernel Handle
// ==================================================

/**
 * @brief Handle for a mesh kernel
 * Maps grid_id to the kernel handle on that grid
 */
using MeshKernelHandle = std::unordered_map<uint32_t, tt::tt_metal::KernelHandle>;

/**
 * @brief Handle for a mesh circular buffer
 * Maps grid_id to the circular buffer handle on that grid
 */
using MeshCBHandle = std::unordered_map<uint32_t, tt::tt_metal::CBHandle>;

/**
 * @brief Handle for a mesh semaphore backed by GlobalSemaphore
 * Contains the GlobalSemaphore object (to keep it alive) and provides address access
 */
struct MeshSemaphoreHandle {
    tt::tt_metal::GlobalSemaphore semaphore;  // Keeps the semaphore alive
    uint32_t address_;                        // The L1 address (same on all devices)

    // Constructor
    MeshSemaphoreHandle(tt::tt_metal::GlobalSemaphore sem) :
        semaphore(std::move(sem)), address_(static_cast<uint32_t>(semaphore.address())) {}

    // For compatibility with existing code that uses .at(grid_id)
    uint32_t at(uint32_t /*grid_id*/) const { return address_; }

    // Direct address access
    uint32_t address() const { return address_; }
};

}  // namespace tt::tt_metal::experimental::udm

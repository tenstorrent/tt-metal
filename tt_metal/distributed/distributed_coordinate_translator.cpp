// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {
namespace {

// TODO: #21096 - Remove these aliases once the distributed namespace is removed.
using distributed::MeshCoordinate;
using distributed::MeshShape;

// Validates the configuration of the distributed coordinate system.
void validate_config(const MeshShape& global_shape, const MeshShape& local_shape, const MeshCoordinate& local_offset) {
    TT_FATAL(
        local_offset.dims() == global_shape.dims() && local_offset.dims() == local_shape.dims(),
        "Dimension mismatch between global shape {}, local shape {}, and offset {}",
        global_shape,
        local_shape,
        local_offset);

    for (size_t dim = 0; dim < local_offset.dims(); ++dim) {
        TT_FATAL(
            local_offset[dim] + local_shape[dim] <= global_shape[dim],
            "Local mesh extends beyond global mesh boundaries at dimension {}; local shape {}, global shape {}, local "
            "offset {}",
            dim,
            local_shape,
            global_shape,
            local_offset);
    }
}

}  // namespace

DistributedCoordinateTranslator::DistributedCoordinateTranslator(
    const MeshShape& global_shape, const MeshShape& local_shape, const MeshCoordinate& local_offset) :
    global_shape_(global_shape), local_shape_(local_shape), local_offset_(local_offset) {
    validate_config(global_shape, local_shape, local_offset);
}

bool DistributedCoordinateTranslator::is_local(const MeshCoordinate& global_coord) const {
    TT_FATAL(
        global_coord.dims() == global_shape_.dims(),
        "Global coordinate {} has different dimensions than the global shape {}",
        global_coord,
        global_shape_);
    TT_FATAL(
        distributed::MeshCoordinateRange(global_shape_).contains(global_coord),
        "Global coordinate {} is out of bounds for the global shape {}",
        global_coord,
        global_shape_);

    // Check if the coordinate falls within this host's local mesh bounds
    for (size_t dim = 0; dim < global_coord.dims(); ++dim) {
        if (global_coord[dim] < local_offset_[dim] || global_coord[dim] >= local_offset_[dim] + local_shape_[dim]) {
            return false;
        }
    }
    return true;
}

MeshCoordinate DistributedCoordinateTranslator::local_to_global(const MeshCoordinate& local_coord) const {
    TT_FATAL(
        local_coord.dims() == local_shape_.dims(),
        "Local coordinate {} has different dimensions than the local shape {}",
        local_coord,
        local_shape_);
    TT_FATAL(
        distributed::MeshCoordinateRange(local_shape_).contains(local_coord),
        "Local coordinate {} is out of bounds for the local shape {}",
        local_coord,
        local_shape_);

    MeshCoordinate global_coord = local_coord;
    for (size_t dim = 0; dim < local_coord.dims(); ++dim) {
        global_coord[dim] += local_offset_[dim];
    }
    return global_coord;
}

}  // namespace tt::tt_metal

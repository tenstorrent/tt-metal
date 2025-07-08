// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/coordinate_translator.hpp"
#include <tt-metalium/assert.hpp>

namespace tt::tt_metal::distributed {

CoordinateTranslator::CoordinateTranslator(const MeshShape& local_shape, const MeshCoordinate& local_offset)
    : local_shape_(local_shape), local_offset_(local_offset) {
    // Validate dimensions match
    TT_FATAL(local_shape.dims() == local_offset.dims(),
             "Dimension mismatch: local_shape has {} dimensions, local_offset has {} dimensions",
             local_shape.dims(), local_offset.dims());
}

std::optional<MeshCoordinate> CoordinateTranslator::global_to_local(const MeshCoordinate& global_coord) const {
    tt::stl::SmallVector<uint32_t> local_coord(global_coord.dims());
    for (size_t dim = 0; dim < global_coord.dims(); ++dim) {
        if (global_coord[dim] < local_offset_[dim] || 
            global_coord[dim] >= local_offset_[dim] + local_shape_[dim]) {
            return std::nullopt;
        }
        local_coord[dim] = global_coord[dim] - local_offset_[dim];
    }
    return MeshCoordinate(local_coord);
}

MeshCoordinate CoordinateTranslator::local_to_global(const MeshCoordinate& local_coord) const {
    // Validate input dimensions
    TT_FATAL(local_coord.dims() == local_shape_.dims(),
             "Dimension mismatch: local_coord has {} dimensions, expected {}",
             local_coord.dims(), local_shape_.dims());
    
    tt::stl::SmallVector<uint32_t> global_coord(local_coord.dims());
    for (size_t dim = 0; dim < local_coord.dims(); ++dim) {
        // Validate local coordinate is within bounds
        TT_FATAL(local_coord[dim] < local_shape_[dim],
                 "Local coordinate[{}]={} exceeds local shape[{}]={}",
                 dim, local_coord[dim], dim, local_shape_[dim]);
        
        global_coord[dim] = local_coord[dim] + local_offset_[dim];
    }
    return MeshCoordinate(global_coord);
}

bool CoordinateTranslator::is_local_coordinate(const MeshCoordinate& global_coord) const {
    return global_to_local(global_coord).has_value();
}

MeshCoordinate CoordinateTranslator::translate_or_fatal(const MeshCoordinate& global_coord) const {
    auto local_coord = global_to_local(global_coord);
    TT_FATAL(local_coord.has_value(), 
             "Global coordinate {} is out of bounds for local mesh with offset {} and shape {}",
             global_coord, local_offset_, local_shape_);
    return local_coord.value();
}

const MeshShape& CoordinateTranslator::local_shape() const noexcept { 
    return local_shape_; 
}

const MeshCoordinate& CoordinateTranslator::local_offset() const noexcept { 
    return local_offset_; 
}

}  // namespace tt::tt_metal::distributed
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <vector>

namespace tt::tt_metal {

DistributedHostBuffer DistributedHostBuffer::create(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset) {
    TT_FATAL(
        global_shape.dims() == local_shape.dims(),
        "Global shape {} and local shape {} must have the same dimensions",
        global_shape,
        local_shape);
    TT_FATAL(
        global_shape.dims() == local_offset.dims(),
        "Global shape {} and local offset {} must have the same dimensions",
        global_shape,
        local_offset);

    for (size_t dim = 0; dim < global_shape.dims(); ++dim) {
        TT_FATAL(
            local_offset[dim] + local_shape[dim] <= global_shape[dim],
            "Local shape exceeds global shape at dimension {}: {} + {} > {}",
            dim,
            local_offset[dim],
            local_shape[dim],
            global_shape[dim]);
    }
    return DistributedHostBuffer(
        std::move(global_shape),
        std::move(local_offset),
        distributed::MeshContainer<HostBuffer>(local_shape, HostBuffer()));
}

std::optional<distributed::MeshCoordinate> DistributedHostBuffer::global_to_local(
    const distributed::MeshCoordinate& coord) const {
    const auto& local_shape = local_buffers_.shape();
    tt::stl::SmallVector<uint32_t> local_coord(coord.dims());
    for (size_t dim = 0; dim < coord.dims(); ++dim) {
        if (coord[dim] < local_offset_[dim] || coord[dim] >= local_offset_[dim] + local_shape[dim]) {
            return std::nullopt;
        }
        local_coord[dim] = coord[dim] - local_offset_[dim];
    }
    return distributed::MeshCoordinate(local_coord);
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    TT_FATAL(
        distributed::MeshCoordinateRange(global_shape_).contains(coord),
        "Coordinate {} is outside the global shape bounds {}",
        coord,
        global_shape_);

    auto local_coord_opt = global_to_local(coord);
    return local_coord_opt.has_value() ? std::optional<HostBuffer>(local_buffers_.at(*local_coord_opt)) : std::nullopt;
}

void DistributedHostBuffer::emplace_shard(
    const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer) {
    TT_FATAL(
        distributed::MeshCoordinateRange(global_shape_).contains(coord),
        "Coordinate {} is outside the global shape bounds {}",
        coord,
        global_shape_);

    populated_shards_.insert(coord);
    auto local_coord_opt = global_to_local(coord);
    if (local_coord_opt.has_value()) {
        local_buffers_.at(*local_coord_opt) = produce_buffer();
    }
}

DistributedHostBuffer DistributedHostBuffer::transform(const TransformFn& fn) const {
    std::vector<HostBuffer> transformed_buffers;
    transformed_buffers.reserve(local_buffers_.shape().mesh_size());
    for (const auto& local_buffer : local_buffers_.values()) {
        transformed_buffers.push_back(fn(local_buffer));
    }
    DistributedHostBuffer transformed_buffer(
        global_shape_,
        local_offset_,
        distributed::MeshContainer<HostBuffer>(local_buffers_.shape(), std::move(transformed_buffers)));
    return transformed_buffer;
}

void DistributedHostBuffer::apply(const ApplyFn& fn) const {
    for (const auto& local_buffer : local_buffers_.values()) {
        fn(local_buffer);
    }
}

distributed::MeshShape DistributedHostBuffer::shape() const { return global_shape_; }

const std::unordered_set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const {
    return populated_shards_;
}

}  // namespace tt::tt_metal

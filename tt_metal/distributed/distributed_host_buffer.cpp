// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <vector>
#include <functional>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include "common/executor.hpp"

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
        global_shape, local_offset, distributed::MeshContainer<Shard>(local_shape, Shard{.is_populated = false}));
}

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshShape& shape) {
    return DistributedHostBuffer::create(shape, shape, distributed::MeshCoordinate::zero_coordinate(shape.dims()));
}

std::optional<distributed::MeshCoordinate> DistributedHostBuffer::global_to_local(
    const distributed::MeshCoordinate& coord) const {
    const auto& local_shape = local_shards_.shape();
    tt::stl::SmallVector<uint32_t> local_coord(coord.dims());
    for (size_t dim = 0; dim < coord.dims(); ++dim) {
        if (coord[dim] < local_offset_[dim] || coord[dim] >= local_offset_[dim] + local_shape[dim]) {
            return std::nullopt;
        }
        local_coord[dim] = coord[dim] - local_offset_[dim];
    }
    return distributed::MeshCoordinate(local_coord);
}

std::vector<size_t> DistributedHostBuffer::get_populated_local_shard_indices() const {
    std::vector<size_t> indices;
    indices.reserve(local_shards_.values().size());
    for (size_t i = 0; i < local_shards_.values().size(); ++i) {
        if (local_shards_.values()[i].is_populated) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    TT_FATAL(
        distributed::MeshCoordinateRange(global_shape_).contains(coord),
        "Coordinate {} is outside the global shape bounds {}",
        coord,
        global_shape_);

    if (auto local_coord_opt = global_to_local(coord);
        local_coord_opt.has_value() && local_shards_.at(*local_coord_opt).is_populated) {
        return std::make_optional(local_shards_.at(*local_coord_opt).buffer);
    } else {
        return std::nullopt;
    }
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
        local_shards_.at(*local_coord_opt) = Shard{.buffer = produce_buffer(), .is_populated = true};
    }
}

DistributedHostBuffer DistributedHostBuffer::transform(
    const TransformFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_local_shard_indices();
    const auto& local_shards = local_shards_.values();
    std::vector<Shard> transformed_shards(local_shards.size());
    if (policy == ProcessShardExecutionPolicy::SEQUENTIAL || indices_to_process.size() < 2) {
        std::for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] = Shard{.buffer = fn(local_shards[i].buffer), .is_populated = true};
        });
    } else {
        tf::Taskflow taskflow;
        taskflow.for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] = Shard{.buffer = fn(local_shards[i].buffer), .is_populated = true};
        });
        detail::GetExecutor().run(taskflow).wait();
    }
    return DistributedHostBuffer(
        global_shape_,
        local_offset_,
        distributed::MeshContainer<Shard>(local_shards_.shape(), std::move(transformed_shards)));
}

void DistributedHostBuffer::apply(const ApplyFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_local_shard_indices();
    const auto& local_shards = local_shards_.values();
    if (policy == ProcessShardExecutionPolicy::SEQUENTIAL || indices_to_process.size() < 2) {
        std::for_each(
            indices_to_process.begin(), indices_to_process.end(), [&](size_t i) { fn(local_shards[i].buffer); });
    } else {
        tf::Taskflow taskflow;
        taskflow.for_each(
            indices_to_process.begin(), indices_to_process.end(), [&](size_t i) { fn(local_shards[i].buffer); });
        detail::GetExecutor().run(taskflow).wait();
    }
}

const distributed::MeshShape& DistributedHostBuffer::shape() const { return global_shape_; }

const std::set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const { return populated_shards_; }

}  // namespace tt::tt_metal

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
    return DistributedHostBuffer::create(DistributedMeshShape(global_shape, local_shape, local_offset));
}

DistributedHostBuffer DistributedHostBuffer::create(const DistributedMeshShape& distributed_mesh_shape) {
    return DistributedHostBuffer(
        distributed_mesh_shape,
        distributed::MeshContainer<Shard>(distributed_mesh_shape.shape(), Shard{.is_populated = false}),
        /*populated_shards=*/std::set<distributed::MeshCoordinate>{});
}

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshShape& shape) {
    return DistributedHostBuffer::create(shape, shape, distributed::MeshCoordinate::zero_coordinate(shape.dims()));
}

std::vector<size_t> DistributedHostBuffer::get_populated_shard_indices() const {
    std::vector<size_t> indices;
    indices.reserve(shards_.values().size());
    for (size_t i = 0; i < shards_.values().size(); ++i) {
        if (shards_.values()[i].is_populated) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    if (distributed_mesh_shape_.is_local(coord) && shards_.at(coord).is_populated) {
        return std::make_optional(shards_.at(coord).buffer);
    } else {
        return std::nullopt;
    }
}

void DistributedHostBuffer::emplace_shard(
    const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer) {
    shard_coords_.insert(coord);

    if (distributed_mesh_shape_.is_local(coord)) {
        shards_.at(coord) = Shard{.buffer = produce_buffer(), .is_populated = true};
    }
}

DistributedHostBuffer DistributedHostBuffer::transform(
    const TransformFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_shard_indices();
    const auto& shards = shards_.values();
    std::vector<Shard> transformed_shards(shards.size());
    if (policy == ProcessShardExecutionPolicy::SEQUENTIAL || indices_to_process.size() < 2) {
        std::for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] = Shard{.buffer = fn(shards[i].buffer), .is_populated = true};
        });
    } else {
        tf::Taskflow taskflow;
        taskflow.for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] = Shard{.buffer = fn(shards[i].buffer), .is_populated = true};
        });
        detail::GetExecutor().run(taskflow).wait();
    }
    return DistributedHostBuffer(
        distributed_mesh_shape_,
        distributed::MeshContainer<Shard>(shards_.shape(), std::move(transformed_shards)),
        shard_coords_);
}

void DistributedHostBuffer::apply(const ApplyFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_shard_indices();
    const auto& local_shards = shards_.values();
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

const distributed::MeshShape& DistributedHostBuffer::shape() const { return distributed_mesh_shape_.shape(); }

const std::set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const { return shard_coords_; }

}  // namespace tt::tt_metal

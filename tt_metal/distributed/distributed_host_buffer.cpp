// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>

#include <vector>
#include <functional>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include "common/executor.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

namespace tt::tt_metal {

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshShape& shape) {
    return DistributedHostBuffer::create(
        shape,
        shape,
        distributed::MeshCoordinate::zero_coordinate(shape.dims()),
        tt::tt_metal::MetalContext::instance().get_control_plane().get_host_local_context());
}

DistributedHostBuffer DistributedHostBuffer::create(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset,
    const std::shared_ptr<distributed::multihost::DistributedContext>& context) {
    DistributedCoordinateTranslator translator(global_shape, local_shape, local_offset);
    std::vector<distributed::MaybeRemote<Shard>> shards(
        global_shape.mesh_size(), distributed::MaybeRemote<Shard>::remote());

    int shard_index = 0;
    for (const auto& coord : distributed::MeshCoordinateRange(global_shape)) {
        if (translator.is_local(coord)) {
            shards[shard_index] = distributed::MaybeRemote<Shard>::local(Shard{.is_populated = false});
        }
        ++shard_index;
    }

    return DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard>(global_shape, std::move(shards)),
        /*populated_shards=*/{},
        context);
}

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshDeviceView& mesh_device_view) {
    std::vector<distributed::MaybeRemote<Shard>> shards(
        mesh_device_view.shape().mesh_size(), distributed::MaybeRemote<Shard>::remote());

    auto distributed_context =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_distributed_context(mesh_device_view.mesh_id());

    int shard_index = 0;
    for (auto maybe_device : mesh_device_view) {
        maybe_device.if_local([&](const auto&) {
            shards[shard_index] = distributed::MaybeRemote<Shard>::local(Shard{.is_populated = false});
        });
        ++shard_index;
    }

    return DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard>(mesh_device_view.shape(), std::move(shards)),
        /*populated_shards=*/std::set<distributed::MeshCoordinate>{},
        std::move(distributed_context));
}

std::vector<size_t> DistributedHostBuffer::get_populated_shard_indices() const {
    const auto& shards_flat = shards_.values();

    std::vector<size_t> indices;
    indices.reserve(shards_flat.size());
    for (size_t i = 0; i < shards_flat.size(); ++i) {
        if (shards_flat[i].is_local() && shards_flat[i]->is_populated) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    const auto& shard = shards_.at(coord);
    if (shard.is_local() && shard->is_populated) {
        return std::make_optional(shard->buffer);
    } else {
        return std::nullopt;
    }
}

void DistributedHostBuffer::emplace_shard(
    const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer) {
    shard_coords_.insert(coord);

    auto& shard = shards_.at(coord);
    if (shard.is_local()) {
        shard->buffer = produce_buffer();
        shard->is_populated = true;
    }
}

bool DistributedHostBuffer::is_local(const distributed::MeshCoordinate& coord) const { return shards_.is_local(coord); }

DistributedHostBuffer DistributedHostBuffer::transform(
    const TransformFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_shard_indices();
    const auto& shards = shards_.values();
    std::vector<distributed::MaybeRemote<Shard>> transformed_shards(
        shards.size(), distributed::MaybeRemote<Shard>::remote());
    if (policy == ProcessShardExecutionPolicy::SEQUENTIAL || indices_to_process.size() < 2) {
        std::for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] =
                distributed::MaybeRemote<Shard>::local(Shard{.buffer = fn(shards[i]->buffer), .is_populated = true});
        });
    } else {
        tf::Taskflow taskflow;
        taskflow.for_each(indices_to_process.begin(), indices_to_process.end(), [&](size_t i) {
            transformed_shards[i] =
                distributed::MaybeRemote<Shard>::local(Shard{.buffer = fn(shards[i]->buffer), .is_populated = true});
        });
        detail::GetExecutor().run(taskflow).wait();
    }
    return DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard>(shards_.shape(), std::move(transformed_shards)),
        shard_coords_,
        context_);
}

void DistributedHostBuffer::apply(const ApplyFn& fn, ProcessShardExecutionPolicy policy) const {
    const std::vector<size_t> indices_to_process = get_populated_shard_indices();
    const auto& local_shards = shards_.values();
    if (policy == ProcessShardExecutionPolicy::SEQUENTIAL || indices_to_process.size() < 2) {
        std::for_each(
            indices_to_process.begin(), indices_to_process.end(), [&](size_t i) { fn(local_shards[i]->buffer); });
    } else {
        tf::Taskflow taskflow;
        taskflow.for_each(
            indices_to_process.begin(), indices_to_process.end(), [&](size_t i) { fn(local_shards[i]->buffer); });
        detail::GetExecutor().run(taskflow).wait();
    }
}

const distributed::MeshShape& DistributedHostBuffer::shape() const { return shards_.shape(); }

const std::set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const { return shard_coords_; }

const std::shared_ptr<distributed::multihost::DistributedContext>& DistributedHostBuffer::context() const {
    return context_;
}

}  // namespace tt::tt_metal

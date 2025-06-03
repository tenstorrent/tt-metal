// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <vector>

namespace tt::tt_metal {
namespace {

// `ParallelForAdaptor` implementation that executes added tasks immediately on the same thread.
class InlineParallelForAdaptor : public DistributedHostBuffer::ParallelForAdaptor {
public:
    void add_task(std::function<void()>&& task) override { task(); }
    void wait() override {}
};

DistributedHostBuffer::ParallelForAdaptor* get_inline_parallel_for_adaptor() {
    static tt::stl::Indestructible<InlineParallelForAdaptor> adaptor;
    return &(adaptor.get());
}

}  // namespace

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
        distributed::MeshContainer<Shard>(local_shape, Shard{.is_populated = false}));
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

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    TT_FATAL(
        distributed::MeshCoordinateRange(global_shape_).contains(coord),
        "Coordinate {} is outside the global shape bounds {}",
        coord,
        global_shape_);

    auto local_coord_opt = global_to_local(coord);
    return local_coord_opt.has_value() ? std::optional<HostBuffer>(local_shards_.at(*local_coord_opt).buffer)
                                       : std::nullopt;
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

DistributedHostBuffer DistributedHostBuffer::transform(const TransformFn& fn, ParallelForAdaptor* parallel_for) const {
    parallel_for = parallel_for ? parallel_for : get_inline_parallel_for_adaptor();

    std::vector<Shard> transformed_shards;
    transformed_shards.reserve(local_shards_.shape().mesh_size());
    for (const auto& local_shard : local_shards_.values()) {
        if (local_shard.is_populated) {
            parallel_for->add_task(
                [&]() { transformed_shards.push_back(Shard{.buffer = fn(local_shard.buffer), .is_populated = true}); });
        } else {
            transformed_shards.push_back(local_shard);
        }
    }
    parallel_for->wait();

    DistributedHostBuffer transformed_buffer(
        global_shape_,
        local_offset_,
        distributed::MeshContainer<Shard>(local_shards_.shape(), std::move(transformed_shards)));
    return transformed_buffer;
}

void DistributedHostBuffer::apply(const ApplyFn& fn, ParallelForAdaptor* parallel_for) const {
    parallel_for = parallel_for ? parallel_for : get_inline_parallel_for_adaptor();

    for (const auto& local_buffer : local_shards_.values()) {
        if (local_buffer.is_populated) {
            parallel_for->add_task([&]() { fn(local_buffer.buffer); });
        }
    }
    parallel_for->wait();
}

distributed::MeshShape DistributedHostBuffer::shape() const { return global_shape_; }

const std::set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const { return populated_shards_; }

}  // namespace tt::tt_metal

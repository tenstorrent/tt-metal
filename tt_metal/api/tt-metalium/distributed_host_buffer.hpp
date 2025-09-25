// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <functional>
#include <vector>

namespace tt::tt_metal {

// `DistributedHostBuffer` is an abstraction layer over host-side physical data, allowing convenient representation for
// data distributed over multiple hosts. The buffer incorporates information about the global shape across multiple
// hosts along with the local shape and offset, to skip data loading and processing for remote data shards.
//
// TODO: provide a way to allocate individual buffers on a memory arena.
class DistributedHostBuffer {
public:
    DistributedHostBuffer(const DistributedHostBuffer&) = default;
    DistributedHostBuffer& operator=(const DistributedHostBuffer&) = default;
    DistributedHostBuffer(DistributedHostBuffer&&) = default;
    DistributedHostBuffer& operator=(DistributedHostBuffer&&) = default;

    // Creates a multi-host distributed buffer with the specified parameters.
    // The size of `global_buffers` indicates the global size of the buffer; `local_shape` and `local_offset` must be
    // consistent with the global shape. `global_buffers` that are remote to this host will be deallocated and ignored
    // for all subsequent operations. Only local buffers will be retained by this `DistributedHostBuffer` instance.
    static DistributedHostBuffer create(
        const distributed::MeshShape& global_shape,
        const distributed::MeshShape& local_shape,
        const distributed::MeshCoordinate& local_offset,
        const std::shared_ptr<distributed::multihost::DistributedContext>& context);

    // Creates a multi-host distributed buffer that matches shape and multi-host distribution of the mesh device view.
    static DistributedHostBuffer create(const distributed::MeshDeviceView& mesh_device_view);

    // Shorthand for creating a distributed buffer for a single host.
    static DistributedHostBuffer create(const distributed::MeshShape& shape);

    // Returns the shard at the specified `coord`.
    // Returns `std::nullopt` if the index is out of local bounds or if the shard is not populated.
    // Throws if the index is out of global bounds.
    std::optional<HostBuffer> get_shard(const distributed::MeshCoordinate& coord) const;

    // Emplaces the shard at the specified `coord`, calling `produce_buffer` to create the buffer only when needed.
    // No-op if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    using ProduceBufferFn = std::function<HostBuffer(const distributed::MeshCoordinate&)>;
    void emplace_shard(const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer);

    // Returns true if the shard at the specified `coord` is local, false if remote.
    bool is_local(const distributed::MeshCoordinate& coord) const;

    // Specifies the execution policy for the `transform` and `apply` functions.
    enum class ProcessShardExecutionPolicy {
        SEQUENTIAL,
        PARALLEL,
    };

    // `transform` and `apply` functions abstract away the details of the underlying data storage.
    // For global multi-host buffers, these functions will only be invoked for the local populated shards.
    using TransformFn = std::function<HostBuffer(const HostBuffer& buffer)>;
    DistributedHostBuffer transform(
        const TransformFn& fn, ProcessShardExecutionPolicy policy = ProcessShardExecutionPolicy::SEQUENTIAL) const;

    using ApplyFn = std::function<void(const HostBuffer& buffer)>;
    void apply(const ApplyFn& fn, ProcessShardExecutionPolicy policy = ProcessShardExecutionPolicy::SEQUENTIAL) const;

    // NOTE: `coords` are global, so this will skip non local shards.
    // Calls emplace_shard for each coordinate in `coords` using the specified execution policy.
    void emplace_shards(
        const std::vector<distributed::MeshCoordinate>& coords,
        const ProduceBufferFn& produce_buffer,
        ProcessShardExecutionPolicy policy = ProcessShardExecutionPolicy::SEQUENTIAL);

    // Returns the global shape of the buffer.
    const distributed::MeshShape& shape() const;

    // Returns the coordinates of populated shards in the buffer.
    const std::set<distributed::MeshCoordinate>& shard_coords() const;

    // Returns the distributed context for the buffer.
    const std::shared_ptr<distributed::multihost::DistributedContext>& context() const;

private:
    // Converts a global coordinate to a local coordinate.
    // Returns `std::nullopt` if the coordinate is out of local bounds.
    std::optional<distributed::MeshCoordinate> global_to_local(const distributed::MeshCoordinate& coord) const;

    // Returns the indices of populated shards in `shards_`.
    std::vector<size_t> get_populated_shard_indices() const;

    struct Shard {
        HostBuffer buffer;
        bool is_populated = false;
    };

    DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard> shards,
        std::set<distributed::MeshCoordinate> populated_shards,
        std::shared_ptr<distributed::multihost::DistributedContext> context) :
        shards_(std::move(shards)), shard_coords_(std::move(populated_shards)), context_(std::move(context)) {}

    // The shards of the buffer.
    // Remote shards are never materialized, but not all of the local shards are necessarily populated.
    distributed::DistributedMeshContainer<Shard> shards_;

    // Keeps track of global shards that were attempted to be written to.
    std::set<distributed::MeshCoordinate> shard_coords_;

    // The distributed context for the buffer.
    std::shared_ptr<distributed::multihost::DistributedContext> context_;
};

}  // namespace tt::tt_metal

/// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <functional>
#include <unordered_set>
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
        const distributed::MeshCoordinate& local_offset);

    // Returns the shard at the specified `coord`.
    // Returns `std::nullopt` if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    std::optional<HostBuffer> get_shard(const distributed::MeshCoordinate& coord) const;

    // Emplaces the shard at the specified `coord`, calling `produce_buffer` to create the buffer only when needed.
    // No-op if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    void emplace_shard(const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer);

    // `transform` and `apply` functions abstract away the details of the underlying data storage.
    // `linear_index` will be supplied by `DistributedHostBuffer` to indicate the position of the buffer.
    // For global multi-host buffers, these functions will only be invoked for the local shards.
    //
    // TODO: provide an optional way to parallelize the operation.
    using TransformFn = std::function<HostBuffer(const HostBuffer& buffer)>;
    DistributedHostBuffer transform(const TransformFn& fn) const;

    using ApplyFn = std::function<void(const HostBuffer& buffer)>;
    void apply(const ApplyFn& fn) const;

    // Returns the global shape of the buffer.
    distributed::MeshShape shape() const;

    // Returns the coordinates of populated shards in the buffer.
    const std::unordered_set<distributed::MeshCoordinate>& shard_coords() const;

private:
    std::optional<distributed::MeshCoordinate> global_to_local(const distributed::MeshCoordinate& coord) const;

    DistributedHostBuffer(
        distributed::MeshShape global_shape,
        distributed::MeshCoordinate local_offset,
        distributed::MeshContainer<HostBuffer> local_buffers) :
        global_shape_(std::move(global_shape)),
        local_offset_(std::move(local_offset)),
        local_buffers_(std::move(local_buffers)) {}

    distributed::MeshShape global_shape_;
    distributed::MeshCoordinate local_offset_;
    distributed::MeshContainer<HostBuffer> local_buffers_;
    std::unordered_set<distributed::MeshCoordinate> populated_shards_;
};

}  // namespace tt::tt_metal

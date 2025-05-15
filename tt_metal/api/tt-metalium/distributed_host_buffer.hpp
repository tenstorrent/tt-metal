// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/assert.hpp>

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

    // Creates a buffer of the provided `global_size`, spanning a single host.
    // TODO: remove in the long term. For now, this is supporting TTNN's "MultiDeviceHostStorage" API that does not
    // include the shape.
    static DistributedHostBuffer create(size_t global_size);

    // Creates a multi-host distributed buffer with the specified parameters.
    // The size of `global_buffers` indicates the global size of the buffer; `local_shape` and `local_offset` must be
    // consistent with the global shape. `global_buffers` that are remote to this host will be deallocated and ignored
    // for all subsequent operations. Only local buffers will be retained by this `DistributedHostBuffer` instance.
    static DistributedHostBuffer create(
        const distributed::MeshShape& global_shape,
        const distributed::MeshShape& local_shape,
        const distributed::MeshCoordinate& local_offset);

    // TODO: use `MeshCoordinate` to specify `linear_index`. Currently, the problem is that on creation of "multi host
    // device" buffer in TTNN, the shape is not specified, so we cannot onboard `DistributedHostBuffer` to use
    // coordinate system natively.

    // Returns the shard at the specified `linear_index`.
    // Returns `std::nullopt` if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    std::optional<HostBuffer> get_shard(size_t linear_index) const;

    // Emplaces the shard at the specified `linear_index`.
    // No-op if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    void emplace_shard(size_t linear_index, HostBuffer buffer);

    // `transform` and `apply` functions abstract away the details of the underlying data storage.
    // `linear_index` will be supplied by `DistributedHostBuffer` to indicate the position of the buffer.
    // For global multi-host buffers, these functions will only be invoked for the local shards.
    //
    // TODO: provide an optional way to parallelize the operation.
    using TransformFn = std::function<HostBuffer(const HostBuffer& buffer, size_t linear_index)>;
    void transform(const TransformFn& fn);

    using ApplyFn = std::function<void(const HostBuffer& buffer, size_t linear_index)>;
    void apply(const ApplyFn& fn);

    // Returns true if the buffer is allocated.
    bool is_allocated() const;

    // Deallocates the underlying data storage.
    void deallocate();

private:
    DistributedHostBuffer(
        std::function<std::optional<size_t>(size_t)> global_to_local_index, std::vector<HostBuffer> local_buffers) :
        global_to_local_index_(std::move(global_to_local_index)), local_buffers_(std::move(local_buffers)) {}

    std::function<std::optional<size_t>(size_t)> global_to_local_index_;
    std::vector<HostBuffer> local_buffers_;
};

}  // namespace tt::tt_metal

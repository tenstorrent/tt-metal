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

// `HostMeshBuffer` is an abstraction layer over host-side physical data, allowing convenient representation for data
// distributed over multiple hosts.
// The class supports 2 main modes of operation:
//
// 1. Replicated buffer. A single `HostBuffer` represents data to be broadcasted across devices; whether it is a local
// `MeshDevice` or a global mesh in the multi-host configuration.
//
// 2. Sharded buffer. Each device receives a local shard with the data. `HostMeshBuffer` provides `transform` and
// `apply` methods in case host-local physical data manipulation needs to be performed.
//
// TODO: provide a way to allocate individual buffers on a memory arena.
class HostMeshBuffer {
public:
    HostMeshBuffer(const HostMeshBuffer&) = default;
    HostMeshBuffer& operator=(const HostMeshBuffer&) = default;
    HostMeshBuffer(HostMeshBuffer&&) = default;
    HostMeshBuffer& operator=(HostMeshBuffer&&) = default;

    // Creates a replicated buffer from the specified `host_buffer`.
    static HostMeshBuffer create_replicated(HostBuffer host_buffer);

    // Creates a sharded buffer of the provided `global_size` spanning a single host.
    // TODO: remove in the long term. For now, this is supporting TTNN's "MultiDeviceHostStorage" API that does not
    // include the shape.
    static HostMeshBuffer create_sharded(size_t global_size);

    // Creates a multi-host distributed buffer with the specified parameters.
    // The size of `global_buffers` indicates the global size of the buffer; `local_shape` and `local_offset` must be
    // consistent with the global shape. `global_buffers` that are remote to this host will be deallocated and ignored
    // for all subsequent operations. Only local buffers will be retained by this `HostMeshBuffer` instance.
    static HostMeshBuffer create_sharded(
        const distributed::MeshShape& global_shape,
        const distributed::MeshShape& local_shape,
        const distributed::MeshCoordinate& local_offset);

    // TODO: use `MeshCoordinate` to specify `linear_index`. Currently, the problem is that on creation of "multi host
    // device" buffer in TTNN, the shape is not specified, so we cannot onboard `HostMeshBuffer` to use coordinate
    // system natively.

    // Returns the buffer at the specified `linear_index`.
    // Returns `std::nullopt` if the index is out of local bounds.
    // Throws if the index is out of global bounds.
    std::optional<HostBuffer> get_buffer(size_t linear_index) const;

    // Emplaces the buffer at the specified `linear_index`.
    // No-op if the index is out of local bounds.
    // Throws if the index is out of global bounds, or if the buffer is replicated.
    void emplace_buffer(size_t linear_index, HostBuffer buffer);

    // `transform` and `apply` functions abstract away the details of the underlying data storage.
    // `linear_index` will be supplied by `HostMeshBuffer` to indicate the position of the buffer.
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
    struct Replicated {
        HostBuffer buffer;
    };
    struct Sharded {
        std::function<std::optional<size_t>(size_t)> global_to_local_index;
        std::vector<HostBuffer> local_buffers;
    };

    HostMeshBuffer(std::variant<Replicated, Sharded> data) : data_(std::move(data)) {}

    std::variant<Replicated, Sharded> data_;
};

}  // namespace tt::tt_metal

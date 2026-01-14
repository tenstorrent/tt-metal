// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/buffer.hpp>

// Tensor related constructs
#include <tt-metalium/experimental/tensor/spec/spec_fwd.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/memory_pin.hpp>

namespace tt::tt_metal /*::tensor*/ {
/**
 * HostTensor is a host data class. It has the semantics of a container, and all host <-> device communications are
 * explicit.
 *
 */
class HostTensor {
    /**
     * To avoid disruption to existing users, HostTensor will deviate very little from the existing (host) Tensor
     * semantics. The only significant changes are:
     * - Eliminating implicit data movement APIs.
     * - Remove transformation methods like to_layout and pad from the class methods. These seem better as free
     *   functions that operate on a HostTensor than as methods of HostTensor. (Separation of data storage and data
     *   manipulation.) In the existing Tensor, these are already duplicated as both methods and free functions.
     */

public:
    // Special Member functions

    /**
     * Constructs an empty host tensor.
     */
    HostTensor() = default;
    ~HostTensor() = default;

    HostTensor(const HostTensor&) = default;
    HostTensor& operator=(const HostTensor&) = default;

    HostTensor(HostTensor&&) = default;
    HostTensor& operator=(HostTensor&&) = default;

    // End special member functions

    // constructions:

    // TODO: What are the assumptions with these buffers?

    // TODO: Original Tensor has three overloads
    // TODO-ask alex: Original Tensor takes HostBuffer as value, is that a must?
    explicit HostTensor(const HostBuffer&, TensorSpec spec);

    /**
     * From original Tensor:
     * Converts a buffer of elements of type `T` to a `Tensor`.
     * Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
     * elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
     *
     * The data in the buffer is copied into a tensor with host storage.
     */
    template <typename T>
    static HostTensor from_span(std::span<T> buffer, const TensorSpec& spec, T pad_value = 0);

    // TODO: overload for MemoryPin
    // TODO-ask alex: what is this tile thing?
    // TODO-ask alex: why are we not taking a TensorSpec here?
    /**
     * From original Tensor:
     * Creates a `Tensor` with storage "borrowed" from the buffer of elements of type `T`.
     */
    template <typename T>
    static HostTensor from_borrowed_data(
        std::span<T> buffer, const Shape& shape, MemoryPin pin, const std::optional<Tile>& tile = std::nullopt);

    // TODO: This should just be caught by the std::span overload?
    template <typename T>
    static HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value = 0);
    template <typename T>
    static HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value = 0);

    // getters

    /**
     * From original Tensor:
     * Converts a `Tensor` to a `std::vector<T>`.
     * Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
     * the `Tensor`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
     */
    template <typename T>
    std::vector<T> to_vector() const;

    // TODO: we should just specialize std::to_string and omit this.
    std::string write_to_string() const;

    // TODO: parity with DistributedHostBuffer
    const HostBuffer& get_host_buffer() const;

    // TODO(River): understand what is sharding better
    bool is_sharded() const;
    Shape element_size() const;

    // "other getters"

    // TODO: I want to put this into HostBuffer
    /* DataType dtype() const; */
    const Shape& logical_shape() const;
    const Shape& padded_shape() const;

    const TensorSpec& tensor_spec() const;

    // Can't these be derived from other functions?
    uint64_t logical_volume() const;
    uint64_t physical_volume() const;

    // Can't this be accessed from tensor_spec?
    const MemoryConfig& memory_config() const;

    // TODO: is this appropriate for host tensor?
    /**
     * From original Tensor:
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const;

    // TODO(River): learn this
    const std::optional<ShardSpec> shard_spec() const;
    const std::optional<NdShardSpec> nd_shard_spec() const;

    // "Extra helper functions"
    Shape strides() const;
    bool is_scalar() const;

    // Host buffer is always allocated:
    // bool is_allocated() const;

    // We prob need to leak this for compatability:
    // Buffer* buffer() const;
    //
    // /**
    //  * From original tensor:
    //  *  Returns device `MeshBuffer`.
    //  */
    //  std::shared_ptr<distributed::MeshBuffer> mesh_buffer() const;

    // reshape transformation, mutating version
    // TODO: figure out what we will be doing for reshape
    void reshape(/* */);

    /* with_tensor_topology? */
};

}  // namespace tt::tt_metal

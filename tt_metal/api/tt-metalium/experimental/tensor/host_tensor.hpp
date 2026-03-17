// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/buffer.hpp>

// Tensor related constructs
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>

// It is intentional to not reflect the experimental status of this header in its namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graduate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

// Implementation details for HostTensor
class HostTensorImpl;

/**
 * HostTensor represents a Tensor in host memory.
 * Unlike from MeshTensor, copying a HostTensor will perform a value-copy semantics of the underlying config and
 * HostBuffer. Note that this usually doesn't mean a deep copy of the underlying data, as the HostBuffer is usually
 * a view into a contiguous memory region.
 *
 * HostTensor has limited transformation operations supported (via tensor_apis.hpp), and is intended to be used with
 * MeshTensor for host <-> device communication.
 *
 * Invariants of HostTensor:
 * - Default constructed: This is a valueless state, where any access to any member function outside of assignment and
 *   move construction will be UB. This exists to allow for default constructed HostTensor. Incompatible member function
 *   call to this state is checked by TT_ASSERT (enabled at debug build) in accessors. This is mirrors MeshTensor.
 * - Initialized: The HostTensor holds some tensor configurations and associated HostBuffer.
 */
class HostTensor {
    /*
     * Refactoring Notes:
     * To avoid disruption to existing users, HostTensor will deviate very little from the existing (host) Tensor
     * semantics. The only significant changes are:
     * - Eliminating implicit data movement APIs.
     * - Remove transformation methods like to_layout and pad from the class methods. These seem better as free
     *   functions that operate on a HostTensor than as methods of HostTensor. (Separation of data storage and data
     *   manipulation.) In the existing Tensor, these are already duplicated as both methods and free functions.
     */

public:
    using volume_type = std::uint64_t;

    // Special Member functions

    /**
     * Constructs a host tensor in the default constructed state, acting like a nullptr.
     */
    HostTensor() = default;

    explicit HostTensor(DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology);

    /**
     * Move constructor with new spec and topology.
     * Moves the buffer from other and uses the provided spec/topology.
     * This is meant for transition as TTNN-Tensor current has a two-step construction for HostTensor.
     */
    HostTensor(HostTensor&& other, TensorSpec spec, TensorTopology topology);

    ~HostTensor();

    /**
     * Copy constructor.
     *
     * Semantics:
     * - Tensor Spec and Topology are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor(const HostTensor& other);

    /**
     * Copy assignment operator.
     *
     * Semantics:
     * - Tensor Spec and Topology are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor& operator=(const HostTensor& other);

    /**
     * Move constructor.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor becomes a default-constructed HostTensor.
     */
    HostTensor(HostTensor&& other) noexcept;

    /**
     * Move assignment operator.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor becomes a default-constructed HostTensor.
     */
    HostTensor& operator=(HostTensor&& other) noexcept;

    // End special member functions

    // Factory methods for creating an Engaged HostTensor.

    /**
     * Converts a buffer of elements of type `T` to a `Tensor`.
     * Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
     * elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
     *
     * The data in the buffer is copied into a tensor with host storage.
     */
    template <typename T>
    static HostTensor from_span(std::span<T> buffer, const TensorSpec& spec, T pad_value = 0);

    /**
     * Creates a `Tensor` with storage "borrowed" from the buffer of elements of type `T`.
     *
     * We assume buffer is laid out in row-major order.
     * TODO(#38947): tile parameter should be removed.
     */
    template <typename T>
    static HostTensor from_borrowed_data(
        std::span<T> buffer, const Shape& shape, MemoryPin pin, const std::optional<Tile>& tile = std::nullopt);

    template <typename T>
    static HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value = 0);

    /**
     * From original Tensor:
     * Same as `from_vector`, but takes in an rvalue. No copies will be made, if the target layout is row-major,
     * physical shape matches logical shape, and no type conversion is needed.
     */
    template <typename T>
    static HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value = 0);

    // Getters:

    /**
     * Converts a `Tensor` to a `std::vector<T>`.
     * Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
     * the `Tensor`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
     *
     * pre-condition: The HostTensor must be engaged.
     */
    template <typename T>
    std::vector<T> to_vector() const;

    /**
     * Returns the TensorSpec of the HostTensor.
     *
     * pre-condition: The HostTensor must be engaged.
     */
    const TensorSpec& tensor_spec() const;

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     *
     * pre-condition: The HostTensor must be engaged.
     */
    const TensorTopology& tensor_topology() const;

    /**
     * Returns the DistributedHostBuffer of the HostTensor.
     *
     * pre-condition: The HostTensor must be engaged.
     */
    const DistributedHostBuffer& buffer() const;

    // Derivables:

    DataType dtype() const;
    Layout layout() const;
    const Shape& logical_shape() const;
    const Shape& padded_shape() const;

    volume_type logical_volume() const;
    volume_type physical_volume() const;

    const MemoryConfig& memory_config() const;
    bool is_sharded() const;

    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    const std::optional<ShardSpec>& legacy_shard_spec() const;
    const std::optional<NdShardSpec>& nd_shard_spec() const;

    // Utils:

    // Get the element size in bytes
    std::size_t element_size() const;

    Strides strides() const;

    // Questionables:

    // Applies a transformation function to each host buffer across devices in parallel, returning a new HostTensor.
    HostTensor transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const;

    // void update_tensor_topology(TensorTopology tensor_topology);

private:
    std::unique_ptr<HostTensorImpl> impl;
};

}  // namespace tt::tt_metal

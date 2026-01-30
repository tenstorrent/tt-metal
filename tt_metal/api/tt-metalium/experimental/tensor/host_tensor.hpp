// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/buffer.hpp>

// Tensor related constructs
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/memory_pin.hpp>

// It is intentional to not reflect the experimental status of this header in it's namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graudate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

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
    using volumn_type = std::uint64_t;

    // Special Member functions

    /**
     * Constructs an empty host tensor.
     *
     * The empty tensor has the following properties:
     * - logical_shape(): Empty shape
     * - padded_shape(): Empty shape
     * - strides(): Empty
     * - dtype(): DataType::INVALID
     * - layout(): Layout::INVALID
     * - get_host_buffer(): Empty buffer
     * - element_size(): Throws an exception (invalid data type)
     */
    HostTensor();
    ~HostTensor();

    /**
     * Copy constructor.
     *
     * Semantics:
     * - Configs are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor(const HostTensor&);

    /**
     * Copy assignment operator.
     *
     * Semantics:
     * - Configs are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor& operator=(const HostTensor&);

    /**
     * Move constructor.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor has the same state as an default-constructed HostTensor.
     *
     * TODO: Is this over specifiying the move-from object? We might not be able to guarantee the noexcept here.
     */
    HostTensor(HostTensor&& other) noexcept;

    /**
     * Move assignment operator.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor has the same state as an default-constructed HostTensor.
     */
    HostTensor& operator=(HostTensor&& other) noexcept;

    // End special member functions

    // constructions:

    // TODO: What are the assumptions with these buffers?

    // TODO: Original Tensor has three overloads
    // TODO-ask alex: Original Tensor takes HostBuffer as value, is that a must?
    // Make this private + the main constructor?
    explicit HostTensor(const HostBuffer& buffer, TensorSpec spec, TensorTopology topology);

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
    // River: oh it's because the overload should be std::span<const T>
    template <typename T>
    static HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value = 0);

    /**
     * From original Tensor:
     * Same as `from_vector`, but takes in an rvalue. No copies will be made, if the target layout is row-major,
     * physical shape matches logical shape, and no type conversion is needed.
     */
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
    HostBuffer get_host_buffer() const;

    // TODO(River): understand what is sharding better
    bool is_sharded() const;
    // Get the dataum's size in bytes
    std::size_t element_size() const;

    // "other getters"

    // TODO: I want to put this into HostBuffer
    DataType dtype() const;
    Layout layout() const;
    const Shape& logical_shape() const;
    const Shape& padded_shape() const;

    const TensorSpec& tensor_spec() const;

    // Can't these be derived from other functions?
    volumn_type logical_volume() const;
    // This was called "physical_volumn", renaming here to be consistent with `padded_shape`.
    volumn_type padded_volume() const;

    // Can't this be accessed from tensor_spec?
    const MemoryConfig& memory_config() const;

    // TODO: is this appropriate for host tensor?
    /**
     * From original Tensor:
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const;

    // TODO(River): learn this
    // From original Tensor:
    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    // TODO: Is there a way to express this "either or"?
    const std::optional<ShardSpec>& shard_spec() const;
    const std::optional<NdShardSpec>& nd_shard_spec() const;

    // "Extra helper functions"
    // Shape is a weird class to return, isn't a vector sufficient?
    Strides strides() const;

    // Host buffer is always allocated:
    // bool is_allocated() const;

    // // From orignal Tensor:
    // // Returns device `Buffer`.
    // // Throws if the tensor is not allocated on a device.
    // Buffer* buffer() const;
    //
    // Host Tensor doesn't have an associated device,
    // I don't think this is needed?
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

private:
    std::unique_ptr<TensorAttributes> impl;
};

}  // namespace tt::tt_metal

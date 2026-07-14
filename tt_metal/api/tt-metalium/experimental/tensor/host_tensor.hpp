// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
 * HostTensor represents a Tensor in host memory. It is intended to be used with MeshTensor for host <-> device
 * communication, and has a limited set of transformation operations supported (via tensor_apis.hpp).
 *
 * Invariants of HostTensor:
 * - The DistributedHostBuffer data is laid out in a way that conforms to the TensorSpec.
 *
 * Notes:
 * - HostTensor is copyable with value semantics: TensorSpec and TensorTopology are deep-copied; the
 *   DistributedHostBuffer follows its own copy semantics, which is typically a shallow copy (view into the
 *   same underlying data).
 * - Unlike MeshTensor, HostTensor does not have unique ownership semantics.
 *
 * Note: A moved-from HostTensor is in a valid but unspecified state. All member functions except destruction and
 * assignment will fail on a moved-from instance.
 */
class HostTensor {
public:
    using volume_type = std::uint64_t;

    // Special Member functions

    HostTensor() = delete;

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
     * The other HostTensor is left in a valid but unspecified state.
     */
    HostTensor(HostTensor&& other) noexcept;

    /**
     * Move assignment operator.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor is left in a valid but unspecified state.
     */
    HostTensor& operator=(HostTensor&& other) noexcept;

    // End special member functions

    // Factory methods for creating an Engaged HostTensor.

    /**
     * Constructs a host tensor from a distributed host buffer.
     */
    static HostTensor from_buffer(DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology);

    /**
     * Constructs a host tensor from a single device host buffer.
     * The buffer occupies the 0x0 shard of the distributed host buffer.
     */
    static HostTensor from_buffer(HostBuffer buffer, TensorSpec spec, TensorTopology topology);

    /**
     * Converts a buffer of elements of type `T` to a `Tensor`.
     * Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
     * elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
     *
     * The data in the buffer is copied into a tensor with host storage.
     */
    template <typename T>
    static HostTensor from_span(std::span<const T> buffer, const TensorSpec& spec, T pad_value = 0);

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
     */
    template <typename T>
    std::vector<T> to_vector() const;

    /**
     * Returns the TensorSpec of the HostTensor.
     */
    const TensorSpec& tensor_spec() const;

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const;

    /**
     * Returns true if this HostTensor was left in a moved-from state.
     *
     * A HostTensor becomes valueless when it is the source of a move construction or move assignment.
     * Unlike every other member function (except destruction and assignment), this function is safe to
     * call on a moved-from instance; it is in fact the intended way to detect that state.
     */
    bool is_valueless_after_move() const;

    /**
     * Returns the DistributedHostBuffer of the HostTensor.
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

    // Applies a transformation function to each host buffer across devices in parallel, returning a new HostTensor.
    HostTensor transform(const std::function<HostBuffer(const HostBuffer&)>& callable) const;

    // Updates the topology of the HostTensor post construction.
    void update_tensor_topology(TensorTopology tensor_topology);

    /**
     * Access to the implementation.
     *
     * pre-condition: The HostTensor must be initialized.
     */
    HostTensorImpl& impl();
    const HostTensorImpl& impl() const;

private:
    // Internal constructors. Use the from_buffer factories to build a HostTensor from a backing buffer.
    explicit HostTensor(DistributedHostBuffer buffer, TensorSpec spec, TensorTopology topology);

    // impl_ could be a nullptr if HostTensor is in a moved-from state.
    // Avoid using impl_ pointer directly, use the impl() accessor instead.
    // Otherwise, please add manual TT_FATAL checks for nullptr.
    std::unique_ptr<HostTensorImpl> impl_;
};

}  // namespace tt::tt_metal

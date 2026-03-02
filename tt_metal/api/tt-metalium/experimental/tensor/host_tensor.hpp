// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>

// It is intentional to not reflect the experimental status of this header in it's namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graduate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

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

    using attribute_type = TensorImpl<HostStorage>;

public:
    using volume_type = std::uint64_t;

    // Special Member functions

    /**
     * Constructs a host tensor in the default constructed state, acting like a nullptr.
     */
    HostTensor() = default;

    // TODO(#38376), TODO(#38689):
    // These constructors should be hidden or go away.
    // External user should not be able to construct a HostTensor directly and opt to use the from_xxx static methods
    // instead, as the constructor does not perform invariant checks.
    explicit HostTensor(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
        impl(std::make_unique<attribute_type>(std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {
    }

    explicit HostTensor(HostBuffer buffer, TensorSpec spec, TensorTopology topology) :
        impl(std::make_unique<attribute_type>(HostStorage(std::move(buffer)), std::move(spec), std::move(topology))) {}

    ~HostTensor() = default;

    /**
     * Copy constructor.
     *
     * Semantics:
     * - Tensor Spec and Topology are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor(const HostTensor& other) : impl(other.impl ? std::make_unique<attribute_type>(*other.impl) : nullptr) {}

    /**
     * Copy assignment operator.
     *
     * Semantics:
     * - Tensor Spec and Topology are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor& operator=(const HostTensor& other) {
        if (this == &other) {
            return *this;
        }
        impl = other.impl ? std::make_unique<attribute_type>(*other.impl) : nullptr;
        return *this;
    }

    /**
     * Move constructor.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor becomes a default-constructed HostTensor.
     */
    HostTensor(HostTensor&& other) noexcept : impl(std::move(other.impl)) {}

    /**
     * Move assignment operator.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor becomes a default-constructed HostTensor.
     */
    HostTensor& operator=(HostTensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        impl = std::move(other.impl);
        return *this;
    }

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
     */
    template <typename T>
    static HostTensor from_borrowed_data(std::span<T> buffer, const Shape& shape, MemoryPin pin);

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
    const TensorSpec& tensor_spec() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "HostTensor is in a default constructed state");
        return impl->tensor_spec_;
    }

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     *
     * pre-condition: The HostTensor must be engaged.
     */
    const TensorTopology& tensor_topology() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "HostTensor is in a default constructed state");
        return impl->tensor_topology_;
    }

    // DeviceStorage is meant to bridge ttnn::Tensor and HostTensor,
    // this should go away as part of refactoring, see: #38376
    const HostStorage& get_legacy_host_storage() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "HostTensor is in a default constructed state");
        return impl->storage_;
    }

    // Use host_buffer::get_host_buffer instead.
    // HostBuffer get_host_buffer() const;

    /**
     * Returns the DistributedHostBuffer of the HostTensor.
     *
     * pre-condition: The HostTensor must be engaged.
     */
    const DistributedHostBuffer& get_distributed_host_buffer() const { return get_legacy_host_storage().buffer(); }

    // Derivables:

    DataType dtype() const { return tensor_spec().tensor_layout().get_data_type(); }
    Layout layout() const { return tensor_spec().tensor_layout().get_layout(); }
    const Shape& logical_shape() const { return tensor_spec().logical_shape(); }
    const Shape& padded_shape() const { return tensor_spec().padded_shape(); }

    volume_type logical_volume() const { return logical_shape().volume(); }
    volume_type physical_volume() const { return padded_shape().volume(); }

    const MemoryConfig& memory_config() const { return tensor_spec().memory_config(); }
    bool is_sharded() const { return tensor_spec().memory_config().is_sharded(); }

    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    const std::optional<ShardSpec>& legacy_shard_spec() const { return memory_config().shard_spec(); }
    const std::optional<NdShardSpec>& nd_shard_spec() const { return memory_config().nd_shard_spec(); }

    // Utils:

    // Get the dataum's size in bytes
    std::size_t element_size() const {
        // this might be better?
        // return datum_size(datatype_to_dataformat_converter(dtype()));
        switch (dtype()) {
            case DataType::BFLOAT16: return sizeof(bfloat16);
            case DataType::FLOAT32: return sizeof(float);
            case DataType::INT32: return sizeof(int32_t);
            case DataType::UINT32: return sizeof(uint32_t);
            case DataType::UINT16: return sizeof(uint16_t);
            case DataType::UINT8: return sizeof(uint8_t);
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B: return sizeof(std::byte);
            default: TT_THROW("Unsupported data type");
        }
    }

    Strides strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

    // Questionables:

    HostTensor with_tensor_topology(TensorTopology tensor_topology) const {
        return HostTensor(get_legacy_host_storage(), tensor_spec(), std::move(tensor_topology));
    }

private:
    std::unique_ptr<attribute_type> impl;
};

}  // namespace tt::tt_metal

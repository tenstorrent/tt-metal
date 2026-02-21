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

// TODO: Candidate for moving to details folder after refactoring.
#include <tt-metalium/experimental/tensor/details/tensor_attributes.hpp>

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
    using attribute_type = TensorAttributes<HostStorage>;

public:
    using volumn_type = std::uint64_t;

    // Special Member functions

    /**
     * Constructs an empty host tensor, acts as a nullptr.
     */
    HostTensor() = default;
    ~HostTensor() = default;

    /**
     * Copy constructor.
     *
     * Semantics:
     * - Configs are deep copied.
     * - Underlying data has the copy semantics of the HostBuffer
     */
    HostTensor(const HostTensor& other) : impl(other.impl ? std::make_unique<attribute_type>(*other.impl) : nullptr) {}

    /**
     * Copy assignment operator.
     *
     * Semantics:
     * - Configs are deep copied.
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
     * The other HostTensor has the same state as an default-constructed HostTensor.
     */
    HostTensor(HostTensor&& other) noexcept : impl(std::move(other.impl)) {}

    /**
     * Move assignment operator.
     *
     * Takes over properties of the other HostTensor.
     * The other HostTensor has the same state as an default-constructed HostTensor.
     */
    HostTensor& operator=(HostTensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        impl = std::move(other.impl);
        return *this;
    }

    // End special member functions

    // constructions:

    // Make this private + the main constructor?
    explicit HostTensor(HostBuffer buffer, TensorSpec spec, TensorTopology topology) :
        impl(std::make_unique<attribute_type>(HostStorage(std::move(buffer)), std::move(spec), std::move(topology))) {}

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

    /**
     * From original Tensor:
     * Creates a `Tensor` with storage "borrowed" from the buffer of elements of type `T`.
     *
     * Edit from original Tensor:
     * - Remove tile parameter as it doesn't make sense for ROW_MAJOR layout.
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

    HostBuffer get_host_buffer() const {
        // TODO: figure out if we're doing DistributedHostBuffer, hardcoding (0,0) is horrifying.
        // Get shard at (0,0) for single-device tensor - always exists for HostTensor
        auto buffer = get_distributed_host_buffer().get_shard(distributed::MeshCoordinate(0, 0));
        TT_ASSERT(buffer.has_value(), "HostTensor must have a buffer at coordinate (0, 0)");
        return *buffer;
    }

    const DistributedHostBuffer& get_distributed_host_buffer() const { return get_storage().buffer(); }

    bool is_sharded() const {
        // TODO: this is technically divergent from ttnn::Tensor
        return tensor_spec().memory_config().is_sharded();
    }

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

    // "other getters"

    // TODO: I want to put this into HostBuffer
    DataType dtype() const { return tensor_spec().tensor_layout().get_data_type(); }
    Layout layout() const { return tensor_spec().tensor_layout().get_layout(); }
    const Shape& logical_shape() const { return tensor_spec().logical_shape(); }
    const Shape& padded_shape() const { return tensor_spec().padded_shape(); }

    const TensorSpec& tensor_spec() const { return impl->tensor_spec_; }

    // Can't these be derived from other functions?
    volumn_type logical_volume() const { return logical_shape().volume(); }
    // This was called "physical_volumn", renaming here to be consistent with `padded_shape`.
    volumn_type padded_volume() const { return padded_shape().volume(); }

    // Can't this be accessed from tensor_spec?
    const MemoryConfig& memory_config() const { return tensor_spec().memory_config(); }

    /**
     * From original Tensor:
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const { return impl->tensor_topology_; }

    // From original Tensor:
    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    // TODO: Is there a way to express this "either or"?
    const std::optional<ShardSpec>& shard_spec() const { return memory_config().shard_spec(); }
    const std::optional<NdShardSpec>& nd_shard_spec() const { return memory_config().nd_shard_spec(); }

    // "Extra helper functions"
    Strides strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

    // TODO: Remove these after refactoring.
    HostTensor(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
        impl(std::make_unique<attribute_type>(std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {
    }

    // TODO: Does this make sense for HostTensor?
    HostTensor with_tensor_topology(TensorTopology tensor_topology) const {
        return HostTensor(get_storage(), tensor_spec(), std::move(tensor_topology));
    }

    const HostStorage& get_storage() const { return impl->storage_; }

private:
    std::unique_ptr<attribute_type> impl;
};

}  // namespace tt::tt_metal

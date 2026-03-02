// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>

#include <tt_stl/optional_reference.hpp>

// It is intentional to not reflect the experimental status of this header in it's namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graduate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

class MeshTensor;
// To be removed after #37807
namespace do_not_use {
void do_not_use_update_mesh_tensor_storage(MeshTensor&, const DeviceStorage&);
}

namespace distributed {
class MeshDevice;
}

/**
 * MeshTensor is a device memory object. The user’s mental model of MeshTensor is an owning handle to
 * device-allocated memory.
 *
 * MeshTensor should have RAII semantics with unique ownership:
 * - Device memory resource lifetime == object lifetime
 *   - Device memory is allocated on construction, and released on destruction.
 *   - The programmer explicitly manages the device-allocated memory lifetime.
 *   - This can be tricky in an asynchronous runtime environment. For now, the onus is on the programmer to correctly
 *     manage MeshTensor lifetime around queue synchronization events.
 * - Movable (RAII transfer of ownership)
 * - Non-copyable
 * - No equality/inequality operator. (If we did add this, equality would mean the same underlying allocation – no value
 *   semantics)
 *
 * Invariants of MeshTensor:
 * - Default constructed: This is a valueless state, where any access to any member function outside of assignment and
 *   move construction will be UB. This exists to allow for default constructed MeshTensor. Incompatible member function
 *   call to this state is checked by TT_ASSERT (enabled at debug build) in accessors. This is mirrors HostTensor.
 * - Allocated: The device memory is allocated and **solely owned** by MeshTensor, user is able to get non-null
 *   pointers to the underlying storage and associated MeshDevice. Please note that this invariant isn't guaranteed
 *   currently, see: #38375
 * - Deallocated: The device memory is deallocated and the MeshTensor is in a default constructed state, pointer to
 *   Device and MeshBuffer will be null.
 */
class MeshTensor {
    using attribute_type = TensorImpl<DeviceStorage>;

public:
    using volume_type = std::uint64_t;

    // Special Member functions

    /**
     * Construct a tensor that does not own any device memory.
     */
    MeshTensor() = default;

    // TODO(#38376), TODO(#38689):
    // This should be a private constructor, external user should not be able to construct a MeshTensor
    // directly. As this will lead to leaks of the MeshBuffer unique ownership.
    explicit MeshTensor(DeviceStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
        impl(std::make_unique<attribute_type>(std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {
    }

    // Factory methods

    /**
     * Allocate device memory for a tensor with the given tensor spec on the given mesh device.
     * Returns a MeshTensor that owns the allocated device memory.
     */
    static MeshTensor allocate_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

    /**
     * Release ownership of the underlying device memory.
     * Whether or not the device memory is actually deallocated depends on the destructor semantics of the underlying
     * MeshBuffer.
     */
    ~MeshTensor() = default;

    /**
     * A device tensor is non-copyable as this is the sole owner of the underlying device memory.
     */
    MeshTensor(const MeshTensor&) = delete;

    /**
     * A device tensor is non-copyable as this is the sole owner of the underlying device memory.
     */
    MeshTensor& operator=(const MeshTensor&) = delete;

    /**
     * Transfer ownership of the underlying device memory to the other MeshTensor.
     *
     * post-condition: The other MeshTensor will be in a default constructed state.
     */
    MeshTensor(MeshTensor&& other) = default;

    /**
     * Transfer ownership of the underlying device memory to the other MeshTensor.
     *
     * post-condition: The other MeshTensor will be in a default constructed state.
     */
    MeshTensor& operator=(MeshTensor&& other) = default;

    // End speical member functions

    // Deallocation related:

    /**
     * Release ownership of the underlying device memory.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    void deallocate() {
        TT_ASSERT(impl != nullptr, "MeshTensor is in a default constructed state");
        auto& device_storage = impl->storage_;
        // This implicitly deallocates the root mesh buffer if we are the sole owner.
        // An explicit deallocation call is not performed, as current day MeshBuffer could still be shared by other
        // owners. See: #38375
        device_storage.reset_root_mesh_buffer();
    }

    /**
     * Check if the device tensor owns any device memory.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    bool is_allocated() const { return mesh_buffer().has_value(); }

    /**
     * Return the underlying device storage MeshBuffer.
     * empty optional when the device tensor is in deallocated state via deallocate().
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    ttsl::optional_reference<distributed::MeshBuffer> mesh_buffer() const {
        if (auto ptr = mesh_buffer_invariant_breaking()) {
            return (*ptr);
        }
        return std::nullopt;
    }

    /**
     * Wider API compatible mesh_buffer() that returns a shared ownership to the underlying storage.
     *
     * Note: Prefer mesh_buffer() wherever possible, as it breaks unique ownership semantics easily.
     * A core invariant of MeshTensor is that it is the sole owner of the underlying MeshBuffer,
     * one can get the underlying shared_ptr of the MeshBuffer and break the invariant.
     *
     * See: #38691, #38375
     */
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_invariant_breaking() const {
        if (const auto& mesh_buffer = get_legacy_device_storage().mesh_buffer; mesh_buffer != nullptr) {
            return mesh_buffer;
        }
        return nullptr;
    }

    /**
     * Get the device the allocated device memory is on.
     * Returns an empty optional when owned device memory is released via deallocate().
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    ttsl::optional_reference<distributed::MeshDevice> get_device() const {
        if (auto buffer = mesh_buffer()) {
            return *buffer->device();
        }
        return std::nullopt;
    }

    // Getters:

    const TensorSpec& tensor_spec() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "MeshTensor is in a default constructed state");
        return impl->tensor_spec_;
    }

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    const TensorTopology& tensor_topology() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "MeshTensor is in a default constructed state");
        return impl->tensor_topology_;
    }

    // DeviceStorage is meant to bridge ttnn::Tensor and MeshTensor,
    // this should go away as part of refactoring, see: #38376
    const DeviceStorage& get_legacy_device_storage() const {
        // Pre-condition
        TT_ASSERT(impl != nullptr, "MeshTensor is in a default constructed state");
        return impl->storage_;
    }

    // Derivables:

    DataType dtype() const { return tensor_spec().data_type(); }
    Layout layout() const { return tensor_spec().layout(); }
    const Shape& logical_shape() const { return tensor_spec().logical_shape(); }
    const Shape& padded_shape() const { return tensor_spec().padded_shape(); }

    volume_type logical_volume() const { return logical_shape().volume(); }
    volume_type physical_volume() const { return padded_shape().volume(); }

    const MemoryConfig& memory_config() const { return tensor_spec().memory_config(); }
    bool is_sharded() const { return memory_config().is_sharded(); }

    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    const std::optional<ShardSpec>& legacy_shard_spec() const { return memory_config().shard_spec(); }
    const std::optional<NdShardSpec>& nd_shard_spec() const { return memory_config().nd_shard_spec(); }

    // Utils:

    /**
     * Get the size in bytes of a single element held in the tensor.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    std::size_t element_size() const {
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

    // TODO(#38693):
    // This is a hack right now, because this allows multiple MeshTensor holding on to the same MeshBuffer,
    // we need to find an alternative way to do this.
    MeshTensor with_tensor_topology(TensorTopology tensor_topology) const {
        return MeshTensor(get_legacy_device_storage(), tensor_spec(), std::move(tensor_topology));
    }

private:
    // impl could be a nullptr if MeshTensor is in a default constructed state.
    // Avoid using impl pointer directly, use the accessors instead.
    // Otherwise, please add manual TT_ASSERT checks for nullptr.
    std::unique_ptr<attribute_type> impl;

    friend void do_not_use::do_not_use_update_mesh_tensor_storage(MeshTensor&, const DeviceStorage&);
};

}  // namespace tt::tt_metal

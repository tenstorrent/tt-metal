// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/details/tensor_attributes.hpp>

// It is intentional to not reflect the experimental status of this header in it's namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graudate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}

/**
 * DeviceTensor is a device memory object. The user’s mental model of DeviceTensor is an owning handle to
 * device-allocated memory.
 *
 * DeviceTensor should have RAII semantics with unique ownership:
 * - Device memory resource lifetime == object lifetime
 *   - Device memory is allocated on construction, and released on destruction.
 *   - The programmer explicitly manages the device-allocated memory lifetime.
 *   - This can be tricky in an asynchronous runtime environment. For now, the onus is on the programmer to correctly
 *     manage DeviceTensor lifetime around queue synchronization events.
 * - Movable (RAII transfer of ownership)
 * - Non-copyable
 * - No equality/inequality operator. (If we did add this, equality would mean the same underlying allocation – no value
 *   semantics)
 *
 */
class DeviceTensor {
    // TODO: internal constructor
public:
    using volumn_type = std::uint64_t;

    // Special Member functions

    /**
     * Construct a tensor that does not own any device memory.
     */
    DeviceTensor() = default;

    // TODO: Close this constructor after refactoring.
    DeviceTensor(DeviceStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
        impl(std::make_unique<TensorAttributes>(
            std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {}

    /**
     * Deallocates any owning device memory.
     */
    ~DeviceTensor() = default;

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // Transfers ownership of other's memory
    DeviceTensor(DeviceTensor&& other) = default;
    DeviceTensor& operator=(DeviceTensor&& other) = default;

    // End speical member functions

    /**
     * Deallocate and release owned device memory.
     */
    void deallocate() {
        // GraphTracker::instance().track_function_start("Tensor::deallocate", *this, force);
        auto& device_storage = get_storage();
        device_storage.mesh_buffer->deallocate();
        device_storage.mesh_buffer.reset();
        // GraphTracker::instance().track_function_end();
    }

    // Getters

    /**
     * Get the device this DeviceTensor is on.
     *
     * nullptr when deallocated.
     */
    // TODO: make this optional_ref?
    distributed::MeshDevice* get_device() const {
        if (const auto& mesh_buffer = get_storage().mesh_buffer; mesh_buffer != nullptr) {
            return mesh_buffer->device();
        }
        return nullptr;
    }

    // TODO: Should we make this mean something?
    std::string write_to_string() const;

    bool is_sharded() const { return memory_config().is_sharded(); }
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

    // "misc getters"
    DataType dtype() const { return tensor_spec().data_type(); }
    Layout layout() const { return tensor_spec().layout(); }
    const Shape& logical_shape() const { return tensor_spec().logical_shape(); }
    const Shape& padded_shape() const { return tensor_spec().padded_shape(); }

    const TensorSpec& tensor_spec() const { return impl->get_tensor_spec(); }

    volumn_type logical_volume() const { return logical_shape().volume(); }
    volumn_type physical_volume() const { return padded_shape().volume(); }

    const MemoryConfig& memory_config() const { return tensor_spec().memory_config(); }

    /**
     * From original Tensor:
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const { return impl->get_tensor_topology(); }

    // From original Tensor:
    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    // TODO: Is there a way to express this "either or"?
    const std::optional<ShardSpec>& shard_spec() const { return memory_config().shard_spec(); }
    const std::optional<NdShardSpec>& nd_shard_spec() const { return memory_config().nd_shard_spec(); }

    Strides strides() const { return tensor_spec().tensor_layout().compute_strides(logical_shape()); }

    // TODO: this is implemented dependening on both if we have released the buffer and if the buffer is deallocated.
    //     Who would dealloate the buffer?
    bool is_allocated() const {
        if (const auto& mesh_buffer = get_storage().mesh_buffer; mesh_buffer != nullptr) {
            return mesh_buffer->is_allocated();
        }
        return false;
    }

    // We prob need to leak this for compatability:
    //
    // // TODO-ask Alex: can we retire this method in favor of mesh_buffer()
    // Buffer* buffer() const;
    //
    // /**
    //  * From original tensor:
    //  *  Returns device `MeshBuffer`.
    //  */
    //  std::shared_ptr<distributed::MeshBuffer> mesh_buffer() const;

    const DeviceStorage& get_storage() const { return std::get<DeviceStorage>(impl->get_storage()); }

    // TODO: This is a hack right now, because this allows multiple device tensor holding on to the same conceptual
    // storage, find a better way to do this.
    DeviceTensor with_tensor_topology(TensorTopology tensor_topology) {
        return DeviceTensor(get_storage(), tensor_spec(), std::move(tensor_topology));
    }

private:
    std::unique_ptr<TensorAttributes> impl;

    DeviceStorage& get_storage() { return std::get<DeviceStorage>(impl->get_storage()); }
};

}  // namespace tt::tt_metal

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

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
public:
    using volumn_type = std::uint64_t;

    // Special Member functions

    /**
     * Construct a tensor that does not own any device memory.
     */
    DeviceTensor() = default;
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
    void deallocate(/* bool force = false */);

    // reshape transformation, mutating version
    // TODO: figure out what we will be doing for reshape
    void reshape(/* */);

    // ?
    /* with_tensor_topology(TensorTopology tensor_topology) */

    // Getters

    /**
     * Get the device this DeviceTensor is on.
     *
     * throws or nullptr when deallocated?
     */
    distributed::MeshDevice& get_device() const;

    // TODO: Should we make this mean something?
    std::string write_to_string() const;

    // TODO(River): understand what is sharding better
    bool is_sharded() const;
    std::size_t element_size() const;

    // "misc getters"
    DataType dtype() const;
    Layout layout() const;
    const Shape& logical_shape() const;
    const Shape& padded_shape() const;

    const TensorSpec& tensor_spec() const;

    // Can't these be derived from other functions?
    volumn_type logical_volume() const;
    volumn_type physical_volume() const;

    // Can't this be accessed from tensor_spec?
    const MemoryConfig& memory_config() const;

    /**
     * From original Tensor:
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const;

    // TODO(River): learn this
    // From original Tensor:
    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    // TODO: Is there a way to express this "either or"?
    const std::optional<ShardSpec> shard_spec() const;
    const std::optional<NdShardSpec> nd_shard_spec() const;

    // Shape is a weird class to return, isn't a vector sufficient?
    Shape strides() const;
    // Do we need this? This was meant to be pair with item() which is removed?
    bool is_scalar() const;

    // TODO: Would this be better if it's called is_deallocated
    bool is_allocated() const;

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
};

}  // namespace tt::tt_metal

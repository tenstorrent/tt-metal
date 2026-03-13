// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/hal_types.hpp>

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

#include <tt_stl/optional_reference.hpp>

// It is intentional to not reflect the experimental status of this header in its namespace,
// as most of the code movements are based on implementations in TTNN that are well tested and production ready for a
// long time, it is expected for the implementation to graduate out of experimental really quickly.
//
// Using namespace tt::tt_metal avoids double namespace renaming for the refactoring effort.
namespace tt::tt_metal {

// This will be brought in at #37692
class DeviceStorage;

// Implementation details for MeshTensor
class MeshTensorImpl;

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
 *   - This can be tricky in an asynchronous runtime environment. For now, the focus is on the programmer to correctly
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
 */
class MeshTensor {
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
    explicit MeshTensor(DeviceStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);

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

    // End special member functions

    // Deallocation related:

    /**
     * Check if the device tensor owns any device memory.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    bool is_allocated() const;

    /**
     * Return the underlying device storage MeshBuffer.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    distributed::MeshBuffer& mesh_buffer() const;

    /**
     * Wider API compatible mesh_buffer() that returns a shared ownership to the underlying storage.
     *
     * Note: Prefer mesh_buffer() wherever possible, as it breaks unique ownership semantics easily.
     * A core invariant of MeshTensor is that it is the sole owner of the underlying MeshBuffer,
     * one can get the underlying shared_ptr of the MeshBuffer and break the invariant.
     *
     * See: #38691, #38375
     */
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_invariant_breaking() const;

    /**
     * Get the device the allocated device memory is on.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    distributed::MeshDevice& device() const;

    // Getters:

    const TensorSpec& tensor_spec() const;

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    const TensorTopology& tensor_topology() const;

    // DeviceStorage is meant to bridge ttnn::Tensor and MeshTensor,
    // this should go away as part of refactoring, see: #38376
    const DeviceStorage& get_legacy_device_storage() const;

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

    DeviceAddr address() const;

    // Utils:

    /**
     * Get the size in bytes of a single element held in the tensor.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    std::size_t element_size() const;

    Strides strides() const;

    void update_tensor_topology(TensorTopology tensor_topology);

private:
    // impl could be a nullptr if MeshTensor is in a default constructed state.
    // Avoid using impl pointer directly, use the accessors instead.
    // Otherwise, please add manual TT_ASSERT checks for nullptr.
    std::unique_ptr<MeshTensorImpl> impl;
};

}  // namespace tt::tt_metal

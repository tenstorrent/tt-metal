// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// Implementation details for MeshTensor
class MeshTensorImpl;
struct DeviceStorage;

namespace distributed {
class MeshDevice;
}

/**
 * MeshTensor is a device memory object. The user’s mental model of MeshTensor is an owning handle to
 * device-allocated memory.
 *
 * Invariants of MeshTensor:
 * - MeshTensor is the sole owner of the underlying device memory.
 * - MeshTensor object lifetime is the same as the underlying device memory lifetime. An instance of MeshTensor maps to
 * a single allocated device memory.
 * - The underlying device memory is always allocated, large enough to hold the tensor, and laid out in a way
 *   that conforms to the TensorSpec (page size, buffer type, memory layout).
 *
 * Notes:
 * - MeshTensor is non-copyable but movable due to the unique ownership model.
 * - To "deallocate" a MeshTensor, simply let the MeshTensor go out of scope.
 * - The programmer is responsible for managing MeshTensor lifetime around queue synchronization events,
 *   which can be tricky in an asynchronous runtime environment.
 * - There is no invariant between a MeshTensor and it's associated TensorTopology.
 *
 * Note: A moved-from MeshTensor is in a valid but unspecified state. All member functions except destruction and
 * assignment will fail on a moved-from instance.
 */
class MeshTensor {
public:
    using volume_type = std::uint64_t;

    // Special Member functions

    MeshTensor() = delete;

    /**
     * Allocate a MeshTensor on the given device with the given spec and topology.
     */
    static MeshTensor allocate_on_device(
        distributed::MeshDevice& mesh_device, const TensorSpec& spec, const TensorTopology& topology);

    // Internal Constructor for transition.
    explicit MeshTensor(std::shared_ptr<distributed::MeshBuffer> mesh_buffer, TensorSpec spec, TensorTopology topology);

    /**
     * Release ownership of the underlying device memory.
     * Whether or not the device memory is actually deallocated depends on the destructor semantics of the underlying
     * MeshBuffer.
     */
    ~MeshTensor();

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
     * post-condition: The moved-from MeshTensor is left in a valid but unspecified state.
     */
    MeshTensor(MeshTensor&& other) noexcept;

    /**
     * Transfer ownership of the underlying device memory to the other MeshTensor.
     *
     * post-condition: The moved-from MeshTensor is left in a valid but unspecified state.
     */
    MeshTensor& operator=(MeshTensor&& other) noexcept;

    // End special member functions

    /**
     * Return the underlying device storage MeshBuffer.
     */
    const distributed::MeshBuffer& mesh_buffer() const;

    /**
     * Get the device the allocated device memory is on.
     */
    const distributed::MeshDevice& device() const;

    /**
     * Get the mutable device the allocated device memory is on.
     *
     * This function is meant to be compatible with existing code and may be removed in the future,
     * please consider this an internal function and use device() whenever possible.
     *
     * pre-condition: The device tensor must not be in a default constructed state.
     */
    distributed::MeshDevice& mutable_device() const;

    // Getters:

    const TensorSpec& tensor_spec() const;

    /**
     * Multi-device topology configuration - tracks how tensor is distributed across mesh devices
     */
    const TensorTopology& tensor_topology() const;

    /**
     * Returns true if this MeshTensor was left in a moved-from state.
     *
     * A MeshTensor becomes valueless when it is the source of a move construction or move assignment.
     * Unlike every other member function (except destruction and assignment), this function is safe to
     * call on a moved-from instance; it is in fact the intended way to detect that state.
     */
    bool is_valueless_after_move() const;

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
    const std::optional<ShardSpec>& shard_spec() const;
    const std::optional<NdShardSpec>& nd_shard_spec() const;

    DeviceAddr address() const;

    // Utils:

    /**
     * Get the size in bytes of a single element held in the tensor.
     */
    std::size_t element_size() const;

    Strides strides() const;

    /**
     * Update the topology of the MeshTensor post construction.
     */
    void update_tensor_topology(TensorTopology tensor_topology);

    /**
     * Access to the implementation.
     *
     * pre-condition: The MeshTensor must not be in a default constructed state.
     */
    MeshTensorImpl& impl();
    const MeshTensorImpl& impl() const;

private:
    // TODO(#43693): Remove once DeviceStorage no longer keeps a shared_ptr<MeshBuffer>
    // in its DeallocatedTombStone state.
    // DeviceStorage is the sole caller — it uses mesh_buffer_invariant_breaking() to
    // populate the tombstone when a tensor is deallocated, so the device pointer
    // remains accessible on aliased tensors (e.g. after a zero-copy reshape).
    // This breaks MeshTensor's core invariant: that it is the sole owner of the
    // underlying MeshBuffer. Shared ownership leaks out, allowing the MeshBuffer to
    // outlive the MeshTensor.
    friend struct DeviceStorage;

    /**
     * Returns shared ownership of the underlying MeshBuffer.
     *
     * WARNING: Breaks MeshTensor's sole-ownership invariant. The caller becomes a
     * shared owner of the MeshBuffer, which can outlive the MeshTensor.
     * Only accessible to DeviceStorage. See #43693 for removal plan.
     */
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_invariant_breaking() const;

    // impl_ could be a nullptr if MeshTensor is in a moved-from state.
    // Avoid using impl_ pointer directly, use the impl() accessor instead.
    // Otherwise, please add manual TT_FATAL checks for nullptr.
    std::unique_ptr<MeshTensorImpl> impl_;
};

}  // namespace tt::tt_metal

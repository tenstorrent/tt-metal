// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include "impl/buffers/semaphore.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

class Buffer;
class GlobalSemaphore;
class CircularBufferImpl;
class Kernel;
class MeshTensor;
struct KernelSource;
struct EthernetConfig;
struct DramConfig;

namespace distributed {
class MeshDevice;
}

namespace experimental {
class GlobalCircularBuffer;
}

namespace detail {

class ProgramImpl;

// The Metal 1.0 part of building up a Program: the kernels, circular buffers and semaphores added through
// CreateKernel / CreateCircularBuffer / CreateSemaphore / Update*CircularBuffer* in host_api.hpp (and the
// ProgramDescriptor constructor).
//
// Owned by ProgramImpl by composition. It owns that state, the checks that only depend on it (kernel processor
// overlap, circular buffer index conflicts, semaphore id allocation) and the creation entry points. For the checks and
// invalidation that depend on the rest of the program (compiled / finalized state, DFBs, kernel groups, ephemeral
// allocation) it reaches back into its owning ProgramImpl, which is why ProgramImpl is not movable.
class Metal1ProgramBuilder {
public:
    using KernelMap = std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>;

    Metal1ProgramBuilder(ProgramImpl& owner, uint32_t programmable_core_count, uint32_t max_circular_buffers);

    // Kernel creation, one per kernel flavor in CreateKernel / CreateKernelFromString. Each validates the config
    // against the kernels already in the program, builds the kernel and adds it with add_kernel.
    KernelHandle create_data_movement_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DataMovementConfig& config);
    KernelHandle create_compute_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config);
    KernelHandle create_ethernet_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const EthernetConfig& config);
    KernelHandle create_dram_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DramConfig& config);
    // Adds an already-built kernel (also used by the other kernel flavors and Metal 2.0 lowering).
    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& core_type);

    // Kernels.
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;
    size_t num_kernels() const;
    // Programmable core type index -> KernelHandle -> Kernel
    std::vector<KernelMap>& kernels() { return kernels_; }
    const std::vector<KernelMap>& kernels() const { return kernels_; }

    // Circular buffers.
    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);
    // Adds an already-built CB (the ProgramDescriptor path); returns its handle.
    CBHandle add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    // TT_THROWs if no CB has this handle.
    std::shared_ptr<CircularBufferImpl> get_circular_buffer(CBHandle cb_handle) const;
    void update_circular_buffer_total_size(CBHandle cb_handle, uint32_t total_size);
    void update_circular_buffer_page_size(CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size);
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const Buffer& buffer);
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const Buffer& buffer, uint32_t address_offset);
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const MeshTensor& tensor);
    void update_dynamic_circular_buffer_address(
        CBHandle cb_handle, const experimental::GlobalCircularBuffer& global_circular_buffer);
    void update_dynamic_circular_buffer_address_and_total_size(
        CBHandle cb_handle, const Buffer& buffer, uint32_t total_size);
    void update_dynamic_circular_buffer_address_and_total_size(
        CBHandle cb_handle, const MeshTensor& tensor, uint32_t total_size);
    const std::vector<std::shared_ptr<CircularBufferImpl>>& circular_buffers() const { return circular_buffers_; }
    const std::unordered_map<CBHandle, std::shared_ptr<CircularBufferImpl>>& circular_buffer_by_id() const {
        return circular_buffer_by_id_;
    }
    using CoreCBIndices = std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>>;
    const CoreCBIndices& per_core_local_cb_indices() const { return per_core_local_cb_indices_; }
    const CoreCBIndices& per_core_remote_cb_indices() const { return per_core_remote_cb_indices_; }

    // Semaphores.
    // Allocate the next free semaphore ID on the given cores and insert the semaphore.
    // Returns the allocated ID; TT_FATALs if no slot can be found.
    uint32_t create_semaphore(const CoreRangeSet& crs, uint32_t initial_value, CoreType core_type);
    uint32_t create_semaphore(
        const std::variant<CoreRange, CoreRangeSet>& core_spec, uint32_t initial_value, CoreType core_type);
    // Insert a semaphore at the caller-specified ID. Used by:
    //   - Construction paths where the ID is already decided upstream (ProgramDescriptor ctor path).
    //   - create_semaphore above
    // The ProgramDescriptor path is something of a hack; it will eventually be deprecated in favor of Metal 2.0.
    void add_semaphore(const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);
    // Gives the program ownership of the buffer: it stays on device at least until the program is enqueued.
    void assign_global_buffer(const std::shared_ptr<Buffer>& buffer);
    // Global semaphores belong to a mesh device rather than a program, so this does not use any program state.
    static GlobalSemaphore create_global_semaphore(
        distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type);
    const std::vector<Semaphore>& semaphores() const { return semaphores_; }
    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(
        const CoreCoord& core, CoreType core_type) const;

private:
    // Inserts into the kernel map for the core type; TT_FATALs if the kernel overlaps another kernel that uses the
    // same processor.
    KernelHandle insert_kernel(const std::shared_ptr<Kernel>& kernel, uint32_t programmable_core_type_index);
    // Throws if any of the CB's buffer indices exceeds the per-core maximum.
    void validate_circular_buffer_indices(const CircularBufferImpl& circular_buffer) const;
    // Marks the CB's indices as used on each of its cores (throws on a conflict) and takes ownership of it.
    CBHandle register_circular_buffer(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    // Validates that a semaphore ID is within bounds and not already in use on overlapping cores
    void validate_semaphore_id(const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const;
    void insert_semaphore(
        const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, uint32_t l1_alignment, CoreType core_type);
    // The lowest semaphore ID that is free on every given core; TT_FATALs if there is none.
    uint32_t allocate_semaphore_id(const CoreRangeSet& crs, CoreType core_type) const;

    ProgramImpl& owner_;
    uint32_t max_circular_buffers_;

    std::vector<KernelMap> kernels_;

    std::vector<std::shared_ptr<CircularBufferImpl>> circular_buffers_;
    std::unordered_map<CBHandle, std::shared_ptr<CircularBufferImpl>> circular_buffer_by_id_;
    // Tracks which circular buffer indices are being used
    CoreCBIndices per_core_cb_indices_;
    CoreCBIndices per_core_local_cb_indices_;
    CoreCBIndices per_core_remote_cb_indices_;

    std::vector<Semaphore> semaphores_;
};

}  // namespace detail
}  // namespace tt::tt_metal

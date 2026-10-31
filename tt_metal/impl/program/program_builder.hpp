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

// Builds the kernels, circular buffers and semaphores of a program for the Metal 1.0 host API.
// invariants:
//  - A builder belongs to exactly one ProgramImpl and lives as long as it.
class Metal1ProgramBuilder {
public:
    using KernelMap = std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>;

    Metal1ProgramBuilder(ProgramImpl& owner, uint32_t programmable_core_count, uint32_t max_circular_buffers);

    // Creates a data movement kernel on `core_range_set` and adds it to the program.
    // precondition: the program is not compiled.
    // postcondition: the returned handle names the new kernel.
    KernelHandle create_data_movement_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DataMovementConfig& config);
    // Creates a compute kernel on `core_range_set` and adds it to the program.
    // precondition: the program is not compiled.
    // postcondition: the returned handle names the new kernel.
    KernelHandle create_compute_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config);
    // Creates an ethernet kernel on `core_range_set` and adds it to the program.
    // precondition: the program is not compiled.
    // postcondition: the returned handle names the new kernel.
    KernelHandle create_ethernet_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const EthernetConfig& config);
    // Creates a DRAM kernel on `core_range_set` and adds it to the program.
    // precondition: the program is not compiled.
    // postcondition: the returned handle names the new kernel.
    KernelHandle create_dram_kernel(
        const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DramConfig& config);
    // Adds an already built kernel to the program.
    // precondition: the program is not compiled.
    // precondition: no kernel on the same processor already runs on a core of `kernel`.
    // side-effect: the kernel groups of `core_type` are rebuilt on next use.
    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType& core_type);
    // precondition: a kernel with this handle exists.
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;
    size_t num_kernels() const;
    // Kernels of the program, indexed by programmable core type index.
    std::vector<KernelMap>& kernels() { return kernels_; }
    const std::vector<KernelMap>& kernels() const { return kernels_; }

    // Creates a circular buffer on `core_range_set` and adds it to the program.
    // precondition: the program is not compiled and has no dataflow buffer.
    // precondition: no buffer index of `config` is in use on a core of `core_range_set`.
    // side-effect: the circular buffer layout is computed again on next use.
    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    // Same as above, and the circular buffer is backed by `global_circular_buffer`.
    // precondition: the program has no CrossNodeDFB and no PrefetcherPipe.
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);
    // Adds an already built circular buffer to the program.
    CBHandle add_circular_buffer_(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    // precondition: a circular buffer with this handle exists.
    std::shared_ptr<CircularBufferImpl> get_circular_buffer(CBHandle cb_handle) const;
    // side-effect: the circular buffer layout is computed again on next use, unless the buffer is globally allocated.
    void update_circular_buffer_total_size(CBHandle cb_handle, uint32_t total_size);
    void update_circular_buffer_page_size(CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size);
    // precondition: the circular buffer is not backed by a GlobalCircularBuffer.
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const Buffer& buffer);
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const Buffer& buffer, uint32_t address_offset);
    void update_dynamic_circular_buffer_address(CBHandle cb_handle, const MeshTensor& tensor);
    // precondition: the circular buffer is backed by a GlobalCircularBuffer.
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
    const std::vector<Semaphore>& semaphores() const { return semaphores_; }
    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(
        const CoreCoord& core, CoreType core_type) const;

    // Gives the program ownership of `buffer`.
    // postcondition: the buffer stays on device at least until the program is enqueued.
    void assign_global_buffer(const std::shared_ptr<Buffer>& buffer);
    // Creates a global semaphore on `cores` of `device`.
    // postcondition: the semaphore has `initial_value` on every core of `cores`.
    static GlobalSemaphore create_global_semaphore(
        distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type);

private:
    KernelHandle insert_kernel(const std::shared_ptr<Kernel>& kernel, uint32_t programmable_core_type_index);
    void validate_circular_buffer_indices(const CircularBufferImpl& circular_buffer) const;
    CBHandle register_circular_buffer(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    // Validates that a semaphore ID is within bounds and not already in use on overlapping cores
    void validate_semaphore_id(const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const;
    void insert_semaphore(
        const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, uint32_t l1_alignment, CoreType core_type);
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

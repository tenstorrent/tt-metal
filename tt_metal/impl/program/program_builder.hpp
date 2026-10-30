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
#include <vector>

#include "impl/buffers/semaphore.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

class CircularBufferImpl;
class Kernel;

namespace detail {

// The Metal 1.0 part of building up a Program: the kernels, circular buffers and semaphores added through
// CreateKernel / CreateCircularBuffer / CreateSemaphore (and the ProgramDescriptor constructor).
//
// Owned by ProgramImpl by composition. It owns that state and the checks that only depend on it (kernel processor
// overlap, circular buffer index conflicts, semaphore id allocation). Checks and invalidation that depend on the
// rest of the program (compiled / finalized state, DFBs, kernel groups, ephemeral allocation) stay in ProgramImpl.
class Metal1ProgramBuilder {
public:
    using KernelMap = std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>;

    Metal1ProgramBuilder(uint32_t programmable_core_count, uint32_t max_circular_buffers);

    // Kernels. `programmable_core_type_index` selects the per-core-type kernel map.
    // TT_FATALs if the kernel overlaps another kernel that uses the same processor.
    KernelHandle add_kernel(const std::shared_ptr<Kernel>& kernel, uint32_t programmable_core_type_index);
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;
    size_t num_kernels() const;
    // Programmable core type index -> KernelHandle -> Kernel
    std::vector<KernelMap>& kernels() { return kernels_; }
    const std::vector<KernelMap>& kernels() const { return kernels_; }

    // Circular buffers.
    // Throws if any of the CB's buffer indices exceeds the per-core maximum.
    void validate_circular_buffer_indices(const CircularBufferImpl& circular_buffer) const;
    // Marks the CB's indices as used on each of its cores (throws on a conflict) and takes ownership of it.
    CBHandle add_circular_buffer(const std::shared_ptr<CircularBufferImpl>& circular_buffer);
    const std::vector<std::shared_ptr<CircularBufferImpl>>& circular_buffers() const { return circular_buffers_; }
    const std::unordered_map<CBHandle, std::shared_ptr<CircularBufferImpl>>& circular_buffer_by_id() const {
        return circular_buffer_by_id_;
    }
    using CoreCBIndices = std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>>;
    const CoreCBIndices& per_core_local_cb_indices() const { return per_core_local_cb_indices_; }
    const CoreCBIndices& per_core_remote_cb_indices() const { return per_core_remote_cb_indices_; }

    // Semaphores.
    // Validates that a semaphore ID is within bounds and not already in use on overlapping cores
    void validate_semaphore_id(const CoreRangeSet& crs, uint32_t semaphore_id, CoreType core_type) const;
    void add_semaphore(
        const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, uint32_t l1_alignment, CoreType core_type);
    // The lowest semaphore ID that is free on every given core; TT_FATALs if there is none.
    uint32_t allocate_semaphore_id(const CoreRangeSet& crs, CoreType core_type) const;
    const std::vector<Semaphore>& semaphores() const { return semaphores_; }
    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(
        const CoreCoord& core, CoreType core_type) const;

private:
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

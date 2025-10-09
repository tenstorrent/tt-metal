// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt_stl/assert.hpp>

#include <cstdint>
#include <vector>

namespace tt::tt_fabric {

/**
 * Base interface class for fabric channel allocators.
 * Responsible for creating L1 memory allocations for fabric routers.
 */
class FabricChannelAllocator {
public:
    /**
     * Constructor that takes a list of memory regions (start, end address pairs).
     * @param memory_regions Vector of memory regions available for allocation
     */
    explicit FabricChannelAllocator(
        tt::tt_fabric::Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        const std::vector<MemoryRegion>& memory_regions);

    virtual ~FabricChannelAllocator() = default;

    /**
     * Emit compile-time arguments for the fabric router.
     * Fills the provided vector with uint32_t values representing the allocation configuration.
     * @param ct_args Vector to be filled with compile-time arguments
     */
    virtual void emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const = 0;

    /**
     * Get the total available memory size across all regions.
     * @return Total size in bytes
     */
    size_t get_total_available_memory() const;

    /**
     * Get the number of memory regions.
     * @return Number of regions
     */
    size_t get_num_regions() const { return memory_regions_.size(); }

    /**
     * Get a specific memory region.
     * @param index Region index
     * @return Reference to the memory region
     */
    const MemoryRegion& get_region(size_t index) const {
        TT_FATAL(index < memory_regions_.size(), "Region index {} out of bounds", index);
        return memory_regions_[index];
    }

protected:
    tt::tt_fabric::Topology topology_;
    tt::tt_fabric::FabricEriscDatamoverOptions options_;
    std::vector<MemoryRegion> memory_regions_;
};

/**
 * Elastic channels allocator implementation.
 * Dynamically allocates channels based on available memory and demand.
 */
class ElasticChannelsAllocator : public FabricChannelAllocator {
public:
    /**
     * Constructor for elastic channels allocator.
     * @param memory_regions Available memory regions
     * @param buffer_size_bytes Size of each buffer in bytes
     * @param min_buffers_per_channel Minimum buffers per channel
     * @param max_buffers_per_channel Maximum buffers per channel
     */
    ElasticChannelsAllocator(
        tt::tt_fabric::Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        const std::vector<MemoryRegion>& memory_regions);

    void emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const override;
};

}  // namespace tt::tt_fabric

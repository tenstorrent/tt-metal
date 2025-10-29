// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt_stl/assert.hpp>

#include <cstdint>
#include <vector>
#include <ostream>
#include <sstream>
#include <fmt/format.h>

namespace tt::tt_fabric {

/**
 * Base interface class for fabric channel allocators.
 * Responsible for creating L1 memory allocations for fabric router sender and receiver channels.
 * Invoked per router bank so for routers parallelized across multiple cores, multiple of these will be invoked.
 */
class FabricChannelAllocator {
public:
    /**
     * Constructor that takes a list of memory regions (start, end address pairs).
     * @param topology Fabric topology
     * @param options Fabric erisc datamover options
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

    /**
     * Virtual method to format allocator state to an output stream.
     * Derived classes should override this to provide specific formatting.
     * @param os Output stream
     */
    virtual void print(std::ostream& os) const = 0;

    // Stream output operator for logging
    friend std::ostream& operator<<(std::ostream& os, const FabricChannelAllocator& allocator) {
        allocator.print(os);
        return os;
    }

protected:
    tt::tt_fabric::Topology topology_;
    tt::tt_fabric::FabricEriscDatamoverOptions options_;
    std::vector<MemoryRegion> memory_regions_;
};

/**
 * Elastic channels allocator implementation.
 * The `ElasticChannelsAllocator` allocates chunks of memory in a memory pool for elastic channels.
 * The memory pool does not need to be contiguous, but each individual chunk within the pool must be
 * contiguous.
 * Each chunk is a sequence of 1 or more buffer slots (i.e. packet slots)
 */
class ElasticChannelsAllocator : public FabricChannelAllocator {
public:
    /**
     * Constructor for elastic channels allocator.
     * @param topology Fabric topology
     * @param options Fabric erisc datamover options
     * @param memory_regions Available memory regions
     * @param buffer_slot_size_bytes Size of each buffer slot in bytes, including the packet header
     * @param min_buffers_per_chunk Minimum buffer slots per chunk
     * @param max_buffers_per_chunk Maximum buffer slots per chunk
     */
    ElasticChannelsAllocator(
        tt::tt_fabric::Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        const std::vector<MemoryRegion>& memory_regions,
        size_t buffer_slot_size_bytes,
        size_t min_buffers_per_chunk,
        size_t max_buffers_per_chunk
        );

    void emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const override;

    void print(std::ostream& os) const override { os << "ElasticChannelsAllocator (not yet fully implemented)"; }
};

}  // namespace tt::tt_fabric

// fmt formatter specialization for FabricChannelAllocator
template <>
struct fmt::formatter<tt::tt_fabric::FabricChannelAllocator> : fmt::formatter<std::string> {
    auto format(const tt::tt_fabric::FabricChannelAllocator& allocator, fmt::format_context& ctx) const {
        std::ostringstream stream;
        stream << allocator;
        return formatter<std::string>::format(stream.str(), ctx);
    }
};

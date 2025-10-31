// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

#include "tt_metal/api/tt-metalium/fabric_types.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include <vector>

namespace tt::tt_fabric {


/**
 * Memory region definition for fabric channel allocation.
 * Represents a contiguous memory region with start and size.
 */
struct MemoryRegion {
    size_t start_address;
    size_t size;

    MemoryRegion(size_t start, size_t size) : start_address(start), size(size) {
    }

    size_t get_size() const { return size; }
    size_t get_start_address() const { return start_address; }
    size_t get_end_address() const { return start_address + size; }
    bool contains(size_t address) const { return address >= start_address && address < get_end_address(); }
};

enum class FabricEriscDatamoverType {
    Default = 0,
    Dateline = 1,
    DatelineUpstream = 2,
    DatelineUpstreamAdjacentDevice = 3,
    Invalid = 4,
};

enum class FabricEriscDatamoverAxis : std::size_t {
    Short = 0,
    Long = 1,
    Invalid = 2,
};

struct FabricRouterBufferConfig {
    bool enable_dateline_sender_extra_buffer_slots = false;
    bool enable_dateline_receiver_extra_buffer_slots = false;
    bool enable_dateline_upstream_sender_extra_buffer_slots = false;
    bool enable_dateline_upstream_receiver_extra_buffer_slots = false;
    bool enable_dateline_upstream_adjacent_sender_extra_buffer_slots = false;
};

// enable extra buffer slots configuration based on sender/receiver channel and EDM type.
struct FabricEriscDatamoverOptions {
    FabricEriscDatamoverType edm_type = FabricEriscDatamoverType::Default;
    FabricEriscDatamoverAxis edm_axis = FabricEriscDatamoverAxis::Short;
    FabricRouterBufferConfig edm_buffer_config = FabricRouterBufferConfig{};
    FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED;
    eth_chan_directions direction = eth_chan_directions::EAST;  // only used by 2D to get the correct router direction
};

namespace builder_config {
// linear/mesh: for fabric with tensix extension, only one sender channel will be present on fabric router
// ring/torus: for fabric with tensix extension, two sender channel will be present on fabric router (vc0 and vc1)
static constexpr std::size_t num_sender_channels_with_tensix_config = 1;
static constexpr std::size_t num_sender_channels_with_tensix_config_deadlock_avoidance = 2;

// num sender channels based on more accurate topology
static constexpr std::size_t num_sender_channels_1d_linear = 2;
static constexpr std::size_t num_sender_channels_2d_mesh = 4;
static constexpr std::size_t num_sender_channels_1d_ring = 3;
static constexpr std::size_t num_sender_channels_2d_torus = 5;

static constexpr std::size_t num_sender_channels_1d = 3;
static constexpr std::size_t num_sender_channels_2d = 5;
static constexpr std::size_t num_sender_channels = std::max(num_sender_channels_1d, num_sender_channels_2d);
static constexpr std::size_t num_downstream_sender_channels = num_sender_channels - 1;

static constexpr std::size_t num_receiver_channels = 2;

static constexpr std::size_t num_downstream_edms_vc0 = 1;
static constexpr std::size_t num_downstream_edms_2d_vc0 = 4;
static constexpr std::size_t num_downstream_edms_vc1 = 1;
static constexpr std::size_t num_downstream_edms = num_downstream_edms_vc0 + num_downstream_edms_vc1;
static constexpr std::size_t num_downstream_edms_2d = num_downstream_edms_2d_vc0 + num_downstream_edms_vc1;
static constexpr std::size_t max_downstream_edms = std::max(num_downstream_edms, num_downstream_edms_2d);


inline uint32_t get_sender_channel_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_sender_channels_2d : builder_config::num_sender_channels_1d;
}

inline uint32_t get_downstream_edm_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_downstream_edms_2d
                         : builder_config::num_downstream_edms;
}

inline uint32_t get_vc0_downstream_edm_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_downstream_edms_2d_vc0
                         : builder_config::num_downstream_edms_vc0;
}


}  // namespace builder_config

/**
 * Structure to hold all parameters needed for allocator construction.
 * This simplifies passing multiple parameters to allocator constructors.
 */
struct AllocatorConstructionParams {
    Topology topology;
    FabricEriscDatamoverOptions options;
    size_t num_used_sender_channels;
    size_t num_used_receiver_channels;
    size_t channel_buffer_size_bytes;
    size_t available_channel_buffering_space;
    std::vector<MemoryRegion> memory_regions;

    AllocatorConstructionParams(
        Topology topology,
        const FabricEriscDatamoverOptions& options,
        size_t num_used_sender_channels,
        size_t num_used_receiver_channels,
        size_t channel_buffer_size_bytes,
        size_t available_channel_buffering_space,
        const std::vector<MemoryRegion>& memory_regions) :
        topology(topology),
        options(options),
        num_used_sender_channels(num_used_sender_channels),
        num_used_receiver_channels(num_used_receiver_channels),
        channel_buffer_size_bytes(channel_buffer_size_bytes),
        available_channel_buffering_space(available_channel_buffering_space),
        memory_regions(memory_regions) {}
};

}  // namespace tt::tt_fabric

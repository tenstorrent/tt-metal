// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include <vector>
#include <algorithm>

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

// enable extra buffer slots configuration based on sender/receiver channel and EDM type.
struct FabricEriscDatamoverOptions {
    FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED;
    eth_chan_directions direction = eth_chan_directions::EAST;  // only used by 2D to get the correct router direction
};

namespace builder_config {
// Number of Virtual Channels supported (VC0 and VC1)
static constexpr std::size_t MAX_NUM_VCS = 2;

// linear/mesh/ring/torus: for fabric with tensix extension, only one sender channel will be present on fabric router
static constexpr std::size_t num_sender_channels_with_tensix_config = 1;

// num sender channels based on more accurate topology
static constexpr std::size_t num_sender_channels_1d_neighbor_exchange = 1;
static constexpr std::size_t num_sender_channels_1d_linear = 2;
static constexpr std::size_t num_sender_channels_2d_mesh = 4;

// Z router channel counts
// VC0: 5 sender channels (mesh→Z: 0=Worker, 1-4=E/W/N/S mesh directions) + 1 receiver
// VC1: 4 sender channels (Z→mesh, one per direction: 0=E, 1=W, 2=N, 3=S) + 0 receiver (skipped)
static constexpr std::size_t num_sender_channels_z_router_vc0 = 5;
static constexpr std::size_t num_sender_channels_z_router_vc1 = 4;
static constexpr std::size_t num_sender_channels_z_router = num_sender_channels_z_router_vc0 + num_sender_channels_z_router_vc1;
static constexpr std::size_t num_receiver_channels_z_router = 2;  // 1 for VC0, 1 for VC1

static constexpr std::size_t num_sender_channels_1d = 2;
// VC0: Worker + 3 of [N/E/S/W] = 4 channels
// VC1: Up to 3 of [N/E/S/W] for inter-mesh = 3 channels, 1 for Z→mesh
// Total 2D: 4 + 3 +1 = 8 channels
static constexpr std::size_t num_sender_channels_2d = 8;
static constexpr std::size_t num_max_sender_channels =
    std::max({num_sender_channels_1d, num_sender_channels_2d, num_sender_channels_z_router});
static constexpr std::size_t num_receiver_channels_1d = 1;
static constexpr std::size_t num_receiver_channels_2d = 2;
static constexpr std::size_t num_max_receiver_channels = std::max({num_receiver_channels_1d, num_receiver_channels_2d, num_receiver_channels_z_router});

static constexpr std::size_t num_downstream_edms_vc0 = 1;
static constexpr std::size_t num_downstream_edms_2d_vc0 = 3;
static constexpr std::size_t num_downstream_edms_2d_vc1 = 3;  // XY intermesh: 3 mesh directions
static constexpr std::size_t num_downstream_edms_2d_vc1_with_z = 4;  // Z intermesh: 3 mesh + Z
static constexpr std::size_t num_downstream_edms_1d = num_downstream_edms_vc0;
static constexpr std::size_t num_downstream_edms_2d = num_downstream_edms_2d_vc0 + num_downstream_edms_2d_vc1;
static constexpr std::size_t max_downstream_edms = 8;

// 2D mesh directions (N, E, S, W)
static constexpr uint32_t num_mesh_directions_2d = 4;

uint32_t get_sender_channel_count(bool is_2D_routing);

uint32_t get_receiver_channel_count(bool is_2D_routing);

uint32_t get_num_used_sender_channel_count(Topology topology);

uint32_t get_num_tensix_sender_channels(Topology topology, tt::tt_fabric::FabricTensixConfig fabric_tensix_config);

uint32_t get_downstream_edm_count(bool is_2D_routing);

uint32_t get_vc0_downstream_edm_count(bool is_2D_routing);

uint32_t get_vc1_downstream_edm_count(bool is_2D_routing);

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

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

#include "tt_metal/api/tt-metalium/fabric_types.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

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
}  // namespace builder_config
}  // namespace tt::tt_fabric

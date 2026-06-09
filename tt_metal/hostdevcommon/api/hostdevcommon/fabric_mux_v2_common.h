// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_fabric {

struct FabricMuxV2SharedTridRingHeader {
    uint32_t write_count = 0;
    uint32_t read_count = 0;
};

using FabricMuxV2SharedTridRingEntry = uint32_t;

inline constexpr uint32_t FABRIC_MUX_V2_SHARED_TRID_RING_HEADER_WORDS =
    sizeof(FabricMuxV2SharedTridRingHeader) / sizeof(uint32_t);
static_assert(FABRIC_MUX_V2_SHARED_TRID_RING_HEADER_WORDS == 2);

struct FabricMuxV2SharedControlBlock {
    uint32_t drain_initiated = 0;
    uint32_t forwarder_stop_tracking = 0;
    uint32_t forwarder_done = 0;
    uint32_t reserved_0 = 0;
    uint32_t reserved_1 = 0;
    uint32_t reserved_2 = 0;
};

inline constexpr uint32_t FABRIC_MUX_V2_SHARED_CONTROL_BLOCK_WORDS =
    sizeof(FabricMuxV2SharedControlBlock) / sizeof(uint32_t);
static_assert(FABRIC_MUX_V2_SHARED_CONTROL_BLOCK_WORDS == 6);

}  // namespace tt::tt_fabric

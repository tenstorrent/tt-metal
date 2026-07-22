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

static_assert(sizeof(FabricMuxV2SharedTridRingHeader) / sizeof(uint32_t) == 2);

struct FabricMuxV2SharedControlBlock {
    uint32_t drain_initiated = 0;
    uint32_t forwarder_stop_tracking = 0;
    uint32_t forwarder_done = 0;
};

static_assert(sizeof(FabricMuxV2SharedControlBlock) / sizeof(uint32_t) == 3);

}  // namespace tt::tt_fabric

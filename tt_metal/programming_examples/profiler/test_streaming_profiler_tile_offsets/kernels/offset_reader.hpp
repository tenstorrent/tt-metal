// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the tile offsets test (test_streaming_profiler_tile_offsets.cpp) and its reader kernel
// (offset_reader.cpp): the reader's scratch, at the idle eth unreserved base.

#pragma once

#include <cstdint>

#include "hostdev/streaming_profiler_sync.h"

namespace offset_reader {

constexpr uint32_t kMaxTargets = 256;
constexpr uint32_t kExit = 0xFFFFFFFFu;

struct Scratch {
    uint32_t landing[16];           // where the reads of the targets land
    uint32_t go;                    // host-written: the checkpoint to take, or kExit to leave
    uint32_t done;                  // the last checkpoint taken
    uint32_t targets[kMaxTargets];  // y << 16 | x
    kernel_profiler::TileNetPartner out[kMaxTargets];
    uint32_t hist[2 * kernel_profiler::kTileNetBins];
};

}  // namespace offset_reader

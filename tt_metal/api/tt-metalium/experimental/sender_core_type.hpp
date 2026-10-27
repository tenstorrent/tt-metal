// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::experimental {

// Sender domain for a GlobalCircularBuffer or a PrefetcherPipe. Worker = senders are worker cores
// hosting their own slice of the ring in L1. Dram = senders are programmable DRAM cores (Blackhole
// DRISCs) that keep their sender state in their own L1; the ring is sharded over receivers only,
// and receivers return credits to the DRAM core.
enum class SenderCoreType : uint8_t {
    Worker = 0,
    Dram = 1,
};

}  // namespace tt::tt_metal::experimental

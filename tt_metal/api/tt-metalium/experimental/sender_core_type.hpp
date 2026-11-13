// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::experimental {

// Sender domain for a GlobalCircularBuffer or a PrefetcherPipe. Worker = senders are worker cores
// hosting their own slice of the ring in L1. Dram = senders are DRAM-bank endpoints the runtime
// runs (on Blackhole, programmable DRAM cores), named by DRAM-logical coordinate and keeping their
// sender state outside the worker grid; the ring is sharded over receivers only, and receivers
// return credits to wherever that state lives.
enum class SenderCoreType : uint8_t {
    Worker = 0,
    Dram = 1,
};

}  // namespace tt::tt_metal::experimental

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace tt::tt_fabric {

constexpr uint32_t RING_TERMINAL_OFFLOAD_MAX_DEPTH = 8;

struct alignas(16) RingTerminalOffloadDescriptor {
    uint32_t payload_l1_address = 0;
    uint32_t payload_size_bytes = 0;
    uint32_t destination_noc_address_low = 0;
    uint32_t destination_noc_address_high = 0;
};

// Shared by the two RISCs on one Blackhole Ethernet core. ERISC1 publishes
// descriptors after receiving a write-and-forward packet. ERISC0 completes
// the terminal payload copy on NOC0 and advances completed_sequence only after
// the corresponding NOC transaction has flushed.
struct alignas(16) RingTerminalOffloadQueue {
    uint32_t published_sequence = 0;
    uint32_t completed_sequence = 0;
    uint32_t depth = 0;
    uint32_t reserved = 0;
    std::array<RingTerminalOffloadDescriptor, RING_TERMINAL_OFFLOAD_MAX_DEPTH> descriptors = {};
};

static_assert(sizeof(RingTerminalOffloadDescriptor) == 16);
static_assert(sizeof(RingTerminalOffloadQueue) == 16 + 16 * RING_TERMINAL_OFFLOAD_MAX_DEPTH);

}  // namespace tt::tt_fabric

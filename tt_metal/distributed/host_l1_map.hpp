// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Per-core L1 layout. Smaller than the register-file era: the sockets' own bytes_sent /
// bytes_acked pair replaces rdma_signal and rdma_completion.
#pragma once

#include <cstdint>
#include <string>

#include "tt_metal/distributed/host_uva_frame.hpp"

namespace tt::tt_metal::experimental {

struct L1MapNew {
    // A word a kernel polls gets its own line, so setting one never disturbs a neighbour.
    static constexpr uint32_t kDoorbellBytes = 64;
    // The pull kernel reads this back as one uint64_t, so 8 is exact, not a round-up.
    static constexpr uint32_t kDestWordBytes = 8;

    uint32_t l1_size = 0;
    // Signal offsets go on the wire relative to this, so both kernels need it.
    uint32_t l1_base = 0;

    uint32_t payload_addr = 0;    // what the sender kernel puts
    uint32_t stage_addr = 0;      // kFrameTrailerBytes; where the trailer is assembled
    uint32_t stop_addr = 0;       // where a DEVICE_PULL receiver kernel is told to exit
    uint32_t consumed_addr = 0;   // what the far device has pulled, for tt_uva_sync()
    uint32_t dest_word_addr = 0;  // a store's per-message destination, for the pull kernel
    uint32_t verify_addr = 0;     // [0..1] t_begin, [2..3] t_end, [4] iterations, [5..6] t_steady
    uint32_t deliver_addr = 0;    // where the pull kernel lands payload

    // Delivery gets its own buffer so a core can hold an outbound and an inbound payload at
    // once. Costs a second payload of L1, halving the largest that fits.
    bool bidirectional = false;
    uint32_t deliver_end = 0;

    static L1MapNew compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes, bool bidirectional = false);

    uint32_t payload_copies() const { return bidirectional ? 2u : 1u; }
    uint32_t end() const { return bidirectional ? deliver_end : (verify_addr + kDoorbellBytes); }
    // Everything stacked above the payload. Independent of payload_bytes by construction.
    uint32_t control_bytes() const { return verify_addr + kDoorbellBytes - stage_addr; }

    std::string fits(uint32_t payload_bytes) const;
    std::string describe() const;
};

}  // namespace tt::tt_metal::experimental

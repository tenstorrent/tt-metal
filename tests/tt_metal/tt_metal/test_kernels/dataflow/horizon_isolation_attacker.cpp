// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Horizon isolation attacker (runs on the Tensix of one device of a split build). Issues POSTED
// NOC writes of a marker to raw horizon_2x3 REMOTE-window addresses:
//   - selector 0 (this device's own Tensix) at control_addr, before and after the attack writes,
//     to prove the writes leave the core and the NOC still works afterwards;
//   - the selectors that are parked on a split device at target_addr, which must reach no tile.
// Nothing here waits for an ack: a non-posted access to a parked selector never completes.

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
// horizon_2x3 REMOTE window: compare 0x10_0000_0000, selector in bits [31:26].
constexpr uint64_t REMOTE_WINDOW = 0x1000000000ull;
constexpr uint32_t SELECTOR_SHIFT = 26;
constexpr uint32_t PARKED_SELECTORS[] = {1, 3, 5, 6, 63};
constexpr uint32_t NUM_PARKED = sizeof(PARKED_SELECTORS) / sizeof(PARKED_SELECTORS[0]);
constexpr uint32_t WORDS = 4;  // 16-byte payload per write
constexpr uint32_t SLOT = 64;  // one cache line per source payload

uint64_t remote(uint32_t selector, uint32_t addr) {
    return REMOTE_WINDOW | (static_cast<uint64_t>(selector) << SELECTOR_SHIFT) | addr;
}

// Fill a source slot with marker | i and flush it to L1 so the NOC reads the new data.
uint32_t stage(uint32_t src_base, uint32_t slot, uint32_t marker) {
    const uint32_t src = src_base + slot * SLOT;
    CoreLocalMem<uint32_t> buf(src);
    for (uint32_t i = 0; i < WORDS; i++) {
        buf[i] = marker | i;
    }
    flush_l2_cache_line(src);
    return src;
}
}  // namespace

void kernel_main() {
    const uint32_t src_base = get_arg(args::src_addr);
    const uint32_t target_addr = get_arg(args::target_addr);
    const uint32_t control_addr = get_arg(args::control_addr);
    const uint32_t marker = get_arg(args::marker);
    constexpr uint32_t bytes = WORDS * sizeof(uint32_t);

    // Positive control before: own Tensix through REMOTE selector 0.
    uint32_t slot = 0;
    uint32_t src = stage(src_base, slot++, marker | 0xC00);
    noc_async_write_one_packet<true, /*posted=*/true>(src, remote(0, control_addr), bytes);

    // The attack: the same posted write to every parked selector.
    for (uint32_t i = 0; i < NUM_PARKED; i++) {
        src = stage(src_base, slot++, marker | (PARKED_SELECTORS[i] << 4));
        noc_async_write_one_packet<true, /*posted=*/true>(src, remote(PARKED_SELECTORS[i], target_addr), bytes);
    }

    // Positive control after: the NOC still delivers once the parked writes are in flight.
    src = stage(src_base, slot++, marker | 0xC10);
    noc_async_write_one_packet<true, /*posted=*/true>(src, remote(0, control_addr + bytes), bytes);

    noc_async_posted_writes_flushed();
}

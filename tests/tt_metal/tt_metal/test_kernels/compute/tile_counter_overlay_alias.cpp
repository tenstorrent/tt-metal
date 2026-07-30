// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"
#include "ckernel_trisc_common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"

namespace {
constexpr std::uint32_t kTensixOnlyTc = 16;
constexpr std::uint32_t kMaxSteps = 4;

// A counter register reads back a derived view rather than the credit total that was written:
// tile_counters[].f.posted sits at the TILES_AVAILABLE offset and tile_counters[].f.acked at
// SPACE_AVAILABLE. So a push raises .posted by the tiles pushed only while they are unconsumed, and a pop
// raises .acked toward buf_capacity. Sampling right after the push/pop instruction returns the pre-update
// value, so each sample is taken after a bounded series of re-reads.
constexpr std::uint32_t kSettleReads = 256;

// L1 scratch layout (uncached), one writer per word. Must match the host test.
//   [0]        pack: tc16 reset + capacity programmed  (handshake, host-initialized to 0)
//   [1 .. 4]   pack: tiles available after each push
//   [5 .. 8]   unpack: space available after each pop
//   [9]        pack: every push issued                 (handshake, host-initialized to 0)
//   [10]       pack: buf_capacity read back
//   [11]       pack: step count decoded from runtime args
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kTilesAvailableBaseIdx = 1;
constexpr std::uint32_t kSpaceAvailableBaseIdx = kTilesAvailableBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kSpaceAvailableBaseIdx + kMaxSteps;
constexpr std::uint32_t kCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kCapacityIdx + 1;

volatile std::uint32_t* scratch_ptr(std::uint32_t l1_address, std::uint32_t word_idx) {
    return reinterpret_cast<volatile std::uint32_t*>(
        l1_address + MEM_L1_UNCACHED_BASE + word_idx * sizeof(std::uint32_t));
}

std::uint32_t settled_tiles_available() {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[kTensixOnlyTc].f.posted;
    }
    return value;
}

std::uint32_t settled_space_available() {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[kTensixOnlyTc].f.acked;
    }
    return value;
}

std::uint32_t settled_capacity() {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[kTensixOnlyTc].f.buf_capacity;
    }
    return value;
}
}  // namespace

void kernel_main() {
    const std::uint32_t l1_address = get_arg(args::l1_address);
    const std::uint32_t capacity = get_arg(args::capacity);
    const std::uint32_t step_tiles[kMaxSteps] = {
        get_arg(args::step0), get_arg(args::step1), get_arg(args::step2), get_arg(args::step3)};

    // A zero-sized step ends the series, so a case can use fewer than kMaxSteps.
    std::uint32_t num_steps = 0;
    while (num_steps < kMaxSteps && step_tiles[num_steps] != 0) {
        num_steps++;
    }

#ifdef TRISC_PACK
    // Keep the trigger independent of DFB allocation. The bug is in the RTL
    // T6->overlay update path, so one direct legal T6 counter update is enough.
    auto* remapper_control = reinterpret_cast<volatile std::uint32_t*>(REMAP_GLOBAL_CONTROL_REG_ADDR32);
    *remapper_control = 1;
    while ((*remapper_control & 1) == 0) {
    }

    const std::uint32_t overlay_tc0_before = overlay::llk_intf_get_posted(0, 0);

    ckernel::trisc::tile_counters[kTensixOnlyTc].f.reset = 1;
    ckernel::trisc::tile_counters[kTensixOnlyTc].f.buf_capacity = capacity;
    *scratch_ptr(l1_address, kCapacityIdx) = settled_capacity();
    *scratch_ptr(l1_address, kNumStepsIdx) = num_steps;
    *scratch_ptr(l1_address, kReadyIdx) = 1;

    DPRINT(
        "BEFORE: T6 TC{} private, capacity={}, steps={}; overlay TC0 posted={}\n",
        kTensixOnlyTc,
        capacity,
        num_steps,
        overlay_tc0_before);

    // Every push is issued before unpack is allowed to pop, so tiles available is the running total of
    // pushed tiles and a doubled credit shows up immediately as 2x the expected value.
    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];
        TT_WAIT_FREE(ckernel::p_stall::STALL_PACK, num_tiles, kTensixOnlyTc);
        TT_PUSH_TILES(/*PACK_SEL=*/0x1, num_tiles, kTensixOnlyTc);
        const std::uint32_t tiles_available = settled_tiles_available();
        *scratch_ptr(l1_address, kTilesAvailableBaseIdx + step) = tiles_available;
        DPRINT("PACK step{} push={} tiles_available={}\n", step, num_tiles, tiles_available);
    }
    *scratch_ptr(l1_address, kPushesDoneIdx) = 1;

    const std::uint32_t overlay_tc0_after = overlay::llk_intf_get_posted(0, 0);
    DPRINT(
        "AFTER:  overlay TC0 posted={} (expected {}; a change means TC16 reached overlay TC0)\n",
        overlay_tc0_after,
        overlay_tc0_before);
#endif

#ifdef TRISC_UNPACK
    while (*scratch_ptr(l1_address, kReadyIdx) == 0) {
    }
    // Wait for the full push series so space available is a deterministic function of tiles popped.
    while (*scratch_ptr(l1_address, kPushesDoneIdx) == 0) {
    }

    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];
        TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, kTensixOnlyTc);
        TT_POP_TILES(/*UNPACK_SEL=*/0x3, num_tiles, kTensixOnlyTc);
        const std::uint32_t space_available = settled_space_available();
        *scratch_ptr(l1_address, kSpaceAvailableBaseIdx + step) = space_available;
        DPRINT("UNPACK step{} pop={} space_available={}\n", step, num_tiles, space_available);
    }
#endif
}
